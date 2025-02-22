print("Script is starting...")

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import ReduceLROnPlateau

class ImprovedDiseaseNet(nn.Module):
    def __init__(self, num_classes=7, dropout_rate=0.5):
        super(ImprovedDiseaseNet, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32),
            nn.Conv2d(32, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64),
            nn.Conv2d(64, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.4),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 128, kernel_size=3, padding=1, groups=128),
            nn.Conv2d(128, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate)
        )
        
        self.detection_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.features(x)
        disease_prob = self.detection_head(features)
        disease_class = self.classification_head(features)
        return disease_prob, disease_class

class StrawberryDiseaseDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
        
        self.label_map = {
            'Angular Leafspot': 0,
            'Anthracnose Fruit Rot': 1,
            'Blossom Blight': 2,
            'Gray Mold': 3,
            'Leaf Spot': 4,
            'Powdery Mildew Fruit': 5,
            'Powdery Mildew Leaf': 6
        }

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        json_path = os.path.join(self.label_dir, img_name.replace('.jpg', '.json'))
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        with open(json_path, 'r') as f:
            label_data = json.load(f)
        
        has_disease = torch.tensor(len(label_data['shapes']) > 0, dtype=torch.float32)
        
        disease_class = 0
        if len(label_data['shapes']) > 0:
            disease_name = label_data['shapes'][0]['label']
            disease_class = self.label_map[disease_name]
        
        return image, has_disease, torch.tensor(disease_class)

def train_model(model, train_loader, val_loader, num_epochs=30, device='cpu', patience=5):
    criterion_detection = nn.BCELoss()
    criterion_classification = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )
    
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        verbose=True
    )
    
    model = model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_detection_correct = 0
        train_classification_correct = 0
        train_total = 0
        
        for images, has_disease, disease_class in train_loader:
            images = images.to(device)
            has_disease = has_disease.to(device)
            disease_class = disease_class.to(device)
            
            optimizer.zero_grad()
            
            disease_prob, disease_pred = model(images)
            
            detection_loss = criterion_detection(disease_prob.squeeze(), has_disease)
            classification_loss = criterion_classification(disease_pred, disease_class)
            
            total_loss = detection_loss + classification_loss
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += total_loss.item()
            
            predicted_disease = (disease_prob.squeeze() > 0.5).float()
            train_detection_correct += (predicted_disease == has_disease).sum().item()
            
            _, predicted_class = torch.max(disease_pred, 1)
            train_classification_correct += (predicted_class == disease_class).sum().item()
            train_total += has_disease.size(0)
        
        model.eval()
        val_loss = 0.0
        val_detection_correct = 0
        val_classification_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, has_disease, disease_class in val_loader:
                images = images.to(device)
                has_disease = has_disease.to(device)
                disease_class = disease_class.to(device)
                
                disease_prob, disease_pred = model(images)
                
                detection_loss = criterion_detection(disease_prob.squeeze(), has_disease)
                classification_loss = criterion_classification(disease_pred, disease_class)
                val_loss += (detection_loss + classification_loss).item()
                
                predicted_disease = (disease_prob.squeeze() > 0.5).float()
                val_detection_correct += (predicted_disease == has_disease).sum().item()
                
                _, predicted_class = torch.max(disease_pred, 1)
                val_classification_correct += (predicted_class == disease_class).sum().item()
                val_total += has_disease.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        print(f'Epoch {epoch+1}:')
        print(f'Training Loss: {running_loss/len(train_loader):.4f}')
        print(f'Training Detection Accuracy: {100 * train_detection_correct/train_total:.2f}%')
        print(f'Training Classification Accuracy: {100 * train_classification_correct/train_total:.2f}%')
        print(f'Validation Loss: {avg_val_loss:.4f}')
        print(f'Validation Detection Accuracy: {100 * val_detection_correct/val_total:.2f}%')
        print(f'Validation Classification Accuracy: {100 * val_classification_correct/val_total:.2f}%')
        
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            model.load_state_dict(best_model_state)
            break

def k_fold_cross_validation(dataset, k=5, num_epochs=30, device='cpu'):
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_results = []
    
    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        print(f'\nFold {fold + 1}/{k}')
        
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
        
        train_loader = DataLoader(dataset, batch_size=32, sampler=train_subsampler)
        val_loader = DataLoader(dataset, batch_size=32, sampler=val_subsampler)
        
        model = ImprovedDiseaseNet()
        train_model(model, train_loader, val_loader, num_epochs=num_epochs, device=device)
        
        model.eval()
        correct_detection = 0
        correct_classification = 0
        total = 0
        
        with torch.no_grad():
            for images, has_disease, disease_class in val_loader:
                images = images.to(device)
                has_disease = has_disease.to(device)
                disease_class = disease_class.to(device)
                
                disease_prob, disease_pred = model(images)
                
                predicted_disease = (disease_prob.squeeze() > 0.5).float()
                correct_detection += (predicted_disease == has_disease).sum().item()
                
                _, predicted_class = torch.max(disease_pred, 1)
                correct_classification += (predicted_class == disease_class).sum().item()
                total += has_disease.size(0)
        
        fold_results.append({
            'detection_accuracy': 100 * correct_detection/total,
            'classification_accuracy': 100 * correct_classification/total
        })
        
    return fold_results

def main():
    train_dir = "archive (1)/train"
    val_dir = "archive (1)/val"
    test_dir = "archive (1)/test"
    
    print("Checking directories...")
    print(f"Train directory exists: {os.path.exists(train_dir)}")
    print(f"Val directory exists: {os.path.exists(val_dir)}")
    print(f"Test directory exists: {os.path.exists(test_dir)}")
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        print("\nCreating datasets...")
        train_dataset = StrawberryDiseaseDataset(
            image_dir=train_dir,
            label_dir=train_dir,
            transform=train_transform
        )
        
        val_dataset = StrawberryDiseaseDataset(
            image_dir=val_dir,
            label_dir=val_dir,
            transform=val_transform
        )
        
        test_dataset = StrawberryDiseaseDataset(
            image_dir=test_dir,
            label_dir=test_dir,
            transform=val_transform
        )
        
        print("\nPerforming k-fold cross validation...")
        fold_results = k_fold_cross_validation(train_dataset)
        
        print("\nK-fold Cross Validation Results:")
        detection_accuracies = [r['detection_accuracy'] for r in fold_results]
        classification_accuracies = [r['classification_accuracy'] for r in fold_results]
        
        print(f"Mean Detection Accuracy: {np.mean(detection_accuracies):.2f}% ± {np.std(detection_accuracies):.2f}%")
        print(f"Mean Classification Accuracy: {np.mean(classification_accuracies):.2f}% ± {np.std(classification_accuracies):.2f}%")
        
        print("\nTraining final model...")
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        model = ImprovedDiseaseNet()
        device = 'cpu'
        print(f"Using device: {device}")
        
        train_model(model, train_loader, val_loader, num_epochs=30, device=device)
        
        print("\nEvaluating on test set...")
        model.eval()
        test_detection_correct = 0
        test_classification_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for images, has_disease, disease_class in test_loader:
                images = images.to(device)
                has_disease = has_disease.to(device)
                disease_class = disease_class.to(device)
                
                disease_prob, disease_pred = model(images)
                
                predicted_disease = (disease_prob.squeeze() > 0.5).float()
                test_detection_correct += (predicted_disease == has_disease).sum().item()
                
                _, predicted_class = torch.max(disease_pred, 1)
                test_classification_correct += (predicted_class == disease_class).sum().item()
                test_total += has_disease.size(0)
        
        print(f'Final Test Detection Accuracy: {100 * test_detection_correct/test_total:.2f}%')
        print(f'Final Test Classification Accuracy: {100 * test_classification_correct/test_total:.2f}%')
        
        torch.save(model.state_dict(), 'improved_disease_model.pth')
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    print("Entering main...")
    main() 