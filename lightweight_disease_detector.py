print("Script is starting...")

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import os
import json
import matplotlib.pyplot as plt

class LightweightDiseaseNet(nn.Module):
    def __init__(self, num_classes=7):
        super(LightweightDiseaseNet, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32),
            nn.Conv2d(32, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64),
            nn.Conv2d(64, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        self.detection_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
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

def train_model(model, train_loader, val_loader, num_epochs=10, device='cuda'):
    criterion_detection = nn.BCELoss()
    criterion_classification = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model = model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for images, has_disease, disease_class in train_loader:
            images = images.to(device)
            has_disease = has_disease.to(device)
            disease_class = disease_class.to(device)
            
            optimizer.zero_grad()
            
            disease_prob, disease_pred = model(images)
            
            detection_loss = criterion_detection(disease_prob.squeeze(), has_disease)
            classification_loss = criterion_classification(disease_pred, disease_class)
            
            total_loss = detection_loss + 0.5 * classification_loss
            
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item()
            
        model.eval()
        val_loss = 0.0
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
        
        print(f'Epoch {epoch+1}:')
        print(f'Training Loss: {running_loss/len(train_loader):.4f}')
        print(f'Detection Accuracy: {100 * correct_detection/total:.2f}%')
        print(f'Classification Accuracy: {100 * correct_classification/total:.2f}%')

def visualize_results(model, test_loader, device, num_samples=5):
    model.eval()
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, num_samples*5))
    
    disease_names = {
        0: 'Angular Leafspot',
        1: 'Anthracnose Fruit Rot',
        2: 'Blossom Blight',
        3: 'Gray Mold',
        4: 'Leaf Spot',
        5: 'Powdery Mildew Fruit',
        6: 'Powdery Mildew Leaf'
    }
    
    with torch.no_grad():
        for i, (images, has_disease, disease_class) in enumerate(test_loader):
            if i >= num_samples:
                break
                
            images = images.to(device)
            has_disease = has_disease.to(device)
            disease_class = disease_class.to(device)
            
            disease_prob, disease_pred = model(images)
            
            predicted_disease = (disease_prob.squeeze() > 0.5).float()
            _, predicted_class = torch.max(disease_pred, 1)
            
            img = images[0].cpu().permute(1, 2, 0)
            img = img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
            img = img.numpy()
            img = np.clip(img, 0, 1)
            
            axes[i, 0].imshow(img)
            axes[i, 0].set_title('Original Image')
            axes[i, 0].axis('off')
            
            axes[i, 1].text(0.5, 0.5, 
                          f'True: {"Diseased" if has_disease[0] else "Healthy"}\n' +
                          f'Predicted: {"Diseased" if predicted_disease[0] else "Healthy"}\n' +
                          f'True Disease: {disease_names[disease_class[0].item()]}\n' +
                          f'Predicted Disease: {disease_names[predicted_class[0].item()]}',
                          ha='center', va='center')
            axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('test_predictions.png')
    plt.close()
    
def main():
    train_dir = "archive (1)/train"
    val_dir = "archive (1)/val"
    test_dir = "archive (1)/test"
    
    print("Checking directories...")
    print(f"Train directory exists: {os.path.exists(train_dir)}")
    print(f"Val directory exists: {os.path.exists(val_dir)}")
    print(f"Test directory exists: {os.path.exists(test_dir)}")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    try:
        print("\nCreating datasets...")
        train_dataset = StrawberryDiseaseDataset(
            image_dir=train_dir,
            label_dir=train_dir,
            transform=transform
        )
        print(f"Train dataset size: {len(train_dataset)}")
        
        val_dataset = StrawberryDiseaseDataset(
            image_dir=val_dir,
            label_dir=val_dir,
            transform=transform
        )
        print(f"Val dataset size: {len(val_dataset)}")
        
        test_dataset = StrawberryDiseaseDataset(
            image_dir=test_dir,
            label_dir=test_dir,
            transform=transform
        )
        print(f"Test dataset size: {len(test_dataset)}")
        
        print("\nCreating dataloaders...")
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        print("\nInitializing model...")
        model = LightweightDiseaseNet()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        print("\nStarting training...")
        train_model(model, train_loader, val_loader, num_epochs=10, device=device)
        
        print("\nEvaluating on test set...")
        model.eval()
        correct_detection = 0
        correct_classification = 0
        total = 0
        
        with torch.no_grad():
            for images, has_disease, disease_class in test_loader:
                images = images.to(device)
                has_disease = has_disease.to(device)
                disease_class = disease_class.to(device)
                
                disease_prob, disease_pred = model(images)
                
                predicted_disease = (disease_prob.squeeze() > 0.5).float()
                correct_detection += (predicted_disease == has_disease).sum().item()
                
                _, predicted_class = torch.max(disease_pred, 1)
                correct_classification += (predicted_class == disease_class).sum().item()
                total += has_disease.size(0)
        
        print(f'Test Detection Accuracy: {100 * correct_detection/total:.2f}%')
        print(f'Test Classification Accuracy: {100 * correct_classification/total:.2f}%')
        
        torch.save(model.state_dict(), 'lightweight_disease_model.pth')

        print("\nGenerating visualization...")
        visualize_results(model, test_loader, device)
        print("Visualization saved as 'test_predictions.png'")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    print("Entering main...")
    main()