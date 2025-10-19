# train_custom_detector.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import json
from datetime import datetime

class CustomObjectDataset(Dataset):
    def __init__(self, data_dir, target_name="my_object", transform=None):
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Struktura folderów:
        # training_data/
        #   ├── [target_name]/     # Zdjęcia celu (label 1)
        #   ├── similar_objects/   # Podobne obiekty (label 0)  
        #   ├── other_objects/     # Inne obiekty (label 0)
        #   └── background/        # Tło bez celu (label 0)
        
        classes = {
            target_name: 1,           # Cel - pozytywny
            'similar_objects': 0,     # Podobne obiekty - negatywne
            'other_objects': 0,       # Inne obiekty - negatywne
            'background': 0           # Tło - negatywne
        }
        
        for class_name, label in classes.items():
            class_dir = os.path.join(data_dir, class_name)
            if os.path.exists(class_dir):
                for img_file in os.listdir(class_dir):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.images.append(os.path.join(class_dir, img_file))
                        self.labels.append(label)
        
        print(f"Loaded {len(self.images)} images:")
        print(f"- Target '{target_name}': {sum(self.labels)} photos")
        print(f"- Negative: {len(self.labels) - sum(self.labels)} photos")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class CustomObjectDetector:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.transform = None
        self.target_name = "custom_object"
        
    def set_target_name(self, name):
        """Ustawia nazwę rozpoznawanego obiektu"""
        self.target_name = name
        
    def create_model(self):
        """Tworzy model do rozpoznawania konkretnego obiektu"""
        self.model = models.resnet50(pretrained=True)
        
        # Zamroź wczesne warstwy
        for param in list(self.model.parameters())[:-50]:
            param.requires_grad = False
        
        # Dostosuj ostatnią warstwę
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        
        self.model = self.model.to(self.device)
        
        # Transformacje danych
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def train(self, data_dir, target_name="my_object", epochs=30):
        """Trenuje model do rozpoznawania konkretnego obiektu"""
        self.set_target_name(target_name)
        self.create_model()
        
        # Przygotuj dane
        dataset = CustomObjectDataset(data_dir, target_name, self.transform)
        
        if len(dataset) < 50:
            print("⚠️  UWAGA: Masz mało danych treningowych!")
            print("⭐ Dodaj przynajmniej 50-100 zdjęć celu dla dobrych rezultatów")
        
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        val_dataset.dataset.transform = self.val_transform
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        
        criterion = nn.BCELoss()
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=0.0001, weight_decay=1e-4
        )
        
        print(f"🚀 Starting training of {target_name.upper()} model...")
        print(f"📊 Training images: {len(train_dataset)}")
        print(f"📊 Validation images: {len(val_dataset)}")
        
        best_accuracy = 0
        for epoch in range(epochs):
            # Trening
            self.model.train()
            train_loss = 0
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.float().to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images).squeeze()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Walidacja
            self.model.eval()
            val_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.float().to(self.device)
                    outputs = self.model(images).squeeze()
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    predicted = (outputs > 0.5).float()
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            accuracy = 100 * correct / total
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss/len(train_loader):.4f}, '
                  f'Val Loss: {val_loss/len(val_loader):.4f}, Val Accuracy: {accuracy:.2f}%')
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                model_filename = f'{target_name}_model.pth'
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'accuracy': accuracy,
                    'epoch': epoch,
                    'target_name': target_name
                }, model_filename)
                print(f"💾 New best model saved as {model_filename}! Accuracy: {accuracy:.2f}%")
        
        print(f"✅ Training completed! Best accuracy: {best_accuracy:.2f}%")
        
        # Zapisz informacje o modelu
        model_info = {
            'target_name': target_name,
            'training_date': str(datetime.now()),
            'best_accuracy': best_accuracy,
            'total_images': len(dataset),
            'positive_samples': sum(dataset.labels),
            'negative_samples': len(dataset.labels) - sum(dataset.labels)
        }
        
        with open(f'{target_name}_model_info.json', 'w') as f:
            json.dump(model_info, f, indent=2)

def main():
    detector = CustomObjectDetector()
    
    data_dir = 'training_data'
    
    if not os.path.exists(data_dir):
        print("❌ Folder 'training_data' nie istnieje!")
        print("\n📁 Stwórz strukturę folderów uruchamiając:")
        print("python setup_training_folders.py [nazwa_celu]")
        return
    
    # Sprawdź dostępne cele treningowe
    available_targets = [d for d in os.listdir(data_dir) 
                        if os.path.isdir(os.path.join(data_dir, d)) 
                        and d not in ['similar_objects', 'other_objects', 'background']]
    
    if not available_targets:
        print("❌ Nie znaleziono folderu z celem treningowym!")
        print("Uruchom: python setup_training_folders.py [nazwa_celu]")
        return
    
    if len(available_targets) == 1:
        target_name = available_targets[0]
    else:
        print("📁 Dostępne cele treningowe:")
        for i, target in enumerate(available_targets, 1):
            print(f"{i}. {target}")
        choice = input("Wybierz numer celu: ")
        target_name = available_targets[int(choice)-1]
    
    target_dir = os.path.join(data_dir, target_name)
    if not os.path.exists(target_dir) or len(os.listdir(target_dir)) == 0:
        print(f"❌ Dodaj zdjęcia do folderu 'training_data/{target_name}/'")
        return
    
    print(f"🎯 Rozpoczynam trenowanie modelu dla: {target_name}...")
    detector.train(data_dir, target_name, epochs=25)

if __name__ == "__main__":
    main()