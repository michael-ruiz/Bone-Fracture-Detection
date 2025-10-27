import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import time
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
from datetime import datetime

from dataset import BoneFractureDataset

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import dotenv
dotenv.load_dotenv()

# class BoneFractureDataset(Dataset):
#     """Binary classification dataset for bone fractures"""
    
#     def __init__(self, root_dir, split='train', transform=None, verify_images=True):
#         self.root_dir = root_dir
#         self.split = split
#         self.transform = transform
        
#         self.split_dir = os.path.join(root_dir, split)
#         self.classes = ['fractured', 'not fractured']
#         self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
#         self.samples = []
#         corrupted_count = 0
        
#         for class_name in self.classes:
#             class_dir = os.path.join(self.split_dir, class_name)
#             if not os.path.exists(class_dir):
#                 continue
            
#             class_idx = self.class_to_idx[class_name]
#             for img_name in os.listdir(class_dir):
#                 if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
#                     img_path = os.path.join(class_dir, img_name)
                    
#                     # Verify image can be loaded if requested
#                     if verify_images:
#                         try:
#                             img = Image.open(img_path)
#                             img.load()
#                             img.close()
#                             self.samples.append((img_path, class_idx))
#                         except Exception as e:
#                             corrupted_count += 1
#                             print(f"Skipping corrupted image: {img_path}")
#                     else:
#                         self.samples.append((img_path, class_idx))
        
#         if corrupted_count > 0:
#             print(f"Found and skipped {corrupted_count} corrupted images in {split} set")
    
#     def __len__(self):
#         return len(self.samples)
    
#     def __getitem__(self, idx):
#         img_path, label = self.samples[idx]
        
#         try:
#             # Load image with error handling for corrupted files
#             image = Image.open(img_path)
#             image.load()  # Force load to catch truncated images
#             image = image.convert('RGB')
            
#             if self.transform:
#                 image = self.transform(image)
            
#             return image, label
#         except Exception as e:
#             print(f"Error loading image {img_path}: {e}")
#             # Return a blank image and the label if image is corrupted
#             blank_image = Image.new('RGB', (224, 224), color=(0, 0, 0))
#             if self.transform:
#                 blank_image = self.transform(blank_image)
#             return blank_image, label


def get_transforms():
    """Define data transforms"""
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_test_transform


def get_dataloaders(dataset_path, batch_size=32, num_workers=4):
    """Create dataloaders for train, val, and test sets"""
    train_transform, val_test_transform = get_transforms()
    
    train_dataset = BoneFractureDataset(dataset_path, 'train', train_transform)
    val_dataset = BoneFractureDataset(dataset_path, 'val', val_test_transform)
    test_dataset = BoneFractureDataset(dataset_path, 'test', val_test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                            shuffle=False, num_workers=num_workers, pin_memory=True)
    
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader, train_dataset


def get_model(num_classes=2, pretrained=True):
    """Load ResNet50 model"""
    model = models.resnet50(pretrained=pretrained)
    
    # Modify final layer for binary classification
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    running_corrects = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for inputs, labels in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_corrects.double() / len(train_loader.dataset)
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc='Validating'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
    
    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_acc = running_corrects.double() / len(val_loader.dataset)
    
    return epoch_loss, epoch_acc


def train_model(model, train_loader, val_loader, criterion, optimizer, 
                scheduler, device, num_epochs=25, save_path='best_model.pth'):
    """Complete training loop"""
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 60)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, 
                                           optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Scheduler step
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc.item())
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc.item())
        
        print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), save_path)
            print(f'✓ Best model saved with val acc: {best_acc:.4f}')
    
    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    return model, history


def test_model(model, test_loader, device, results_file='test_results.txt'):
    """Test the model and compute detailed metrics"""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Testing'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of class 1
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    accuracy = (all_preds == all_labels).mean()
    
    # Per-class metrics for Fractured (class 0)
    tp = ((all_preds == 0) & (all_labels == 0)).sum()
    tn = ((all_preds == 1) & (all_labels == 1)).sum()
    fp = ((all_preds == 0) & (all_labels == 1)).sum()
    fn = ((all_preds == 1) & (all_labels == 0)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate AUC
    auc = roc_auc_score(all_labels, all_probs)
    
    # Per-class metrics for Not Fractured (class 1)
    precision_nf = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall_nf = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_nf = 2 * (precision_nf * recall_nf) / (precision_nf + recall_nf) if (precision_nf + recall_nf) > 0 else 0
    
    # Print results
    results = f"""
{'='*60}
Test Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*60}

Overall Metrics:
  Accuracy: {accuracy:.4f}
  AUC-ROC: {auc:.4f}

Class: Fractured (0)
  Precision: {precision:.4f}
  Recall: {recall:.4f}
  F1-Score: {f1:.4f}

Class: Not Fractured (1)
  Precision: {precision_nf:.4f}
  Recall: {recall_nf:.4f}
  F1-Score: {f1_nf:.4f}

Confusion Matrix:
  True Positives (Fractured correctly identified): {tp}
  True Negatives (Not Fractured correctly identified): {tn}
  False Positives (Not Fractured misclassified as Fractured): {fp}
  False Negatives (Fractured misclassified as Not Fractured): {fn}

Total Samples: {len(all_labels)}
{'='*60}
"""
    
    print(results)
    
    # Save to file
    with open(results_file, 'w') as f:
        f.write(results)
    print(f'\nResults saved to {results_file}')
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Bone Fracture Detection')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
    print('ROC curve saved to roc_curve.png')
    plt.close()
    
    return accuracy, precision, recall, f1, auc


def plot_history(history, save_path='training_history.png'):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Training history plot saved to {save_path}')
    plt.close()


def main():
    # Configuration
    dataset_path = dotenv.get_key(dotenv.find_dotenv(), "BONE_DATA_PATH")
    batch_size = 32
    num_epochs = 25
    learning_rate = 0.001
    num_workers = 4
    
    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading datasets...")
    train_loader, val_loader, test_loader, train_dataset = get_dataloaders(
        dataset_path, batch_size, num_workers
    )
    
    # Calculate class weights for imbalanced data
    labels = [label for _, label in train_dataset.samples]
    class_counts = torch.bincount(torch.tensor(labels))
    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum()
    class_weights = class_weights.to(device)
    print(f"Class weights: {class_weights}")
    
    # Model
    print("\nInitializing ResNet50 model...")
    model = get_model(num_classes=2, pretrained=True)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                     factor=0.1, patience=3)
    
    # Train
    print("\nStarting training...")
    model, history = train_model(
        model, train_loader, val_loader, criterion, optimizer, 
        scheduler, device, num_epochs=num_epochs, save_path='best_resnet50.pth'
    )
    
    # Plot training history
    plot_history(history)
    
    # Test
    print("\nEvaluating on test set...")
    test_model(model, test_loader, device, results_file='test_results.txt')


if __name__ == "__main__":
    main()