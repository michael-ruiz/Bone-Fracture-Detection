import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import dotenv
dotenv.load_dotenv()


class BoneFractureDataset(Dataset):
    """
    PyTorch Dataset for Kaggle Fracture Multi-Region X-ray Data
    https://www.kaggle.com/datasets/bmadushanirodrigo/fracture-multi-region-x-ray-data
    
    Binary classification: fractured vs not fractured
    
    Dataset structure:
    root_dir/
    ├── train/
    │   ├── fractured/
    │   └── not fractured/
    ├── val/
    │   ├── fractured/
    │   └── not fractured/
    └── test/
        ├── fractured/
        └── not fractured/
    """
    
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (str): Root directory containing train/val/test folders
            split (str): One of 'train', 'val', or 'test'
            transform (callable, optional): Optional transform to be applied on images
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # Path to the split folder
        self.split_dir = os.path.join(root_dir, split)
        
        # Define classes
        self.classes = ['fractured', 'not fractured']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Collect all image paths and labels
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.split_dir, class_name)
            
            if not os.path.exists(class_dir):
                print(f"Warning: {class_dir} does not exist")
                continue
            
            class_idx = self.class_to_idx[class_name]
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, class_idx))
        
        print(f"{split.capitalize()} set: {len(self.samples)} images")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_names(self):
        """Returns the list of class names"""
        return self.classes
    
    def get_class_distribution(self):
        """Returns count of each class"""
        labels = [label for _, label in self.samples]
        unique, counts = torch.tensor(labels).unique(return_counts=True)
        distribution = {self.classes[idx.item()]: count.item() 
                       for idx, count in zip(unique, counts)}
        return distribution


# Define transforms for training with augmentation
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

# Define transforms for validation/test without augmentation
val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])


def get_class_weights(dataset):
    """
    Calculate class weights for handling imbalanced datasets
    Useful for weighted loss functions
    """
    labels = [label for _, label in dataset.samples]
    class_counts = torch.bincount(torch.tensor(labels))
    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum()  # Normalize
    return class_weights


# Example usage
if __name__ == "__main__":
    dataset_path = "data"
    
    # Create datasets
    train_dataset = BoneFractureDataset(
        root_dir=dataset_path,
        split='train',
        transform=train_transform
    )
    
    val_dataset = BoneFractureDataset(
        root_dir=dataset_path,
        split='val',
        transform=val_test_transform
    )
    
    test_dataset = BoneFractureDataset(
        root_dir=dataset_path,
        split='test',
        transform=val_test_transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Print dataset info
    print(f"\n{'='*50}")
    print(f"Dataset Statistics")
    print(f"{'='*50}")
    print(f"Classes: {train_dataset.get_class_names()}")
    print(f"\nTrain distribution: {train_dataset.get_class_distribution()}")
    print(f"Val distribution: {val_dataset.get_class_distribution()}")
    print(f"Test distribution: {test_dataset.get_class_distribution()}")
    
    # Calculate class weights for handling imbalance
    class_weights = get_class_weights(train_dataset)
    print(f"\nClass weights for training: {class_weights}")
    
    # Test loading a batch
    print(f"\n{'='*50}")
    print(f"Testing DataLoader")
    print(f"{'='*50}")
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Labels in batch: {labels}")
    print(f"Unique labels: {labels.unique()}")
    
    # Show sample distribution in batch
    fractured_count = (labels == 0).sum().item()
    not_fractured_count = (labels == 1).sum().item()
    print(f"\nBatch composition:")
    print(f"  Fractured: {fractured_count}")
    print(f"  Not fractured: {not_fractured_count}")