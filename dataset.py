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
    PyTorch Dataset for YOLO-format Bone Fracture Detection Data

    Binary classification: fractured vs not fractured (healthy)

    Dataset structure:
    root_dir/
    ├── train/
    │   ├── images/
    │   └── labels/  (YOLO format)
    ├── valid/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/

    Classes: 0-8 are various fracture types, 2='Healthy'
    Binary mapping: Healthy (class 2) → 0 (not fractured), All others → 1 (fractured)
    """

    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (str): Root directory containing train/valid/test folders
            split (str): One of 'train', 'val', or 'test'
            transform (callable, optional): Optional transform to be applied on images
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        # Path to the split folder (map 'val' to 'valid' for compatibility)
        split_name = 'valid' if split == 'val' else split
        self.split_dir = os.path.join(root_dir, split_name)

        # Define binary classes for output
        self.classes = ['not fractured', 'fractured']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # YOLO class names (from data.yaml)
        self.yolo_classes = ['Comminuted', 'Greenstick', 'Healthy', 'Linear',
                            'Oblique Displaced', 'Oblique', 'Segmental', 'Spiral',
                            'Transverse Displaced', 'Transverse']

        # Path to images and labels
        images_dir = os.path.join(self.split_dir, 'images')
        labels_dir = os.path.join(self.split_dir, 'labels')

        # Collect all image paths and labels
        self.samples = []

        if not os.path.exists(images_dir):
            print(f"Warning: {images_dir} does not exist")
        else:
            for img_name in os.listdir(images_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    img_path = os.path.join(images_dir, img_name)

                    # Get corresponding label file
                    label_name = os.path.splitext(img_name)[0] + '.txt'
                    label_path = os.path.join(labels_dir, label_name)

                    # Read label to determine if fractured or not
                    # If label file exists and contains classes, check if it's 'Healthy' (class 2)
                    is_fractured = 1  # Default: assume fractured

                    if os.path.exists(label_path):
                        with open(label_path, 'r') as f:
                            lines = f.readlines()
                            if lines:
                                # Get the first class in the label (YOLO format: class x y w h)
                                first_class = int(lines[0].split()[0])
                                # Class 2 is 'Healthy', map to 0 (not fractured)
                                # All other classes map to 1 (fractured)
                                is_fractured = 0 if first_class == 2 else 1

                    self.samples.append((img_path, is_fractured))

        print(f"{split_name.capitalize()} set: {len(self.samples)} images")
    
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
    not_fractured_count = (labels == 0).sum().item()
    fractured_count = (labels == 1).sum().item()
    print(f"\nBatch composition:")
    print(f"  Not fractured: {not_fractured_count}")
    print(f"  Fractured: {fractured_count}")