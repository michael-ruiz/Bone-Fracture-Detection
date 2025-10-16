import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
import cv2
import dotenv
dotenv.load_dotenv()

class BoneFractureDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, img_size=640):
        """
        Args:
            root_dir (string): Directory with all the images and labels
            split (string): 'train', 'val', or 'test'
            transform (callable, optional): Optional transform to be applied on a sample
            img_size (int): Size of the image (default 640)
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.img_size = img_size
        
        # Get image directory for the current split
        if split == 'train':
            self.img_dir = os.path.join(root_dir, 'train/images')
            self.label_dir = os.path.join(root_dir, 'train/labels')
        elif split == 'val':
            self.img_dir = os.path.join(root_dir, 'valid/images')
            self.label_dir = os.path.join(root_dir, 'valid/labels')
        elif split == 'test':
            self.img_dir = os.path.join(root_dir, 'test/images')
            self.label_dir = os.path.join(root_dir, 'test/labels')
        
        # Get list of image files
        self.image_files = [f for f in os.listdir(self.img_dir) 
                           if f.endswith(('.jpg', '.jpeg', '.png'))]
        self.image_files.sort()
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Get image path
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Load labels if they exist
        label_path = os.path.join(self.label_dir, img_name.replace('.jpg', '.txt').replace('.png', '.txt'))
        
        # Initialize labels
        labels = []
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    data = line.strip().split()
                    if len(data) >= 5:
                        class_id = int(float(data[0]))
                        x_center = float(data[1])
                        y_center = float(data[2])
                        width = float(data[3])
                        height = float(data[4])
                        
                        # Convert YOLO format to (x_min, y_min, x_max, y_max)
                        x_min = (x_center - width/2) * self.img_size
                        y_min = (y_center - height/2) * self.img_size
                        x_max = (x_center + width/2) * self.img_size
                        y_max = (y_center + height/2) * self.img_size
                        
                        labels.append([class_id, x_min, y_min, x_max, y_max])
        
        # Convert to tensor
        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        
        if len(labels) == 0:
            targets = torch.zeros((0, 5))
        else:
            targets = torch.tensor(labels, dtype=torch.float32)
            
        return image_tensor, targets

def detection_collate_fn(batch):
    images = []
    targets = []
    for img, tgt in batch:
        images.append(img)
        targets.append(tgt)
    return torch.stack(images, 0), targets

def load_bone_fracture_data(root_dir, batch_size=16, shuffle=True):
    """
    Load bone fracture dataset for training
    
    Args:
        root_dir (string): Path to the root directory containing data.yaml
        batch_size (int): Batch size for DataLoader
        shuffle (bool): Whether to shuffle the data
        
    Returns:
        train_loader, val_loader, test_loader: PyTorch DataLoaders
    """

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # Create datasets
    train_dataset = BoneFractureDataset(root_dir, split='train', transform=transform)
    val_dataset = BoneFractureDataset(root_dir, split='val', transform=transform)
    test_dataset = BoneFractureDataset(root_dir, split='test', transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, collate_fn=detection_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=detection_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=detection_collate_fn)
    
    return train_loader, val_loader, test_loader

# Example usage:
if __name__ == "__main__":
    root_dir = os.getenv('BONE_DATA_PATH')
    
    try:
        train_loader, val_loader, test_loader = load_bone_fracture_data(root_dir, batch_size=8)
        
        print("Dataset loaded successfully!")
        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print(f"Test samples: {len(test_loader.dataset)}")
        
        for i, (images, targets) in enumerate(train_loader):
            print(f"Batch {i}: Images shape {images.shape}, Number of targets: {len(targets)}")
            for j, t in enumerate(targets):
                print(f"  Image {j}: target shape {t.shape}")
            if i >= 2:  # Just show first few batches
                break
                
    except Exception as e:
        print(f"Error loading dataset: {e}")
