import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from collections import defaultdict

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class MURADataset(Dataset):
    """
    PyTorch Dataset for MURA (Musculoskeletal Radiographs) Dataset

    Binary classification: normal (0) vs abnormal (1)

    Dataset structure:
    root_dir/
     train/
        XR_ELBOW/
        XR_FINGER/
        XR_FOREARM/
        XR_HAND/
        XR_HUMERUS/
        XR_SHOULDER/
        XR_WRIST/
     valid/
        (same structure)
     train_image_paths.csv
     train_labeled_studies.csv
     valid_image_paths.csv
     valid_labeled_studies.csv

    The labeled_studies CSV contains study paths and labels (0=normal, 1=abnormal)
    The image_paths CSV contains all individual image paths
    """

    def __init__(self, root_dir, split='train', transform=None, use_study_label=True,
                 custom_csv=None, domain_name=None):
        """
        Args:
            root_dir (str): Root directory of MURA dataset (e.g., 'MURA-v1.1')
            split (str): Either 'train' or 'valid'
            transform (callable, optional): Optional transform to be applied on images
            use_study_label (bool): If True, use study-level labels. Each image in a study
                                   gets the same label. If False, tries to use image-level labels.
            custom_csv (str, optional): Path to custom labeled_studies CSV (for domain splits)
            domain_name (str, optional): Name of the domain (for logging purposes)
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.use_study_label = use_study_label
        self.custom_csv = custom_csv
        self.domain_name = domain_name

        # Classes: 0 = normal, 1 = abnormal
        self.classes = ['normal', 'abnormal']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Body part categories in MURA
        self.body_parts = ['XR_ELBOW', 'XR_FINGER', 'XR_FOREARM', 'XR_HAND',
                          'XR_HUMERUS', 'XR_SHOULDER', 'XR_WRIST']

        # Load the dataset
        self.samples = []
        self._load_dataset()

        domain_label = f" ({domain_name})" if domain_name else ""
        print(f"{split.capitalize()} set{domain_label}: {len(self.samples)} images")

    def _load_dataset(self):
        """Load image paths and labels from CSV files"""
        # CSV file paths - use custom CSV if provided, otherwise use default
        if self.custom_csv:
            labeled_studies_csv = self.custom_csv
        else:
            labeled_studies_csv = os.path.join(
                self.root_dir, f'{self.split}_labeled_studies.csv'
            )

        image_paths_csv = os.path.join(
            self.root_dir, f'{self.split}_image_paths.csv'
        )

        # Check if files exist
        if not os.path.exists(labeled_studies_csv):
            raise FileNotFoundError(f"Could not find {labeled_studies_csv}")
        if not os.path.exists(image_paths_csv):
            raise FileNotFoundError(f"Could not find {image_paths_csv}")

        # Load study labels
        # Format: study_path, label (0 or 1)
        study_labels_df = pd.read_csv(
            labeled_studies_csv,
            names=['study', 'label'],
            header=None
        )

        # Create a dictionary mapping study path to label
        study_to_label = {}
        for _, row in study_labels_df.iterrows():
            study_path = row['study']
            label = int(row['label'])
            study_to_label[study_path] = label

        # Load all image paths
        # Format: image_path (one per line)
        image_paths_df = pd.read_csv(
            image_paths_csv,
            names=['image'],
            header=None
        )

        # For each image, find its study and assign the study label
        for _, row in image_paths_df.iterrows():
            image_path = row['image']

            # Extract study path from image path
            # Image path format: MURA-v1.1/train/XR_WRIST/patient00001/study1_positive/image1.png
            # Study path format: MURA-v1.1/train/XR_WRIST/patient00001/study1_positive/
            parts = image_path.split('/')
            # Get everything except the image filename
            study_path = '/'.join(parts[:-1]) + '/'

            # Look up the label for this study
            if study_path in study_to_label:
                label = study_to_label[study_path]
                # Convert path to absolute path
                full_image_path = os.path.join(self.root_dir, '..', image_path)
                # Normalize the path
                full_image_path = os.path.normpath(full_image_path)

                self.samples.append((full_image_path, label, study_path))
            else:
                print(f"Warning: No label found for study {study_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns:
            image (Tensor): Transformed image
            label (int): 0 for normal, 1 for abnormal
        """
        img_path, label, study_path = self.samples[idx]

        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image if loading fails
            image = Image.new('RGB', (224, 224), color='black')

        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_names(self):
        """Returns the list of class names"""
        return self.classes

    def get_class_distribution(self):
        """Returns count of each class"""
        labels = [label for _, label, _ in self.samples]
        unique, counts = torch.tensor(labels).unique(return_counts=True)
        distribution = {self.classes[idx.item()]: count.item()
                       for idx, count in zip(unique, counts)}
        return distribution

    def get_body_part_distribution(self):
        """Returns count of samples per body part"""
        body_part_counts = defaultdict(int)
        for img_path, _, _ in self.samples:
            for body_part in self.body_parts:
                if body_part in img_path:
                    body_part_counts[body_part] += 1
                    break
        return dict(body_part_counts)

    def get_study_path(self, idx):
        """Get the study path for a given sample index"""
        return self.samples[idx][2]


# Define transforms for MURA dataset (following ImageNet standards)
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
    labels = [label for _, label, _ in dataset.samples]
    class_counts = torch.bincount(torch.tensor(labels))
    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum()  # Normalize
    return class_weights


# Example usage
if __name__ == "__main__":
    # Path to MURA dataset
    # Download from: https://stanfordmlgroup.github.io/competitions/mura/
    # or from Kaggle
    dataset_path = "MURA-v1.1"

    # Create datasets
    print("Loading MURA datasets...")
    train_dataset = MURADataset(
        root_dir=dataset_path,
        split='train',
        transform=train_transform
    )

    val_dataset = MURADataset(
        root_dir=dataset_path,
        split='valid',
        transform=val_test_transform
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,  # MURA images can be large, use smaller batch
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Print dataset info
    print(f"\n{'='*60}")
    print(f"MURA Dataset Statistics")
    print(f"{'='*60}")
    print(f"Classes: {train_dataset.get_class_names()}")
    print(f"\nTrain distribution: {train_dataset.get_class_distribution()}")
    print(f"Val distribution: {val_dataset.get_class_distribution()}")

    print(f"\nTrain body part distribution:")
    for body_part, count in train_dataset.get_body_part_distribution().items():
        print(f"  {body_part}: {count}")

    print(f"\nVal body part distribution:")
    for body_part, count in val_dataset.get_body_part_distribution().items():
        print(f"  {body_part}: {count}")

    # Calculate class weights for handling imbalance
    class_weights = get_class_weights(train_dataset)
    print(f"\nClass weights for training: {class_weights}")

    # Test loading a batch
    print(f"\n{'='*60}")
    print(f"Testing DataLoader")
    print(f"{'='*60}")
    try:
        images, labels = next(iter(train_loader))
        print(f"Batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Labels in batch: {labels}")
        print(f"Unique labels: {labels.unique()}")

        # Show sample distribution in batch
        normal_count = (labels == 0).sum().item()
        abnormal_count = (labels == 1).sum().item()
        print(f"\nBatch composition:")
        print(f"  Normal: {normal_count}")
        print(f"  Abnormal: {abnormal_count}")
    except Exception as e:
        print(f"Error loading batch: {e}")
        print("Make sure the MURA dataset path is correct and contains the CSV files.")
