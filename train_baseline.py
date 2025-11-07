"""
Simple Baseline Training - Full MURA Dataset
----------------------------------------------------------------
This script trains EfficientNet-B0 on the complete MURA dataset
without domain splitting.

Features:
- EfficientNet-B0 with ImageNet pretraining
- MLflow experiment tracking
- Grad-CAM visualizations
- Class-weighted loss for imbalance
- Complete metrics and visualizations
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import copy
from datetime import datetime
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
import seaborn as sns

# MLflow for experiment tracking (optional)
try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("MLflow not available - running without experiment tracking")

# Grad-CAM for explainability (optional)
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False
    print("Pytorch-grad-cam not available - skipping Grad-CAM visualizations")

# Import MURA dataset
from MURA_dataset import MURADataset, train_transform, val_test_transform, get_class_weights

import warnings
warnings.filterwarnings("ignore")


class EfficientNetClassifier(nn.Module):
    """EfficientNet-B0 for binary classification"""

    def __init__(self, num_classes=2, pretrained=True):
        super(EfficientNetClassifier, self).__init__()

        # Load EfficientNet-B0
        if pretrained:
            try:
                # Try using timm for better EfficientNet implementation
                import timm
                self.backbone = timm.create_model('efficientnet_b0', pretrained=True)
                num_features = self.backbone.classifier.in_features
                self.backbone.classifier = nn.Linear(num_features, num_classes)
                print("Using timm EfficientNet-B0")
            except ImportError:
                # Fallback to torchvision
                from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
                self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
                num_features = self.backbone.classifier[1].in_features
                self.backbone.classifier[1] = nn.Linear(num_features, num_classes)
                print("Using torchvision EfficientNet-B0")
        else:
            from torchvision.models import efficientnet_b0
            self.backbone = efficientnet_b0(pretrained=False)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier[1] = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


def load_data(data_root, batch_size, num_workers):
    """Load MURA datasets (no domain splits)"""
    print("\n" + "="*60)
    print("Loading MURA Dataset")
    print("="*60)

    # Load full train and validation sets
    train_dataset = MURADataset(
        root_dir=data_root,
        split='train',
        transform=train_transform
    )

    val_dataset = MURADataset(
        root_dir=data_root,
        split='valid',
        transform=val_test_transform
    )

    print(f"\nClass distribution:")
    print(f"Train: {train_dataset.get_class_distribution()}")
    print(f"Val:   {val_dataset.get_class_distribution()}")

    print(f"\nBody part distribution (train):")
    for body_part, count in train_dataset.get_body_part_distribution().items():
        print(f"  {body_part}: {count}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, train_dataset, val_dataset


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, num_epochs):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')

    for inputs, labels in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        running_corrects += torch.sum(preds == labels.data)
        total_samples += batch_size

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects.double() / total_samples

    return epoch_loss, epoch_acc.item()


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc='Validating'):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            running_corrects += torch.sum(preds == labels.data)
            total_samples += batch_size

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects.double() / total_samples

    # Calculate AUC
    auc = roc_auc_score(all_labels, all_probs)

    return epoch_loss, epoch_acc.item(), auc, all_preds, all_labels, all_probs


def plot_training_curves(history, output_dir):
    """Plot training and validation curves"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(history['train_acc'], label='Train Acc', marker='o')
    axes[1].plot(history['val_acc'], label='Val Acc', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # AUC
    axes[2].plot(history['val_auc'], label='Val AUC', marker='s', color='green')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('AUC')
    axes[2].set_title('Validation AUC')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    return plot_path


def plot_roc_curve(labels, probs, output_dir):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})', linewidth=2, color='darkorange')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plot_path = os.path.join(output_dir, 'roc_curve.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    return plot_path


def plot_confusion_matrix(labels, preds, output_dir):
    """Plot confusion matrix"""
    cm = confusion_matrix(labels, preds)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['Normal', 'Abnormal'],
               yticklabels=['Normal', 'Abnormal'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')

    plot_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    return plot_path


def generate_gradcam_samples(model, val_loader, device, output_dir, num_samples=8):
    """Generate Grad-CAM visualizations for sample predictions"""
    if not GRADCAM_AVAILABLE:
        print("Skipping Grad-CAM (pytorch-grad-cam not installed)")
        return

    print("\nGenerating Grad-CAM visualizations...")

    model.eval()

    # Get target layer for Grad-CAM
    try:
        target_layers = [model.backbone.features[-1]]
    except:
        print("Warning: Could not auto-detect target layer for Grad-CAM")
        return

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available())

    # Get a batch of images
    try:
        images, labels = next(iter(val_loader))
    except StopIteration:
        return

    images = images[:num_samples].to(device)
    labels = labels[:num_samples]

    # Create visualization directory
    gradcam_dir = os.path.join(output_dir, 'gradcam')
    os.makedirs(gradcam_dir, exist_ok=True)

    # Generate Grad-CAM for each image
    for idx in range(min(num_samples, len(images))):
        input_tensor = images[idx:idx+1]

        # Get prediction
        with torch.no_grad():
            output = model(input_tensor)
            pred_class = torch.argmax(output, dim=1).item()
            prob = torch.softmax(output, dim=1)[0, pred_class].item()

        # Generate Grad-CAM
        targets = [ClassifierOutputTarget(pred_class)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]

        # Prepare original image
        img = input_tensor[0].cpu().numpy().transpose(1, 2, 0)
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)

        # Overlay Grad-CAM
        visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)

        # Create side-by-side visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].imshow(img)
        axes[0].set_title(f'Original\nTrue: {["Normal", "Abnormal"][labels[idx]]}')
        axes[0].axis('off')

        axes[1].imshow(visualization)
        axes[1].set_title(f'Grad-CAM\nPred: {["Normal", "Abnormal"][pred_class]} ({prob:.2f})')
        axes[1].axis('off')

        plt.tight_layout()
        save_path = os.path.join(gradcam_dir, f'sample_{idx}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    print(f"Grad-CAM visualizations saved to: {gradcam_dir}")


def train_model(config):
    """Complete training pipeline"""

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    os.makedirs(config['output_dir'], exist_ok=True)

    # Initialize MLflow if available
    if MLFLOW_AVAILABLE:
        mlflow.set_experiment(config['experiment_name'])
        mlflow.start_run(run_name=config['run_name'])
        mlflow.log_params(config)
        print(f"MLflow experiment: {config['experiment_name']}")

    # Load data
    train_loader, val_loader, train_dataset, val_dataset = load_data(
        config['data_root'],
        config['batch_size'],
        config['num_workers']
    )

    # Initialize model
    print("\nInitializing EfficientNet-B0 model...")
    model = EfficientNetClassifier(
        num_classes=2,
        pretrained=config['pretrained']
    ).to(device)

    # Calculate class weights
    if config['use_class_weights']:
        class_weights = get_class_weights(train_dataset).to(device)
        print(f"Class weights: {class_weights}")
    else:
        class_weights = None

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=3, verbose=True
    )

    # Training loop
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_auc = 0.0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_auc': []
    }

    start_time = time.time()

    for epoch in range(config['num_epochs']):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch, config['num_epochs']
        )

        # Validate
        val_loss, val_acc, val_auc, val_preds, val_labels, val_probs = validate(
            model, val_loader, criterion, device
        )

        # Scheduler step
        scheduler.step(val_loss)

        # Log metrics
        if MLFLOW_AVAILABLE:
            mlflow.log_metrics({
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_auc': val_auc,
                'learning_rate': optimizer.param_groups[0]['lr']
            }, step=epoch)

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)

        print(f'\nEpoch {epoch+1}/{config["num_epochs"]}:')
        print(f'  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}')
        print(f'  Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f}')

        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            best_model_wts = copy.deepcopy(model.state_dict())
            best_predictions = (val_preds, val_labels, val_probs)

            checkpoint_path = os.path.join(config['output_dir'], 'best_model.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f'  ✓ Best model saved! (AUC: {best_auc:.4f})')

    training_time = time.time() - start_time
    print(f'\nTraining complete in {training_time//60:.0f}m {training_time%60:.0f}s')
    print(f'Best Val AUC: {best_auc:.4f}')

    # Load best model
    model.load_state_dict(best_model_wts)

    # Generate visualizations
    print("\n" + "="*60)
    print("Generating Visualizations")
    print("="*60)

    val_preds, val_labels, val_probs = best_predictions

    curves_path = plot_training_curves(history, config['output_dir'])
    print(f"Training curves: {curves_path}")

    roc_path = plot_roc_curve(val_labels, val_probs, config['output_dir'])
    print(f"ROC curve: {roc_path}")

    cm_path = plot_confusion_matrix(val_labels, val_preds, config['output_dir'])
    print(f"Confusion matrix: {cm_path}")

    generate_gradcam_samples(model, val_loader, device, config['output_dir'])

    # Log artifacts to MLflow
    if MLFLOW_AVAILABLE:
        mlflow.log_artifact(curves_path)
        mlflow.log_artifact(roc_path)
        mlflow.log_artifact(cm_path)
        mlflow.pytorch.log_model(model, "model")
        mlflow.end_run()

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Best Validation AUC: {best_auc:.4f}")
    print(f"Model saved: {config['output_dir']}/best_model.pth")
    print(f"Visualizations: {config['output_dir']}/")

    if MLFLOW_AVAILABLE:
        print(f"\nView results: mlflow ui")
        print(f"Open http://localhost:5000")

    return model, history, best_auc


def main():
    """Main training function"""

    # Configuration
    config = {
        # Experiment
        'experiment_name': 'MURA_Simple_Baseline',
        'run_name': f'EfficientNet_B0_{datetime.now().strftime("%Y%m%d_%H%M%S")}',

        # Data
        'data_root': 'MURA-v1.1',

        # Model
        'pretrained': True,

        # Training
        'batch_size': 16,
        'num_epochs': 20,
        'learning_rate': 0.0001,
        'weight_decay': 1e-4,
        'use_class_weights': True,

        # System
        'num_workers': 4,
        'output_dir': 'outputs/simple_baseline',
    }

    # Print configuration
    print("\n" + "="*60)
    print("Simple Baseline Training (No Domain Splits)")
    print("="*60)
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Train
    model, history, best_auc = train_model(config)


if __name__ == "__main__":
    main()
