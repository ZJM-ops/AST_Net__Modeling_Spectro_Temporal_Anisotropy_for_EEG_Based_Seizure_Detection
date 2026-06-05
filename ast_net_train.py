from datetime import datetime
from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_curve)
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

class MultiScaleResNeXtBlock(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, **kwargs):
        super(MultiScaleResNeXtBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        width = int(planes * (base_width / 64.0)) * groups
        
        # Split phase: Dimensionality reduction
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(width)
        
        # Transform phase: Asymmetric multi-scale splitting
        self.scale_width = width // 4
        self.last_width = width - (self.scale_width * 3) 
        
        # Branch 1 (1x1): Capture localized point-wise energy intensities (Spikes)
        self.conv_branch1 = nn.Conv2d(self.scale_width, self.scale_width, 
                                      kernel_size=1, stride=stride, padding=0, bias=False)
        
        # Branch 2 (3x3): Capture standard localized contextual textures
        self.conv_branch2 = nn.Conv2d(self.scale_width, self.scale_width, 
                                      kernel_size=3, stride=stride, padding=1, 
                                      groups=self.scale_width // 2 if self.scale_width > 2 else 1, bias=False)
        
        # Branch 3 (3x11): Temporal asymmetric focus for continuous oscillatory ripples
        self.conv_branch3 = nn.Conv2d(self.scale_width, self.scale_width, 
                                      kernel_size=(3, 11), stride=stride, padding=(1, 5), 
                                      groups=self.scale_width // 2 if self.scale_width > 2 else 1, bias=False)
        
        # Branch 4 (7x1): Spectral asymmetric focus to decouple bandwidth extensions from wideband noise
        self.conv_branch4 = nn.Conv2d(self.last_width, self.last_width, 
                                      kernel_size=(7, 1), stride=stride, padding=(3, 0), 
                                      groups=self.last_width // 2 if self.last_width > 2 else 1, bias=False)

        self.bn2 = norm_layer(width) 

        # Merge phase: Dimensionality expansion
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Slice features across channels
        x1, x2, x3, x4 = torch.split(out, [self.scale_width, self.scale_width, self.scale_width, self.last_width], dim=1)
        
        out1 = self.conv_branch1(x1)
        out2 = self.conv_branch2(x2)
        out3 = self.conv_branch3(x3)
        out4 = self.conv_branch4(x4)
        
        # Concatenate multi-scale representations
        out = torch.cat([out1, out2, out3, out4], dim=1)
        
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

def apply_multiscale_block(model):
    """Refactor baseline model by systematically substituting standard bottlenecks."""
    from timm.models.resnet import Bottleneck as TimmBottleneck
    
    print("Converting model to Asymmetric Multi-Scale ResNeXt...")
    count = 0
    for name, module in model.named_modules():
        if name.startswith('layer') and len(list(module.children())) > 0:
            for i, child in enumerate(module.children()):
                if isinstance(child, TimmBottleneck):
                    inplanes = child.conv1.in_channels
                    expansion = child.expansion
                    actual_out_channels = child.conv3.out_channels
                    original_planes = actual_out_channels // expansion
                    
                    new_block = MultiScaleResNeXtBlock(
                        inplanes=inplanes,
                        planes=original_planes,
                        stride=child.conv2.stride[0], 
                        downsample=child.downsample,
                        groups=child.conv2.groups,
                        base_width=4, 
                        norm_layer=type(child.bn1)
                    )
                    
                    module[i] = new_block
                    count += 1
    print(f"Successfully replaced {count} blocks with Asymmetric Multi-Scale Blocks.")
    return model

class MultiChannelDataset(Dataset):
    def __init__(self, file_paths, transform=None, single_channel_index=None):
        self.segs = {}
        for p in file_paths:
            path_obj = Path(p)
            label_folder = path_obj.parent.name
            filename = path_obj.name

            seg_id_parts = filename.split('_seg')
            if len(seg_id_parts) > 1:
                base_name = '_seg'.join(seg_id_parts[:-1])
                seg_num = seg_id_parts[-1].split('_')[0]
                seg_id = f"{label_folder}_{base_name}_seg{seg_num}"
            else:
                seg_id = filename.split('_ch')[0]

            if seg_id not in self.segs:
                self.segs[seg_id] = []
            self.segs[seg_id].append(p)

        self.seg_ids = [k for k, v in self.segs.items() if len(v) == 8]
        for k in self.seg_ids:
            self.segs[k] = sorted(self.segs[k], key=lambda x: int(Path(x).stem.split("ch")[-1]))

        self.transform = transform
        self.label_mapping = {'Baseline': 0, 'Ictal': 1}
        self.single_channel_index = single_channel_index

    def __len__(self):
        return len(self.seg_ids)

    def __getitem__(self, idx):
        seg_id = self.seg_ids[idx]
        img_paths = self.segs[seg_id]

        if len(img_paths) != 8:
            raise ValueError(f"Sample {seg_id} holds invalid channel counts: {len(img_paths)}")

        if self.single_channel_index is not None:
            img_path = img_paths[self.single_channel_index - 1]
            imgs = Image.open(img_path).convert("RGB")
            if self.transform:
                imgs = self.transform(imgs)
        else:
            imgs = [Image.open(p).convert("RGB") for p in img_paths]
            if self.transform:
                imgs = [self.transform(img) for img in imgs]
            imgs = torch.cat(imgs, dim=0)

        label = self.label_mapping[Path(img_paths[0]).parent.name]
        return imgs, label

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def create_resnext50_32x4d(num_classes=2, pretrained=False, device="cuda"):
    model = timm.create_model('resnext50_32x4d', pretrained=pretrained, num_classes=num_classes)
    model = apply_multiscale_block(model)
    return model.to(device)

if __name__ == '__main__':
    # Initialize workspace and configurations
    SCRIPT_DIR = Path(__file__).resolve().parent
    SPLIT_DATA_DIR = SCRIPT_DIR / "data" / "split_data_42"
    MODEL_SAVE_DIR = SCRIPT_DIR / "models" / "cnn_models_0109"
    
    TRAIN_IMAGES_PATH = SPLIT_DATA_DIR / 'train_images.txt'
    VAL_IMAGES_PATH = SPLIT_DATA_DIR / 'val_images.txt'
    TEST_IMAGES_PATH = SPLIT_DATA_DIR / 'test_images.txt'

    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # Hyperparameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.01 
    WEIGHT_DECAY = 1e-4
    EPOCHS = 100
    NUM_CLASSES = 2
    IMAGE_SIZE = 224
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PATIENCE = 10

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir_name = f"{timestamp}_AsymmetricMS_ResNeXt50_b{BATCH_SIZE}_sgd{LEARNING_RATE}"
    run_save_dir = MODEL_SAVE_DIR / run_dir_name
    run_save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Execution runtime artefacts directed to: {run_save_dir}")
    print(f"Hardware Environment Target: {DEVICE}")

    # Load image pathway indexes
    with open(TRAIN_IMAGES_PATH, 'r') as f:
        train_paths = [line.strip() for line in f]
    with open(VAL_IMAGES_PATH, 'r') as f:
        val_paths = [line.strip() for line in f]
    with open(TEST_IMAGES_PATH, 'r') as f:
        test_paths = [line.strip() for line in f]

    base_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    all_test_probs = []
    test_labels = []
    channel_results = []

    # Iterate separate training loops over the 8 effective EEG channels
    for i in range(1, 9): 
        print(f"\nTraining Asymmetric Multi-Scale ResNeXt-50: Channel {i} (SGD LR={LEARNING_RATE})")

        train_dataset = MultiChannelDataset(train_paths, transform=base_transform, single_channel_index=i)
        val_dataset = MultiChannelDataset(val_paths, transform=base_transform, single_channel_index=i)
        test_dataset = MultiChannelDataset(test_paths, transform=base_transform, single_channel_index=i)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

        model = create_resnext50_32x4d(num_classes=NUM_CLASSES, pretrained=False, device=DEVICE)
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)
        scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float('inf')
        best_model_path = run_save_dir / f'best_resnext50_asym_ms_ch{i}.pth'
        early_stopping = EarlyStopping(patience=PATIENCE)

        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0
            for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} Ch{i}"):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(train_loader.dataset)

            # Validation cycle
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
            val_loss /= len(val_loader.dataset)

            scheduler.step()
            print(f"Epoch {epoch+1} ended. Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f}")

            early_stopping(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_path)

            if early_stopping.early_stop:
                print(f"Early stop triggered at epoch {epoch+1} for channel {i}")
                break

        # Inference evaluation execution
        if best_model_path.exists():
            model.load_state_dict(torch.load(best_model_path))
            
        model.eval()
        channel_probs = []
        channel_labels = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                channel_probs.extend(probs.cpu().numpy())
                channel_labels.extend(labels.cpu().numpy())

        all_test_probs.append(channel_probs)
        if not test_labels:
            test_labels = channel_labels

        preds = (np.array(channel_probs) > 0.5).astype(int)
        acc = accuracy_score(channel_labels, preds)
        f1 = f1_score(channel_labels, preds)
        precision = precision_score(channel_labels, preds)
        recall = recall_score(channel_labels, preds)
        fpr, tpr, _ = roc_curve(channel_labels, np.array(channel_probs))
        roc_auc = auc(fpr, tpr)

        channel_results.append({
            'channel': i, 'accuracy': acc, 'f1_score': f1,
            'precision': precision, 'recall': recall, 'auc': roc_auc
        })
        print(f"Ch{i} Results -> Acc: {acc:.4f}, F1: {f1:.4f}, AUC: {roc_auc:.4f}")

        # Resource decontamination
        del model, optimizer, scheduler, criterion
        torch.cuda.empty_cache()

    # Save tracking summaries
    channel_results_df = pd.DataFrame(channel_results)
    channel_results_df.to_csv(run_save_dir / "individual_channel_metrics.csv", index=False)

    print("\nExecuting Strategic Ensemble Calculations...")
    all_test_probs = np.array(all_test_probs)
    ensemble_probs = np.mean(all_test_probs, axis=0)
    ensemble_preds = (ensemble_probs > 0.5).astype(int)

    ens_acc = accuracy_score(test_labels, ensemble_preds)
    ens_f1 = f1_score(test_labels, ensemble_preds)
    ens_prec = precision_score(test_labels, ensemble_preds)
    ens_rec = recall_score(test_labels, ensemble_preds)
    ens_cm = confusion_matrix(test_labels, ensemble_preds)

    print(f"\nFinal Ensemble Results -> Accuracy: {ens_acc:.4f} | F1-Score: {ens_f1:.4f}")
    print("Confusion Matrix:\n", ens_cm)

    pd.DataFrame({"acc": [ens_acc], "f1": [ens_f1], "precision": [ens_prec], "recall": [ens_rec]}).to_csv(run_save_dir / "test_metrics_ensemble.csv", index=False)
    pd.DataFrame(ens_cm).to_csv(run_save_dir / "confusion_matrix_ensemble.csv", index=False)
    print("\nExecution complete.")