import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import argparse

from dataset import RestorationDataset
from unet import ResNetUNet
from losses import CombinedLoss

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(dataloader, desc="Training")
    for damaged, clean in pbar:
        damaged = damaged.to(device)
        clean = clean.to(device)
        
        optimizer.zero_grad()
        outputs = model(damaged)
        
        loss = criterion(outputs, clean)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * damaged.size(0)
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
    return running_loss / len(dataloader.dataset)

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for damaged, clean in tqdm(dataloader, desc="Validation"):
            damaged = damaged.to(device)
            clean = clean.to(device)
            
            outputs = model(damaged)
            loss = criterion(outputs, clean)
            running_loss += loss.item() * damaged.size(0)
            
    return running_loss / len(dataloader.dataset)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='d:/MV/restoration_dataset')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--loss_mode', type=str, default='baseline', choices=['baseline', 'perceptual1', 'perceptual2', 'ssim'])
    parser.add_argument('--test_run', action='store_true', help='Run for a few batches only')
    parser.add_argument('--resume', type=str, default='', help='Path to checkpoint to resume from')
    parser.add_argument('--start_epoch', type=int, default=0, help='Epoch to start from')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load datasets
    train_metadata = f"{args.dataset_dir}/metadata/train.json"
    val_metadata = f"{args.dataset_dir}/metadata/val.json"
    
    train_dataset = RestorationDataset(train_metadata, args.dataset_dir, is_train=True)
    val_dataset = RestorationDataset(val_metadata, args.dataset_dir, is_train=False)
    
    # Optional subset for quick test_run
    if args.test_run:
        train_dataset = torch.utils.data.Subset(train_dataset, range(16))
        val_dataset = torch.utils.data.Subset(val_dataset, range(8))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    model = ResNetUNet().to(device)
    
    if args.resume:
        print(f"Resuming from {args.resume}")
        model.load_state_dict(torch.load(args.resume, map_location=device))
    
    # Freeze encoder initially (transfer learning best practice)
    # The inc and layer1..4 are from pretrained resnet
    # We will only train decoder for this quick test / first few epochs
    for name, param in model.named_parameters():
        if 'up' in name or 'outc' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
            
    # Select Loss
    print(f"Using Loss Mode: {args.loss_mode}")
    criterion = CombinedLoss(mode=args.loss_mode).to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    
    # Training Loop
    best_loss = float('inf')
    checkpoints_dir = Path("d:/MV/checkpoints")
    checkpoints_dir.mkdir(exist_ok=True)
    
    print(f"Starting training for {args.epochs} epochs (Test Run: {args.test_run})")
    for epoch in range(args.start_epoch, args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        print(f"Epoch [{epoch+1}/{args.epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), checkpoints_dir / f"best_{args.loss_mode}.pth")
            print("Saved best model.")
            
        # Add checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), checkpoints_dir / f"checkpoint_ep{epoch+1}_{args.loss_mode}.pth")
            print(f"Saved checkpoint for epoch {epoch+1}.")

if __name__ == "__main__":
    main()
