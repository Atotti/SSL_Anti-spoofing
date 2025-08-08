#!/usr/bin/env python3
"""CM Hub training wrapper for SSL_Anti-spoofing model using Hydra."""

import os
import sys
import json
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model import Model
from data_utils_SSL import Dataset_ASVspoof2019_train
from core_scripts.startup_config import set_random_seed


def load_wav_scp(scp_path):
    """Load wav.scp file and return list of (utt_id, wav_path)."""
    entries = []
    with open(scp_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(None, 1)
                if len(parts) == 2:
                    entries.append((parts[0], parts[1]))
    return entries


def load_utt2label(json_path):
    """Load utt2label JSON."""
    with open(json_path, 'r') as f:
        return json.load(f)


def evaluate_accuracy(dev_loader, model, device):
    """Evaluate model on dev set."""
    val_loss = 0.0
    num_total = 0.0
    model.eval()
    
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    
    with torch.no_grad():
        for batch_x, batch_y in dev_loader:
            batch_size = batch_x.size(0)
            num_total += batch_size
            batch_x = batch_x.to(device)
            batch_y = batch_y.view(-1).type(torch.int64).to(device)
            batch_out = model(batch_x)
            
            batch_loss = criterion(batch_out, batch_y)
            val_loss += (batch_loss.item() * batch_size)
    
    val_loss /= num_total
    return val_loss


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    """Main training function."""
    # Print config
    print("Training configuration:")
    print(OmegaConf.to_yaml(cfg))
    
    # Set random seed
    set_random_seed(cfg.train.seed, cfg)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    train_entries = load_wav_scp(cfg.data.train_scp)
    dev_entries = load_wav_scp(cfg.data.dev_scp)
    utt2label = load_utt2label(cfg.data.labels)
    
    # Filter entries to only those with labels
    train_list = [utt_id for utt_id, _ in train_entries if utt_id in utt2label]
    dev_list = [utt_id for utt_id, _ in dev_entries if utt_id in utt2label]
    
    train_labels = {utt_id: utt2label[utt_id] for utt_id in train_list}
    dev_labels = {utt_id: utt2label[utt_id] for utt_id in dev_list}
    
    print(f"Train samples: {len(train_list)}")
    print(f"Dev samples: {len(dev_list)}")
    
    # Create dummy args object for model
    class Args:
        def __init__(self):
            self.track = cfg.track
            # Add other needed args from original main_SSL_LA.py
            self.algo = 5  # Default RawBoost algo
    
    args = Args()
    
    # Create datasets
    train_dataset = Dataset_ASVspoof2019_train(
        args,
        list_IDs=train_list,
        labels=train_labels,
        base_dir="",  # Not used
        track=cfg.track,
        algo=args.algo
    )
    
    dev_dataset = Dataset_ASVspoof2019_train(
        args,
        list_IDs=dev_list,
        labels=dev_labels,
        base_dir="",
        track=cfg.track,
        algo=args.algo
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4
    )
    
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4
    )
    
    # Create model
    print("Creating model...")
    model = Model(args, device)
    model = model.to(device)
    
    # Loss and optimizer
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    
    # Create output directories
    out_dir = Path(cfg.train.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb if enabled
    if cfg.wandb.enable:
        try:
            import wandb
            wandb.init(
                project=os.environ.get('WANDB_PROJECT', 'cmhub'),
                name=cfg.wandb.run_name,
                config=OmegaConf.to_container(cfg, resolve=True)
            )
        except ImportError:
            print("WARNING: wandb not installed, logging disabled")
            cfg.wandb.enable = False
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(cfg.train.max_epochs):
        # Training
        model.train()
        train_loss = 0.0
        num_total = 0
        
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            batch_size = batch_x.size(0)
            num_total += batch_size
            
            batch_x = batch_x.to(device)
            batch_y = batch_y.view(-1).type(torch.int64).to(device)
            
            # Forward pass
            batch_out = model(batch_x)
            batch_loss = criterion(batch_out, batch_y)
            
            # Backward pass
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            
            train_loss += batch_loss.item() * batch_size
            
            # Progress
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{cfg.train.max_epochs}, "
                      f"Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {batch_loss.item():.4f}")
        
        # Calculate epoch metrics
        train_loss /= num_total
        val_loss = evaluate_accuracy(dev_loader, model, device)
        
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        
        # Log to wandb
        if cfg.wandb.enable:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss
            })
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = ckpt_dir / "best.pth"
            torch.save(model.state_dict(), best_path)
            print(f"Saved best model to {best_path}")
        
        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            ckpt_path = ckpt_dir / f"epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")
    
    print(f"Training completed. Best val_loss: {best_val_loss:.4f}")
    
    # Finish wandb
    if cfg.wandb.enable:
        wandb.finish()


if __name__ == "__main__":
    main()