#!/usr/bin/env python3
"""CM Hub scoring wrapper for SSL_Anti-spoofing model."""

import argparse
import json
import sys
import os
import torch
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model import Model
from data_utils_SSL import Dataset_ASVspoof2021_eval
from torch.utils.data import DataLoader


def load_wav_scp(scp_path):
    """Load wav.scp file."""
    entries = []
    with open(scp_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(None, 1)
                if len(parts) == 2:
                    entries.append((parts[0], parts[1]))
    return entries


def score_files(model, wav_entries, device):
    """Score audio files and yield results."""
    model.eval()
    
    # Create dataset for all files
    utt_ids = [utt_id for utt_id, _ in wav_entries]
    dataset = Dataset_ASVspoof2021_eval(
        list_IDs=utt_ids,
        base_dir="",  # Not used
    )
    
    # Create loader
    loader = DataLoader(dataset, batch_size=10, shuffle=False, drop_last=False)
    
    with torch.no_grad():
        idx = 0
        for batch_x, batch_utt_ids in loader:
            batch_x = batch_x.to(device)
            batch_out = model(batch_x)
            
            # Process each sample in the batch
            for i in range(batch_out.size(0)):
                # Get the score for bonafide class (index 1)
                # The model outputs logits for [spoof, bonafide]
                logits = batch_out[i]
                
                # Convert to probabilities
                probs = torch.softmax(logits, dim=0)
                
                # Score: log odds of bonafide vs spoof
                # Positive score = bonafide, negative = spoof
                score = float(torch.log(probs[1] / probs[0]))
                
                yield {
                    "utt_id": batch_utt_ids[i],
                    "score": score
                }
                idx += 1


def main():
    parser = argparse.ArgumentParser(description='CM Hub scoring wrapper')
    parser.add_argument('--track', required=True, choices=['LA', 'PA', 'DF'])
    parser.add_argument('--ckpt', required=True, help='Checkpoint path')
    parser.add_argument('--wav-scp', required=True, help='wav.scp file path')
    parser.add_argument('--stdout-jsonl', action='store_true', help='Output JSONL to stdout')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = Model(args, device)
    
    # Load checkpoint
    checkpoint = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    
    # Load wav.scp
    wav_entries = load_wav_scp(args.wav_scp)
    
    # Score files and output
    for result in score_files(model, wav_entries, device):
        if args.stdout_jsonl:
            print(json.dumps(result))
            sys.stdout.flush()
        else:
            # For debugging
            print(f"{result['utt_id']}: {result['score']:.4f}", file=sys.stderr)


if __name__ == '__main__':
    main()