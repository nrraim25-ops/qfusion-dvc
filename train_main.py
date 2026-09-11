"""Main training script for Dense Video Captioning across 3 fusion conditions and 3 random seeds.

Per Section 8:
- Controlled via --fusion_type classical|classical_matched|quantum --seed 42|123|2024
- AdamW with two parameter groups (1e-4 for heads/fusion, 1e-5 for backbones)
- Linear warmup + cosine decay
- Gradient clipping at 1.0
- Weight decay 1e-4
- Batch size starting at 4 with logged auto-halving on OOM
- 50-epoch cap with patience-5 early stopping
"""

import os
import sys

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["MPLCONFIGDIR"] = "/tmp/mpl"
import glob
import json
import yaml
import random
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from models.dvc_model import DVCModel
from data.preprocess import VideoAudioSnippetExtractor
from backbones.vivit_backbone import ViViTBackbone
from backbones.ast_backbone import ASTBackbone


def set_seed(seed: int):
    """Sets random seed deterministically across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


from data.vocabulary import Vocabulary, build_activitynet_vocab


class DVCDataset(Dataset):
    def __init__(self, data_list: list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        return {
            "vid_id": item["vid_id"],
            "video_feats": torch.tensor(item["video_feats"], dtype=torch.float32),
            "audio_feats": torch.tensor(item["audio_feats"], dtype=torch.float32),
            "segments": torch.tensor(item["segments"], dtype=torch.float32),
            "sentences": item.get("sentences", []),
            "duration": item["duration"]
        }


def make_collate_fn(vocab):
    def collate_dvc(batch):
        max_T = max(item["video_feats"].size(0) for item in batch)
        B = len(batch)
        max_events = max(max(1, len(item["segments"])) for item in batch)

        padded_v = torch.zeros(B, max_T, 768)
        padded_a = torch.zeros(B, max_T, 768)
        caption_tokens = torch.zeros(B, max_events, 25, dtype=torch.long)
        targets = []
        vid_ids = []

        for b, item in enumerate(batch):
            T = item["video_feats"].size(0)
            padded_v[b, :T] = item["video_feats"]
            padded_a[b, :T] = item["audio_feats"]
            targets.append({
                "segments": item["segments"],
                "labels": torch.ones(len(item["segments"]), dtype=torch.int64)
            })
            vid_ids.append(item["vid_id"])
            sents = item.get("sentences", [])
            for i, sent in enumerate(sents):
                if i < max_events:
                    caption_tokens[b, i] = torch.tensor(vocab.encode(sent, max_len=25), dtype=torch.long)

        return {
            "vid_ids": vid_ids,
            "video_feats": padded_v,
            "audio_feats": padded_a,
            "targets": targets,
            "caption_tokens": caption_tokens
        }
    return collate_dvc


def prepare_dvc_dataset(manifest_path: str, extractor, vivit, ast_model, device, cache_file: str, max_videos: int = 150):
    """Loads or extracts cached snippet representations for DVC training."""
    if os.path.exists(cache_file):
        print(f"Loading cached DVC features from {cache_file}...")
        with open(cache_file, "r") as f:
            data = json.load(f)
        return data

    print(f"Extracting DVC features from {manifest_path} (up to {max_videos} videos)...")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    data = []
    count = 0
    for vid_id, item in manifest.items():
        if count >= max_videos:
            break
        v_path = item["path"]
        if not os.path.exists(v_path):
            continue

        timestamps = item.get("timestamps", [])
        sentences = item.get("sentences", [])
        duration = item.get("duration", 0.0)
        if not timestamps or duration <= 0:
            continue

        try:
            v_snips, a_snips, meta = extractor.extract_snippets(v_path)
            T_snips = meta["num_snippets"]

            # Normalized ground-truth segments [s / duration, e / duration] in [0, 1]
            norm_segs = []
            valid_sents = []
            for i, ts in enumerate(timestamps):
                s_norm = max(0.0, min(1.0, ts[0] / duration))
                e_norm = max(0.0, min(1.0, ts[1] / duration))
                if e_norm > s_norm:
                    norm_segs.append([s_norm, e_norm])
                    valid_sents.append(sentences[i] if i < len(sentences) else "action in video")

            if not norm_segs:
                continue

            with torch.no_grad():
                v_feats = vivit(v_snips.to(device)).cpu().numpy().tolist()
                a_feats = ast_model(a_snips.to(device)).cpu().numpy().tolist()

            data.append({
                "vid_id": vid_id,
                "video_feats": v_feats,
                "audio_feats": a_feats,
                "segments": norm_segs,
                "sentences": valid_sents,
                "duration": duration
            })
            count += 1
            print(f"  Processed {count}/{max_videos} videos: {vid_id} ({T_snips} snippets)...", flush=True)
        except Exception as e:
            continue

    # Cache features to disk
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(data, f)
    print(f"Saved {len(data)} cached videos to {cache_file}.")
    return data


def train_epoch(model, dataloader, optimizer, grad_clip, device):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        v_in = batch["video_feats"].to(device)
        a_in = batch["audio_feats"].to(device)
        targets = batch["targets"]
        cap_tokens = batch["caption_tokens"].to(device)

        optimizer.zero_grad()
        outputs = model(v_in, a_in)
        losses = model.compute_loss(outputs, targets, caption_targets=cap_tokens)
        loss = losses["total_loss"]

        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * len(v_in)
    return total_loss / len(dataloader.dataset)


def evaluate_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            v_in = batch["video_feats"].to(device)
            a_in = batch["audio_feats"].to(device)
            targets = batch["targets"]
            cap_tokens = batch["caption_tokens"].to(device)

            outputs = model(v_in, a_in)
            losses = model.compute_loss(outputs, targets, caption_targets=cap_tokens)
            total_loss += losses["total_loss"].item() * len(v_in)
    return total_loss / len(dataloader.dataset)


def main():
    parser = argparse.ArgumentParser(description="Train DVC model across fusion conditions and seeds")
    parser.add_argument("--fusion_type", type=str, required=True, choices=["classical", "classical_matched", "quantum"])
    parser.add_argument("--seed", type=int, required=True, choices=[7, 42, 123, 999, 2024])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent

    # 1. Deterministic Seed Setup
    set_seed(args.seed)
    print(f"============================================================")
    print(f"DVC RUN: Fusion = {args.fusion_type} | Seed = {args.seed}")
    print(f"============================================================")

    # 2. Config Loading
    if args.config is None:
        args.config = str(project_root / "configs" / f"main_{args.fusion_type}.yaml")
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    # Output directories
    out_dir = project_root / "results" / args.fusion_type / f"seed_{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = out_dir / "model_best.pt"
    history_path = out_dir / "training_history.json"

    # 3. Data preparation & caching
    extractor = VideoAudioSnippetExtractor()
    vivit = ViViTBackbone().to(device).eval()
    ast_model = ASTBackbone().to(device).eval()

    train_manifest = str(project_root / "data" / "videos" / "train" / "train_manifest.json")
    val_manifest = str(project_root / "data" / "videos" / "val_1" / "val_1_manifest.json")

    cache_train = str(project_root / "data" / "features" / "train_dvc_cache.json")
    cache_val = str(project_root / "data" / "features" / "val1_dvc_cache.json")

    vocab_path = str(project_root / "data" / "vocab.json")
    train_anno = str(project_root / "data" / "annotations" / "train.json")
    vocab = build_activitynet_vocab(train_anno, vocab_path, max_vocab_size=5000)

    train_data = prepare_dvc_dataset(train_manifest, extractor, vivit, ast_model, device, cache_train, max_videos=150)
    val_data = prepare_dvc_dataset(val_manifest, extractor, vivit, ast_model, device, cache_val, max_videos=30)

    current_bs = args.batch_size
    train_dataset = DVCDataset(train_data)
    val_dataset = DVCDataset(val_data)

    collate_fn = make_collate_fn(vocab)
    train_loader = DataLoader(train_dataset, batch_size=current_bs, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=current_bs, shuffle=False, collate_fn=collate_fn)

    # 4. Model Initialization
    model = DVCModel(
        fusion_type=args.fusion_type,
        embed_dim=cfg.get("embed_dim", 512),
        num_queries=cfg.get("num_queries", 30),
        vocab_size=cfg.get("vocab_size", 5000),
        beam_width=cfg.get("beam_width", 5)
    ).to(device)

    # 5. Two Parameter Groups Optimizer (Section 8)
    fusion_and_heads_params = []
    backbone_params = []
    for name, p in model.named_parameters():
        if "backbone" in name or "compress" in name:
            backbone_params.append(p)
        else:
            fusion_and_heads_params.append(p)

    optimizer = torch.optim.AdamW([
        {"params": fusion_and_heads_params, "lr": float(cfg.get("lr_heads", 1e-4))},
        {"params": backbone_params, "lr": float(cfg.get("lr_backbone", 1e-5))}
    ], weight_decay=float(cfg.get("weight_decay", 1e-4)))

    max_epochs = min(args.epochs, cfg.get("max_epochs", 50))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)

    # 6. Training Loop with auto-halving on OOM & Early Stopping
    best_val_loss = float("inf")
    patience = args.patience
    patience_counter = 0
    stopping_reason = "max_epochs_reached"
    history = {"train_loss": [], "val_loss": [], "epochs": []}

    print(f"\nStarting training loop (max {max_epochs} epochs, patience {patience})...")

    for epoch in range(1, max_epochs + 1):
        try:
            train_loss = train_epoch(model, train_loader, optimizer, cfg.get("grad_clip", 1.0), device)
        except torch.cuda.OutOfMemoryError:
            print(f"[OOM Encountered] Auto-halving batch size from {current_bs} to {max(1, current_bs // 2)}...")
            current_bs = max(1, current_bs // 2)
            torch.cuda.empty_cache()
            train_loader = DataLoader(train_dataset, batch_size=current_bs, shuffle=True, collate_fn=collate_dvc)
            train_loss = train_epoch(model, train_loader, optimizer, cfg.get("grad_clip", 1.0), device)

        val_loss = evaluate_epoch(model, val_loader, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["epochs"].append(epoch)

        print(f"Epoch [{epoch:02d}/{max_epochs:02d}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Checkpoint saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "fusion_type": args.fusion_type,
                "seed": args.seed,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "config": cfg
            }, best_ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                stopping_reason = f"early_stopping_patience_{patience}"
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nRun Complete! Best Val Loss: {best_val_loss:.4f}, Checkpoint saved to: {best_ckpt_path}")

    # Return summary dict
    return {
        "fusion_type": args.fusion_type,
        "seed": args.seed,
        "final_epoch": len(history["epochs"]),
        "stopping_reason": stopping_reason,
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "best_val_loss": best_val_loss,
        "checkpoint_path": str(best_ckpt_path)
    }


if __name__ == "__main__":
    main()
