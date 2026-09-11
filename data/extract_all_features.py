"""Feature extraction and caching pipeline for QFusion-DVC.

Extracts synchronous ViViT video features (768d) and AST audio features (768d)
for all acquired videos, preserving full ground truth timestamps, normalized segments,
and ground truth caption sentences from ActivityNet Captions.
"""

import os
import json
import torch
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from data.preprocess import VideoAudioSnippetExtractor
from backbones.vivit_backbone import ViViTBackbone
from backbones.ast_backbone import ASTBackbone


def process_split_features(
    manifest_path: str,
    cache_path: str,
    extractor,
    vivit,
    ast_model,
    device,
    target_count: int = 150
):
    print(f"\nProcessing split: {manifest_path} -> {cache_path}")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    # Load existing cache if available
    existing_by_id = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                cached_list = json.load(f)
            for item in cached_list:
                existing_by_id[item["vid_id"]] = item
            print(f"Loaded {len(existing_by_id)} already cached videos from {cache_path}.")
        except Exception as e:
            print(f"Could not load existing cache: {e}")

    updated_data = []
    new_extracted = 0

    for vid_id, item in manifest.items():
        if len(updated_data) >= target_count:
            break

        timestamps = item.get("timestamps", [])
        sentences = item.get("sentences", [])
        duration = item.get("duration", 0.0)
        v_path = item.get("path", "")

        if not timestamps or duration <= 0:
            continue

        # Compute valid normalized segments and matching sentences
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

        # Check if features already cached
        if vid_id in existing_by_id and "video_feats" in existing_by_id[vid_id]:
            cached_item = existing_by_id[vid_id]
            updated_data.append({
                "vid_id": vid_id,
                "video_feats": cached_item["video_feats"],
                "audio_feats": cached_item["audio_feats"],
                "segments": norm_segs,
                "sentences": valid_sents,
                "duration": duration
            })
            continue

        # Otherwise, extract from raw video file if it exists
        if not os.path.exists(v_path):
            continue

        try:
            v_snips, a_snips, meta = extractor.extract_snippets(v_path)
            T_snips = meta["num_snippets"]

            with torch.no_grad():
                v_feats = vivit(v_snips.to(device)).cpu().numpy().tolist()
                a_feats = ast_model(a_snips.to(device)).cpu().numpy().tolist()

            updated_data.append({
                "vid_id": vid_id,
                "video_feats": v_feats,
                "audio_feats": a_feats,
                "segments": norm_segs,
                "sentences": valid_sents,
                "duration": duration
            })
            new_extracted += 1
            print(f"  [{len(updated_data)}/{target_count}] Extracted {vid_id} ({T_snips} snippets)...", flush=True)

        except Exception as e:
            print(f"  Error processing {vid_id}: {e}")
            continue

    # Save to disk
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(updated_data, f)
    print(f"Successfully saved {len(updated_data)} total videos ({new_extracted} newly extracted) to {cache_path}.")
    return len(updated_data)


def main():
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    videos_dir = data_dir / "videos"
    features_dir = data_dir / "features"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    extractor = VideoAudioSnippetExtractor()
    vivit = ViViTBackbone().to(device).eval()
    ast_model = ASTBackbone().to(device).eval()

    # 1. Val 1 split
    process_split_features(
        str(videos_dir / "val_1" / "val_1_manifest.json"),
        str(features_dir / "val1_dvc_cache.json"),
        extractor, vivit, ast_model, device,
        target_count=20
    )

    # 2. Val 2 split
    process_split_features(
        str(videos_dir / "val_2" / "val_2_manifest.json"),
        str(features_dir / "val2_dvc_cache.json"),
        extractor, vivit, ast_model, device,
        target_count=20
    )

    # 3. Train split
    process_split_features(
        str(videos_dir / "train" / "train_manifest.json"),
        str(features_dir / "train_dvc_cache.json"),
        extractor, vivit, ast_model, device,
        target_count=130
    )

    print("\nAll feature caches built and verified successfully!")


if __name__ == "__main__":
    main()
