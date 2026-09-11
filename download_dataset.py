"""Dataset acquisition script for ActivityNet Captions using yt-dlp with parallel workers."""

import os
import sys
import json
import argparse
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path


def download_single_video(vid_id, anno, output_dir, ffmpeg_path, yt_dlp_bin):
    yt_id = vid_id[2:] if vid_id.startswith("v_") else vid_id
    out_path = os.path.join(output_dir, f"{vid_id}.mp4")

    if os.path.exists(out_path) and os.path.getsize(out_path) > 10000:
        return vid_id, True, {
            "path": out_path,
            "duration": anno.get("duration", 0),
            "timestamps": anno.get("timestamps", []),
            "sentences": anno.get("sentences", [])
        }

    yt_url = f"https://www.youtube.com/watch?v={yt_id}"
    cmd = [
        str(yt_dlp_bin),
        "--ffmpeg-location", str(ffmpeg_path),
        "-f", "bestvideo[height<=360]+bestaudio/best[height<=360]",
        "--merge-output-format", "mp4",
        "--socket-timeout", "15",
        "--retries", "2",
        "-o", out_path,
        yt_url
    ]

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        if res.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 10000:
            return vid_id, True, {
                "path": out_path,
                "duration": anno.get("duration", 0),
                "timestamps": anno.get("timestamps", []),
                "sentences": anno.get("sentences", [])
            }
        else:
            if os.path.exists(out_path):
                try:
                    os.remove(out_path)
                except OSError:
                    pass
            return vid_id, False, None
    except Exception:
        if os.path.exists(out_path):
            try:
                os.remove(out_path)
            except OSError:
                pass
        return vid_id, False, None


def download_split(split_name, annotations_file, output_dir, target_count, ffmpeg_path, max_workers=4):
    print(f"\n--- Downloading split: {split_name} (target: {target_count}, workers: {max_workers}) ---")
    os.makedirs(output_dir, exist_ok=True)

    with open(annotations_file, "r") as f:
        data = json.load(f)

    successful = {}
    missing_unavailable = 0
    total_attempted = 0
    lock = threading.Lock()

    project_root = Path(__file__).resolve().parent
    bin_dir = project_root / "bin"
    yt_dlp_bin = project_root / "venv" / "bin" / "yt-dlp"

    # First check already existing files
    for vid_id, anno in data.items():
        out_path = os.path.join(output_dir, f"{vid_id}.mp4")
        if os.path.exists(out_path) and os.path.getsize(out_path) > 10000:
            successful[vid_id] = {
                "path": out_path,
                "duration": anno.get("duration", 0),
                "timestamps": anno.get("timestamps", []),
                "sentences": anno.get("sentences", [])
            }
            total_attempted += 1
            if len(successful) >= target_count:
                break

    print(f"[{split_name}] Found {len(successful)} already downloaded videos.")

    if len(successful) < target_count:
        candidates = [(vid_id, anno) for vid_id, anno in data.items() if vid_id not in successful]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_vid = {}
            cand_iter = iter(candidates)

            # Prime the pool
            for _ in range(max_workers * 2):
                try:
                    vid_id, anno = next(cand_iter)
                    future = executor.submit(download_single_video, vid_id, anno, output_dir, bin_dir, yt_dlp_bin)
                    future_to_vid[future] = vid_id
                except StopIteration:
                    break

            while future_to_vid and len(successful) < target_count:
                done_future = next(as_completed(future_to_vid))
                vid_id = future_to_vid.pop(done_future)

                with lock:
                    total_attempted += 1

                try:
                    res_id, is_success, metadata = done_future.result()
                    if is_success:
                        with lock:
                            successful[res_id] = metadata
                            print(f"[{split_name}] ({len(successful)}/{target_count}) Succeeded: {res_id}")
                    else:
                        with lock:
                            missing_unavailable += 1
                            print(f"[{split_name}] Missing/Unavailable: {res_id} (Missing count: {missing_unavailable})")
                except Exception as e:
                    with lock:
                        missing_unavailable += 1

                # Schedule next candidate if target not yet reached
                if len(successful) < target_count:
                    try:
                        next_vid_id, next_anno = next(cand_iter)
                        new_future = executor.submit(download_single_video, next_vid_id, next_anno, output_dir, bin_dir, yt_dlp_bin)
                        future_to_vid[new_future] = next_vid_id
                    except StopIteration:
                        pass

    manifest_path = os.path.join(output_dir, f"{split_name}_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(successful, f, indent=2)

    missing_rate = (missing_unavailable / total_attempted * 100.0) if total_attempted > 0 else 0.0
    return {
        "split": split_name,
        "total_attempted": total_attempted,
        "successful": len(successful),
        "missing_unavailable": missing_unavailable,
        "missing_rate": missing_rate,
        "manifest_path": manifest_path
    }


def main():
    parser = argparse.ArgumentParser(description="Download ActivityNet Captions videos with yt-dlp")
    parser.add_argument("--train_count", type=int, default=600, help="Target count for train split")
    parser.add_argument("--val1_count", type=int, default=200, help="Target count for val_1 split")
    parser.add_argument("--val2_count", type=int, default=200, help="Target count for val_2 split")
    parser.add_argument("--workers", type=int, default=4, help="Number of concurrent download workers")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    data_dir = project_root / "data"
    anno_dir = data_dir / "annotations"
    videos_dir = data_dir / "videos"
    evidence_dir = project_root / "gate_evidence"
    bin_dir = project_root / "bin"

    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(evidence_dir, exist_ok=True)

    results = []
    # Train split
    r_train = download_split(
        "train",
        str(anno_dir / "train.json"),
        str(videos_dir / "train"),
        args.train_count,
        bin_dir,
        max_workers=args.workers
    )
    results.append(r_train)

    # Val 1 split
    r_val1 = download_split(
        "val_1",
        str(anno_dir / "val_1.json"),
        str(videos_dir / "val_1"),
        args.val1_count,
        bin_dir,
        max_workers=args.workers
    )
    results.append(r_val1)

    # Val 2 split (strictly required per spec Section 3)
    r_val2 = download_split(
        "val_2",
        str(anno_dir / "val_2.json"),
        str(videos_dir / "val_2"),
        args.val2_count,
        bin_dir,
        max_workers=args.workers
    )
    results.append(r_val2)

    # Generate evidence report
    evidence_lines = [
        "=" * 60,
        "GATE 2 ACCEPTANCE TEST: DATASET ACQUISITION REPORT",
        "=" * 60,
        f"Timestamp: {datetime.now().isoformat()}",
        f"Source: Official ActivityNet Captions via yt-dlp",
        "",
        f"{'Split':<10} | {'Attempted':<10} | {'Succeeded':<10} | {'Missing/Deleted':<16} | {'Missing Rate (%)':<16}",
        "-" * 72
    ]

    total_attempted = sum(r["total_attempted"] for r in results)
    total_successful = sum(r["successful"] for r in results)
    total_missing = sum(r["missing_unavailable"] for r in results)
    overall_missing_rate = (total_missing / total_attempted * 100.0) if total_attempted > 0 else 0.0

    for r in results:
        evidence_lines.append(
            f"{r['split']:<10} | {r['total_attempted']:<10} | {r['successful']:<10} | {r['missing_unavailable']:<16} | {r['missing_rate']:<16.2f}"
        )
    evidence_lines.append("-" * 72)
    evidence_lines.append(
        f"{'OVERALL':<10} | {total_attempted:<10} | {total_successful:<10} | {total_missing:<16} | {overall_missing_rate:<16.2f}"
    )
    evidence_lines.append("")
    evidence_lines.append(f"val_1 Split Present: {'YES' if r_val1['successful'] > 0 else 'NO'}")
    evidence_lines.append(f"val_2 Split Present (Required): {'YES' if r_val2['successful'] > 0 else 'NO'}")
    evidence_lines.append("")
    passed = (r_train["successful"] > 0 and r_val1["successful"] > 0 and r_val2["successful"] > 0)
    evidence_lines.append(f"GATE 2 VERIFICATION RESULT: {'PASSED' if passed else 'FAILED'}")
    evidence_lines.append("=" * 60)

    report_text = "\n".join(evidence_lines) + "\n"
    evidence_file = evidence_dir / "gate2_dataset_acquisition.txt"
    with open(evidence_file, "w") as ef:
        ef.write(report_text)

    print("\n" + report_text)
    print(f"Evidence file saved to: {evidence_file}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
