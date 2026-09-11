"""Preprocessing pipeline for ActivityNet Captions: 2.0s snippets, 25fps/native-fps, 224x224 crops, 16kHz mono audio."""

import os
import math
import torch
import torchaudio
import torchvision.transforms as T
import numpy as np
import av
from pathlib import Path


class VideoAudioSnippetExtractor:
    def __init__(
        self,
        snippet_duration: float = 2.0,
        target_fps: int = 25,
        frames_per_snippet: int = 32,
        crop_size: int = 224,
        audio_sr: int = 16000
    ):
        self.snippet_duration = snippet_duration
        self.target_fps = target_fps
        self.frames_per_snippet = frames_per_snippet
        self.crop_size = crop_size
        self.audio_sr = audio_sr

        # Standard video transforms for Kinetics/ViViT: Resize + CenterCrop + Normalization
        self.video_transform = T.Compose([
            T.Resize((crop_size, crop_size), antialias=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_snippets(self, video_path: str):
        """Extracts synchronous 2.0-second video and audio snippets from a media file.
        
        Returns:
            video_snippets: Tensor of shape [T, frames_per_snippet, 3, 224, 224] (float32)
            audio_snippets: Tensor of shape [T, samples_per_snippet] (float32, 16kHz mono)
            metadata: dict with duration, T snippets, and original media properties
        """
        container = av.open(video_path)

        # 1. Inspect streams
        video_stream = next((s for s in container.streams if s.type == "video"), None)
        audio_stream = next((s for s in container.streams if s.type == "audio"), None)

        if video_stream is None:
            raise ValueError(f"No video stream found in {video_path}")

        # Compute video duration
        v_duration = float(video_stream.duration * video_stream.time_base) if video_stream.duration else 0.0
        if v_duration <= 0.0:
            v_duration = float(container.duration / av.time.time_base) if container.duration else 0.0

        if v_duration < self.snippet_duration:
            num_snippets = 1
        else:
            num_snippets = max(1, int(math.floor(v_duration / self.snippet_duration)))

        # 2. Extract video frames
        frames = []
        for frame in container.decode(video=0):
            # Convert PyAV frame to RGB numpy array
            img = frame.to_ndarray(format="rgb24")
            # img shape: [H, W, 3] -> convert to tensor [3, H, W]
            t_frame = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            frames.append(t_frame)

        if not frames:
            raise ValueError(f"Failed to decode any video frames from {video_path}")

        total_frames = len(frames)
        all_frames = torch.stack(frames) # [total_frames, 3, H, W]

        # 3. Chunk video frames into T snippets
        video_snippets_list = []
        for i in range(num_snippets):
            start_t = i * self.snippet_duration
            end_t = (i + 1) * self.snippet_duration
            
            # Map time interval to frame indices
            start_idx = int(round((start_t / v_duration) * (total_frames - 1)))
            end_idx = int(round((end_t / v_duration) * (total_frames - 1)))
            if end_idx <= start_idx:
                end_idx = min(total_frames, start_idx + self.frames_per_snippet)

            snippet_raw = all_frames[start_idx:end_idx]
            if len(snippet_raw) == 0:
                snippet_raw = all_frames[max(0, start_idx - 1):start_idx + 1]

            # Resample temporally to exactly frames_per_snippet (e.g. 32 frames)
            indices = np.linspace(0, len(snippet_raw) - 1, self.frames_per_snippet).astype(int)
            sampled = snippet_raw[indices] # [32, 3, H, W]

            # Apply 224x224 crop & normalization
            processed = self.video_transform(sampled) # [32, 3, 224, 224]
            video_snippets_list.append(processed)

        video_snippets = torch.stack(video_snippets_list) # [T, 32, 3, 224, 224]

        # 4. Extract audio
        samples_per_snippet = int(self.snippet_duration * self.audio_sr)
        expected_total_samples = num_snippets * samples_per_snippet

        if audio_stream is not None:
            container.seek(0)
            audio_frames = []
            for frame in container.decode(audio=0):
                audio_frames.append(frame.to_ndarray())

            if audio_frames:
                raw_audio = np.concatenate(audio_frames, axis=1) # [channels, samples]
                audio_tensor = torch.from_numpy(raw_audio).float()
                
                # Convert to mono if multi-channel
                if audio_tensor.ndim > 1 and audio_tensor.shape[0] > 1:
                    audio_tensor = torch.mean(audio_tensor, dim=0, keepdim=True)
                elif audio_tensor.ndim == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)

                # Resample to 16 kHz
                orig_sr = audio_stream.codec_context.sample_rate
                if orig_sr != self.audio_sr:
                    resampler = torchaudio.transforms.Resample(orig_sr, self.audio_sr)
                    audio_tensor = resampler(audio_tensor)

                audio_tensor = audio_tensor.squeeze(0) # [total_samples]
            else:
                audio_tensor = torch.zeros(expected_total_samples)
        else:
            audio_tensor = torch.zeros(expected_total_samples)

        # Pad or trim audio to match expected length
        if audio_tensor.shape[0] < expected_total_samples:
            pad = torch.zeros(expected_total_samples - audio_tensor.shape[0])
            audio_tensor = torch.cat([audio_tensor, pad], dim=0)
        else:
            audio_tensor = audio_tensor[:expected_total_samples]

        # Reshape audio into snippets [T, samples_per_snippet]
        audio_snippets = audio_tensor.view(num_snippets, samples_per_snippet)

        container.close()

        metadata = {
            "video_path": video_path,
            "duration": v_duration,
            "num_snippets": num_snippets,
            "snippet_duration": self.snippet_duration,
            "video_snippet_shape": list(video_snippets.shape),
            "audio_snippet_shape": list(audio_snippets.shape),
            "audio_sr": self.audio_sr
        }

        return video_snippets, audio_snippets, metadata
