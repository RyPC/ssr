"""Dataset loaders: GRID corpus, LRS2/LRS3, and custom self-recorded data.

On-disk layout conventions
---------------------------
CustomPhraseDataset (matches scripts/record_phrases.py output):
    root/
      <phrase label>/
        <clip dir (e.g. timestamp)>/
          frames.npy            # (T, H, W, 3) uint8, OR
          frame_0000.png, frame_0001.png, ...  # used if frames.npy absent

    The phrase label is the name of the label directory. The character
    vocabulary (index 0 = CTC blank) is built dynamically from the sorted
    unique characters across all discovered phrase labels.

GridCorpusDataset (GRID corpus convention):
    root/
      s1/                       # one dir per speaker
        video/*.mpg             # video files (assumed under a "video"
                                 # subdir if present, else directly in s1/)
        align/*.align           # word-timing transcripts, same basename
                                 # as the matching video
      s2/
        ...

    Each .align file has whitespace-separated lines:
        <start_frame> <end_frame> <word>
    per the standard GRID format. The label text is the space-joined
    sequence of non-"sil" words. Frames are decoded from the video via
    OpenCV; exact frame-rate alignment with the .align timestamps is not
    guaranteed to be frame-accurate here (GRID align files are in 1/100s
    units at 25fps in the original corpus) -- this loader takes the
    pragmatic approach of decoding the whole clip rather than trimming to
    word boundaries, since no real GRID data was available to validate
    against during implementation.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

CTC_BLANK = ""  # index 0 of vocab is always the CTC blank token


def _build_char_vocab(labels: list[str]) -> tuple[list[str], dict[str, int]]:
    chars = sorted(set("".join(labels)))
    vocab = [CTC_BLANK] + chars
    char_to_idx = {ch: i for i, ch in enumerate(vocab)}
    return vocab, char_to_idx


def _load_frames_dir(clip_dir: Path) -> np.ndarray:
    """Load a clip's frames as (T, H, W, 3) uint8 from frames.npy or PNGs."""
    npy_path = clip_dir / "frames.npy"
    if npy_path.exists():
        return np.load(npy_path)

    png_paths = sorted(clip_dir.glob("frame_*.png"))
    if not png_paths:
        raise FileNotFoundError(f"no frames.npy or frame_*.png found in {clip_dir}")

    try:
        import cv2

        frames = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in png_paths]
    except ImportError:
        from PIL import Image

        frames = [np.array(Image.open(p).convert("RGB")) for p in png_paths]

    return np.stack(frames)


def _frames_to_tensor(frames_thwc: np.ndarray) -> torch.Tensor:
    """(T, H, W, 3) uint8 -> (T, 3, H, W) float in [0, 1]."""
    frames = torch.from_numpy(frames_thwc).float() / 255.0
    return frames.permute(0, 3, 1, 2).contiguous()


class CustomPhraseDataset(Dataset):
    """Self-recorded phrase dataset (Phase 1, 50-200 phrases)."""

    def __init__(self, root: Path):
        self.root = Path(root)

        self.clips: list[tuple[Path, str]] = []
        for label_dir in sorted(p for p in self.root.iterdir() if p.is_dir()):
            label = label_dir.name
            for clip_dir in sorted(p for p in label_dir.iterdir() if p.is_dir()):
                self.clips.append((clip_dir, label))

        if not self.clips:
            raise ValueError(f"no clips found under {self.root}")

        labels = [label for _, label in self.clips]
        self.vocab, self.char_to_idx = _build_char_vocab(labels)

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int):
        clip_dir, label = self.clips[idx]
        frames_thwc = _load_frames_dir(clip_dir)
        frames = _frames_to_tensor(frames_thwc)
        label_indices = torch.tensor([self.char_to_idx[ch] for ch in label], dtype=torch.long)
        return frames, label_indices, label


class GridCorpusDataset(Dataset):
    def __init__(self, root: Path):
        self.root = Path(root)

        self.clips: list[tuple[Path, Path]] = []  # (video_path, align_path)
        for speaker_dir in sorted(p for p in self.root.iterdir() if p.is_dir()):
            video_dir = speaker_dir / "video" if (speaker_dir / "video").is_dir() else speaker_dir
            align_dir = speaker_dir / "align" if (speaker_dir / "align").is_dir() else speaker_dir

            for video_path in sorted(video_dir.glob("*.mpg")):
                align_path = align_dir / f"{video_path.stem}.align"
                if align_path.exists():
                    self.clips.append((video_path, align_path))

        if not self.clips:
            raise ValueError(f"no (video, align) pairs found under {self.root}")

        labels = [self._parse_align(align_path) for _, align_path in self.clips]
        self.vocab, self.char_to_idx = _build_char_vocab(labels)

    @staticmethod
    def _parse_align(align_path: Path) -> str:
        words = []
        for line in align_path.read_text().splitlines():
            parts = line.split()
            if len(parts) != 3:
                continue
            _, _, word = parts
            if word.lower() != "sil":
                words.append(word)
        return " ".join(words)

    @staticmethod
    def _decode_video(video_path: Path) -> np.ndarray:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        frames = []
        try:
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        finally:
            cap.release()

        if not frames:
            raise RuntimeError(f"no frames decoded from {video_path}")
        return np.stack(frames)

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int):
        video_path, align_path = self.clips[idx]
        label = self._parse_align(align_path)
        frames_thwc = self._decode_video(video_path)
        frames = _frames_to_tensor(frames_thwc)
        label_indices = torch.tensor([self.char_to_idx[ch] for ch in label], dtype=torch.long)
        return frames, label_indices, label


def pad_collate(batch: list[tuple[torch.Tensor, torch.Tensor, str]]) -> dict:
    """Pad a batch of (frames, label_indices, label_text) to max T / max label length.

    Returns a dict with `frames` (B, T_max, C, H, W), `frame_lengths` (B,),
    `labels` (B, L_max), `label_lengths` (B,) -- the lengths are required by
    torch.nn.CTCLoss.
    """
    frames_list, label_list, _texts = zip(*batch)

    frame_lengths = torch.tensor([f.shape[0] for f in frames_list], dtype=torch.long)
    label_lengths = torch.tensor([l.shape[0] for l in label_list], dtype=torch.long)

    t_max = int(frame_lengths.max())
    c, h, w = frames_list[0].shape[1:]
    frames_padded = torch.zeros(len(batch), t_max, c, h, w, dtype=frames_list[0].dtype)
    for i, f in enumerate(frames_list):
        frames_padded[i, : f.shape[0]] = f

    l_max = int(label_lengths.max())
    labels_padded = torch.zeros(len(batch), l_max, dtype=torch.long)
    for i, l in enumerate(label_list):
        labels_padded[i, : l.shape[0]] = l

    return {
        "frames": frames_padded,
        "frame_lengths": frame_lengths,
        "labels": labels_padded,
        "label_lengths": label_lengths,
    }
