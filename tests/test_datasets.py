import numpy as np
import torch

from ssr.data.datasets import CustomPhraseDataset, pad_collate


def _make_clip(clip_dir, num_frames, size=32):
    clip_dir.mkdir(parents=True)
    frames = np.random.randint(0, 256, size=(num_frames, size, size, 3), dtype=np.uint8)
    np.save(clip_dir / "frames.npy", frames)


def _make_dataset(tmp_path):
    _make_clip(tmp_path / "hello" / "clip1", num_frames=5)
    _make_clip(tmp_path / "world" / "clip1", num_frames=7)
    return tmp_path


def test_custom_phrase_dataset_len_and_getitem(tmp_path):
    root = _make_dataset(tmp_path)
    dataset = CustomPhraseDataset(root)

    assert len(dataset) == 2
    assert "" == dataset.vocab[0]  # blank token at index 0

    for i in range(len(dataset)):
        frames, label_indices, label_text = dataset[i]
        assert label_text in {"hello", "world"}
        assert frames.dtype == torch.float32
        assert frames.shape[1:] == (3, 32, 32)
        assert frames.min() >= 0.0 and frames.max() <= 1.0
        assert label_indices.dtype == torch.long
        assert label_indices.shape[0] == len(label_text)


def test_pad_collate(tmp_path):
    root = _make_dataset(tmp_path)
    dataset = CustomPhraseDataset(root)

    batch = [dataset[0], dataset[1]]
    out = pad_collate(batch)

    t_max = max(batch[0][0].shape[0], batch[1][0].shape[0])
    l_max = max(batch[0][1].shape[0], batch[1][1].shape[0])

    assert out["frames"].shape == (2, t_max, 3, 32, 32)
    assert out["frame_lengths"].tolist() == [batch[0][0].shape[0], batch[1][0].shape[0]]
    assert out["labels"].shape == (2, l_max)
    assert out["label_lengths"].tolist() == [batch[0][1].shape[0], batch[1][1].shape[0]]
