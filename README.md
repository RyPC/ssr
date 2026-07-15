# Silent Speech Recognition (SSR)

Camera-based silent typing: converts silent mouth movements into text.

See [ssr_project.md](ssr_project.md) for the full design doc.

## Pipeline

```
Camera Video Stream
  -> Face + Mouth Tracking      (ssr/capture)
  -> Visual Feature Encoder     (ssr/models)
  -> Sequence Model             (ssr/models)
  -> Beam Search Decoder        (ssr/decoding)
  -> Language Correction Model  (ssr/correction)
  -> Final Text Output
  -> Personalization Layer      (ssr/personalization)
```

## Project layout

- `ssr/capture` — face mesh + lip landmark extraction (MediaPipe/OpenCV)
- `ssr/models` — visual encoder + temporal sequence model (CNN+LSTM / transformer)
- `ssr/decoding` — beam search decoder over token probabilities
- `ssr/correction` — language correction / reranking model
- `ssr/personalization` — per-user vocabulary and correction adaptation
- `ssr/data` — dataset loading (GRID, LRS2/LRS3, custom)
- `ssr/utils` — shared config, logging, types
- `scripts` — CLI entry points (record, train, run pipeline)
- `tests` — unit tests
- `mobile/ios`, `mobile/android` — mobile integration (CoreML / TFLite)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quickstart

```bash
# 1. record a few clips per phrase (Phase 1 custom dataset)
python scripts/record_phrases.py --phrase "can you pick up some milk"

# 2. train the MVP CNN+LSTM model (CTC loss) on recorded clips
python scripts/train.py --data-dir ssr/data/raw --out model.pt

# 3. run the live pipeline: record -> decode -> correct -> personalize
python scripts/run_pipeline.py --model model.pt --camera 0
```

## Status

All stages have a working MVP implementation, end-to-end-tested with
synthetic data (`pytest`):

- **Capture** (`ssr/capture`) — MediaPipe Face Mesh lip tracking, mouth ROI
  cropping, OpenCV camera recording loop.
- **Models** (`ssr/models`) — CNN frame encoder + bidirectional LSTM,
  CTC-style token output.
- **Decoding** (`ssr/decoding`) — CTC prefix beam search -> top-K text
  candidates.
- **Correction** (`ssr/correction`) — offline plausibility reranking +
  light edit-distance word correction against a small built-in corpus
  (swap in a real corpus via `ssr/correction/corpus.txt`).
- **Personalization** (`ssr/personalization`) — per-user phrase/word
  substitution profile.
- **Data** (`ssr/data`) — custom phrase dataset loader (matches
  `record_phrases.py` output) and a GRID corpus loader.
- **Pipeline** (`ssr/pipeline.py`) — wires all of the above into
  `SilentSpeechPipeline.process_clip` / `.process_clip_rois`.

Not implemented: temporal transformer encoder variant, real
LRS2/LRS3 loaders, mobile (iOS/Android) integration — see
`mobile/ios/README.md` and `mobile/android/README.md`.
