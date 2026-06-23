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
python scripts/run_pipeline.py --camera 0
```
