# Silent Speech Recognition (Camera-Based “Silent Typing”)

## 1. Goal

Build a mobile-first system that converts silent mouth movements (no audio) into text using:

- Front-facing camera video
- Lip/mouth motion modeling
- Sequence decoding + language correction model
- Optional personalization layer per user

Core idea:
Turn mouthing words into near-real-time text input, like speech-to-text but visual-only.

---

## 2. High-Level System Architecture

[Camera Video Stream]
        ↓
[Face + Mouth Tracking]
        ↓
[Visual Feature Encoder]
        ↓
[Sequence Model (Lip Reading)]
        ↓
[Beam Search Decoder → Top-K text candidates]
        ↓
[Language Correction Model (small LM / transformer)]
        ↓
[Final Text Output]

Optional:
[Personalization Layer (user-specific adaptation)]

---

## 3. Key Design Decisions

We are NOT doing raw end-to-end LLM from video.

Instead:
- structured pipeline
- constrained decoding
- correction model separate from vision model

---

## 4. Stage 1 — Camera + Mouth Tracking

Tools:
- MediaPipe Face Mesh
- OpenCV

Output:
- lip landmarks
- mouth openness
- mouth geometry features

---

## 5. Stage 2 — Visual Speech Model

Options:
- CNN + LSTM (MVP)
- Temporal Transformer (better)

Datasets:
- GRID corpus
- LRS2 / LRS3
- custom user dataset (recommended)

Output:
- token probabilities

---

## 6. Stage 3 — Beam Search Decoder

Goal: generate top-K sentence candidates.

Example:
1. can you pick up some milk
2. cam you pig up sum silk
3. can you big up some milk

Beam width: 5–20

---

## 7. Stage 4 — Language Correction Model

Input:
Top-K candidate sentences

Output:
Final corrected sentence

Example:
Input:
- cam you pig up sum silk
- can you pick up some milk

Output:
Can you pick up some milk?

Models:
- small transformer reranker
- lightweight LLM (optional)

Constraint:
Do NOT invent new meaning—only select or lightly correct.

---

## 8. Stage 5 — Personalization Layer

Stores:
- user phrases
- corrections
- vocabulary preferences

Methods:
- n-gram boosting
- fine-tuning small model
- embedding-based personalization

Example:
grb bb → grab boba

---

## 9. Stage 6 — Mobile Integration

iOS:
- Swift
- AVFoundation
- CoreML
- Vision framework

Android:
- CameraX
- TensorFlow Lite

Latency target:
<150–300ms

---

## 10. Stage 7 — Optional LLM Layer

Use only for:
- grammar cleanup
- formatting

Do NOT use for full interpretation.

Example:
"cn u pik up milk pls" → "Can you pick up milk please?"

---

## 11. Data Strategy

Phase 1:
- self-recorded dataset
- 50–200 phrases

Phase 2:
- public datasets (GRID, LRS2, LRS3)

Phase 3:
- real user corrections

---

## 12. MVP Roadmap

Week 1–2:
- face tracking + lip landmarks

Week 3–4:
- simple classifier

Week 5–6:
- beam search decoding

Week 7–8:
- correction model

Week 9+:
- personalization + mobile app

---

## 13. Risks

- lip reading ambiguity
- lighting sensitivity
- over-correction
- data scarcity

---

## 14. Success Criteria

- 20–50 phrase reliability
- real-time input
- usable in normal lighting
- low latency

---

## 15. Long-Term Vision

- silent texting interface
- accessibility tool
- AR input system
- alternative to voice dictation
