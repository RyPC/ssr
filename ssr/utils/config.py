from dataclasses import dataclass, field


@dataclass
class CaptureConfig:
    camera_index: int = 0
    target_fps: int = 30
    mouth_roi_size: int = 96


@dataclass
class ModelConfig:
    encoder: str = "cnn_lstm"  # or "transformer"
    hidden_size: int = 256
    num_layers: int = 2
    vocab_size: int = 40  # phoneme/character vocab


@dataclass
class DecodingConfig:
    beam_width: int = 10
    top_k: int = 5


@dataclass
class CorrectionConfig:
    model_name: str = "small-transformer-reranker"


@dataclass
class PipelineConfig:
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    decoding: DecodingConfig = field(default_factory=DecodingConfig)
    correction: CorrectionConfig = field(default_factory=CorrectionConfig)
    latency_budget_ms: int = 300
