from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_url: str
    local_data_file: Path


@dataclass(frozen=True)
class BaseModelConfig:
    model_name: str
    source_url: str
    local_model_file: Path
    root_dir: Path


@dataclass(frozen=True)
class TrainingConfig:
    base_model: Path
    updated_model_path: Path
    params_epochs: int
    params_batch_size: int
    params_warmup_steps: int
    local_data_file: Path


@dataclass(frozen=True)
class EvalConfig:
    test_model_path: Path
    naive_model_path: Path
    benchmark_dataset_path: Path
    mlflow_uri: str
