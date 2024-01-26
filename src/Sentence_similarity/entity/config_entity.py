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
