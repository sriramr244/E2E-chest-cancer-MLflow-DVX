from Sentence_similarity.constants import *
from Sentence_similarity.utils.common import read_yaml, create_directories
from Sentence_similarity.entity.config_entity import (
    DataIngestionConfig,
    BaseModelConfig,
    TrainingConfig,
    EvalConfig,
)


class ConfigurationManager:
    def __init__(
        self, config_file_path=CONFIG_FILE_PATH, params_file_path=PARAMS_FILE_PATH
    ):
        self.config = read_yaml(config_file_path)
        self.params = read_yaml(params_file_path)
        create_directories([self.config.artifacts_root])

    def get_DI_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])
        DI_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_url=config.source_url,
            local_data_file=config.local_data_file,
        )
        return DI_config

    def get_BaseModel_config(self) -> BaseModelConfig:
        config = self.config.base_model
        create_directories([config.root_dir])
        BM_config = BaseModelConfig(
            root_dir=config.root_dir,
            source_url=config.source_url,
            local_model_file=config.local_model_file,
            model_name=config.model_name,
        )
        return BM_config

    def get_Training_config(self) -> TrainingConfig:
        config = self.config.training
        create_directories([config.root_dir])
        Training_config = TrainingConfig(
            base_model=Path(self.config.base_model.local_model_file),
            updated_model_path=Path(config.updated_model_path),
            params_batch_size=self.params.batch_size,
            params_epochs=self.params.epochs,
            params_warmup_steps=self.params.warmup_steps,
            local_data_file=Path(self.config.data_ingestion.local_data_file),
        )
        return Training_config

    def get_eval_config(self) -> EvalConfig:
        return EvalConfig(
            test_model_path=Path(self.config.training.updated_model_path),
            naive_model_path=Path(self.config.base_model.local_model_file),
            benchmark_dataset_path=Path(self.config.evaluation.benchmark_data),
            mlflow_uri="https://dagshub.com/sriramr244/E2E-sentence-similarity-MLflow-DVX.mlflow",
        )
