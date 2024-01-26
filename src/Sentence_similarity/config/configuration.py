from Sentence_similarity.constants import *
from Sentence_similarity.utils.common import read_yaml, create_directories
from Sentence_similarity.entity.config_entity import (
    DataIngestionConfig,
    BaseModelConfig,
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
