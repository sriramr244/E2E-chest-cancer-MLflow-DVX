import os
import gdown
from Sentence_similarity import logger
from Sentence_similarity.entity.config_entity import BaseModelConfig
from sentence_transformers import SentenceTransformer
from Sentence_similarity.utils.common import write_pickle


class BaseModelPrep:
    def __init__(self, config: BaseModelConfig) -> None:
        self.config = config

    def download_model(self):
        try:
            if self.config.source_url == "first_time":
                model = SentenceTransformer("paraphrase-MiniLM-L12-v2")
                write_pickle(self.config.local_model_file, model)
                logger.info(
                    "paraphrase-MiniLM-L12-v2 was downloaded from SentenceTransformer"
                )
            else:
                logger.info("Private model was downloaded from SentenceTransformer")
                "Write logic for downloading the model from different places"
        except Exception as e:
            raise e
