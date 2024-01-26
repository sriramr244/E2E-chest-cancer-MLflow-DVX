from Sentence_similarity.config.configuration import ConfigurationManager
from Sentence_similarity.components.base_model import BaseModelPrep
from Sentence_similarity import logger

STAGE_NAME = "BASE MODEL PREPARATION STAGE"


class BaseModelTrainingPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            base_model_config = config.get_BaseModel_config()
            Base_model_preparation = BaseModelPrep(config=base_model_config)
            Base_model_preparation.download_model()
        except Exception as e:
            raise e


if __name__ == "__main__":
    try:
        logger.info(("+-" * 5) + f"{STAGE_NAME} Started" + ("-+" * 5))
        obj = BaseModelTrainingPipeline()
        obj.main()
        logger.info(("-" * 5) + f"{STAGE_NAME} Completed" + ("-" * 5))

    except Exception as e:
        logger.exception(e)
        raise e
