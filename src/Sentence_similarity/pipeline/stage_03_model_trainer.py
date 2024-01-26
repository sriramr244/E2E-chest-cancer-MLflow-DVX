from Sentence_similarity.components.training import TrainerLLM
from Sentence_similarity.config.configuration import ConfigurationManager
from Sentence_similarity import logger


STAGE_NAME = "Training"


class TrainingPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            train_config = config.get_Training_config()
            train_obj = TrainerLLM(config=train_config)
            train_obj.train()
            train_obj.save_trained_model()
        except Exception as e:
            raise e


if __name__ == "__main__":
    try:
        logger.info(("+-" * 5) + f"{STAGE_NAME} Started" + ("-+" * 5))
        obj = TrainingPipeline()
        obj.main()
        logger.info(("-" * 5) + f"{STAGE_NAME} Completed" + ("-" * 5))

    except Exception as e:
        logger.exception(e)
        raise e
