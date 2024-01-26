from Sentence_similarity.config.configuration import ConfigurationManager
from Sentence_similarity.components.data_ingestion import DataIngestion
from Sentence_similarity import logger

STAGE_NAME = "DATA INGESTION STAGE"


class DataIngestionTrainingPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            data_ingestion_config = config.get_DI_config()
            data_ingestion = DataIngestion(config=data_ingestion_config)
            data_ingestion.download_file()
        except Exception as e:
            raise e


if __name__ == "__main__":
    try:
        logger.info(("+-" * 5) + "{STAGE_NAME} Started" + ("-+" * 5))
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(("-" * 5) + "{STAGE_NAME} Completed" + ("-" * 5))

    except Exception as e:
        logger.exception(e)
        raise e
