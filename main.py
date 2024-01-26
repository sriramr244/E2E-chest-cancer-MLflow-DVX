from Sentence_similarity import logger
from Sentence_similarity.pipeline.stage_01_data_ingestion import (
    DataIngestionTrainingPipeline,
)


STAGE_NAME = "DATA INGESTION STAGE"
try:
    logger.info((">" * 5) + f"{STAGE_NAME} Started" + ("<" * 5))
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(("->" * 5) + f"{STAGE_NAME} Completed" + ("<-" * 5))

except Exception as e:
    logger.exception(e)
    raise e