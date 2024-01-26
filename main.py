from Sentence_similarity import logger
from Sentence_similarity.pipeline.stage_01_data_ingestion import (
    DataIngestionTrainingPipeline,
)
from Sentence_similarity.pipeline.stage_02_base_model import BaseModelPipeline
from Sentence_similarity.pipeline.stage_03_model_trainer import TrainingPipeline
from Sentence_similarity.pipeline.stage_04_model_evaluation import EvaluationPipeline


STAGE_NAME = "DATA INGESTION STAGE"
try:
    logger.info((">" * 5) + f"{STAGE_NAME} Started" + ("<" * 5))
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(("->" * 5) + f"{STAGE_NAME} Completed" + ("<-" * 5))

except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "BASE MODEL PREPARATION STAGE"
try:
    logger.info(("+-" * 5) + f"{STAGE_NAME} Started" + ("-+" * 5))
    obj = BaseModelPipeline()
    obj.main()
    logger.info(("-" * 5) + f"{STAGE_NAME} Completed" + ("-" * 5))

except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "TRAINING"

try:
    logger.info(("+-" * 5) + f"{STAGE_NAME} Started" + ("-+" * 5))
    obj = TrainingPipeline()
    obj.main()
    logger.info(("-" * 5) + f"{STAGE_NAME} Completed" + ("-" * 5))

except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "EVALUATION"
try:
    logger.info(f"*******************")
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    model_evalution = EvaluationPipeline()
    model_evalution.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

except Exception as e:
    logger.exception(e)
    raise e
