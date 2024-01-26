from Sentence_similarity.config.configuration import ConfigurationManager
from Sentence_similarity.components.model_evaluation import ModelEval
from Sentence_similarity import logger


STAGE_NAME = "Evaluation stage"


class EvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            eval_config = config.get_eval_config()
            evaluation = ModelEval(eval_config)
            evaluation.eval()
            evaluation.save_score()
            evaluation.log_into_mlflow()

        except Exception as e:
            raise (e)
