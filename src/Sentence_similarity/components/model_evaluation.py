from Sentence_similarity.entity.config_entity import EvalConfig
from Sentence_similarity.utils.common import read_pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.metrics import mean_squared_error
from Sentence_similarity.constants import *
import mlflow
import mlflow.sentence_transformers
from urllib.parse import urlparse
from Sentence_similarity import logger
from Sentence_similarity.utils.common import read_yaml


class ModelEval:
    def __init__(self, config: EvalConfig) -> None:
        self.config = config
        self.get_data()

    def get_model(self, model_path):
        return read_pickle(model_path)

    def get_data(self):
        df = pd.read_csv(self.config.benchmark_dataset_path, sep="\t")
        self.sentence1 = df["Sentence 1"]
        self.sentence2 = df["Sentence 2"]
        self.true_similarity = df["Similarity"]

    def get_no_datapoints(self):
        return len(self.sentence1)

    def do_eval_data(self, model_path):
        model = self.get_model(model_path)
        sentence1_embeddings = (
            model.encode(self.sentence1, convert_to_tensor=True).cpu().numpy()
        )
        sentence2_embeddings = (
            model.encode(self.sentence2, convert_to_tensor=True).cpu().numpy()
        )
        similarity = np.array(
            [
                cosine_similarity(i.reshape(1, -1), j.reshape(1, -1))
                for i, j in zip(sentence1_embeddings, sentence2_embeddings)
            ]
        )

        return {
            "similarity_scores": similarity,
            "rmse": np.sqrt(
                mean_squared_error(
                    np.array(sentence1_embeddings), np.array(sentence2_embeddings)
                )
            ),
        }

    def eval(self):
        self.evaluation_output = {}
        self.evaluation_output["Base_model"] = self.do_eval_data(
            self.config.naive_model_path
        )
        self.evaluation_output["Updated_model"] = self.do_eval_data(
            self.config.test_model_path
        )

    def save_score(self):
        print(self.evaluation_output)

    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        test_model = self.get_model(self.config.test_model_path)

        with mlflow.start_run():
            params = read_yaml(PARAMS_FILE_PATH)
            mlflow.log_params(
                {
                    "epochs": params.epochs,
                    "batch_size": params.batch_size,
                    "warmup_steps": params.warmup_steps,
                }
            )
            mlflow.log_metrics(
                {
                    "rmse_before_training": self.evaluation_output["Base_model"][
                        "rmse"
                    ],
                    "rmse_after_training": self.evaluation_output["Updated_model"][
                        "rmse"
                    ],
                    "length of training set": self.get_no_datapoints(),
                }
            )
            if tracking_url_type_store != "file":
                logger.info("Saving the model in dagshub")
                mlflow.sentence_transformers.log_model(
                    model=test_model,
                    registered_model_name="CERCLING_KEYWORD",
                    artifact_path="model",
                )
            else:
                logger.info("Saving the model locally")
                mlflow.sentence_transformers.log_model(
                    model=test_model,
                    artifact_path="model",
                )
        mlflow.end_run()
