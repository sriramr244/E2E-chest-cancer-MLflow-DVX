from Sentence_similarity import logger
from Sentence_similarity.entity.config_entity import TrainingConfig
from Sentence_similarity.utils.common import read_pickle, write_pickle
import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader


class TrainerLLM:
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.get_model()

    def get_model(self):
        self.model = read_pickle(self.config.base_model)

    def get_training_data(self):
        df = pd.read_csv(self.config.local_data_file)
        examples = []
        for _, row in df.iterrows():
            examples.append(
                InputExample(
                    texts=[row["Sentence 1"], row["Sentence 2"]],
                    label=float(row["Similarity"]),
                )
            )
        return DataLoader(
            examples, shuffle=True, batch_size=self.config.params_batch_size
        )

    def save_trained_model(self):
        write_pickle(self.config.updated_model_path, self.model)
        logger.info("New Model saved")

    def train(self):
        train_dataloader = self.get_training_data()
        train_loss = losses.CosineSimilarityLoss(self.model)
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=self.config.params_epochs,
            warmup_steps=self.config.params_warmup_steps,
        )
