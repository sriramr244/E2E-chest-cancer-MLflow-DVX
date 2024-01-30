import os

# Disable parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ[
    "MLFLOW_TRACKING_URI"
] = "https://dagshub.com/sriramr244/E2E-sentence-similarity-MLflow-DVX.mlflow "
os.environ["MLFLOW_TRACKING_USERNAME"] = "sriramr244"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "a24a6cc6fb2a4c49dab0fe93311ed2485fd042b9"
