# E2E-Sentence-similarity-MLflow-DVX

## Overview

This project consists of four primary pipelines: Data Ingestion, Base Model Preparation, Model Trainer, and Model Evaluation. It's designed to facilitate the process of training and evaluating machine learning models, specifically focusing on sentence similarity using the cosine similarity score.

## Pipelines

### 1. Data Ingestion
- **Functionality**: Downloads the CSV data from Google Drive using `gdown`.

### 2. Base Model Preparation
- **Functionality**: Downloads the base model from the Sentence Transformer API. For subsequent runs, the model can be configured to download from an S3 bucket.

### 3. Model Trainer
- **Training Metric**: Utilizes cosine similarity score.

### 4. Model Evaluation
- **Benchmark Dataset**: The model is evaluated against a benchmark dataset.
- **Experiment Tracking**: The experiment is tracked using MLflow.

## Repository and MLflow Integration

- The repository is hosted on DagsHub.
- Local MLflow UI can be triggered using the command `mlflow ui`.
- The code is developed in a modular fashion for ease of maintenance and scalability.

## To Do

1. **DVC Implementation**: Integration with Data Version Control for better handling of datasets and model versioning.
2. **Data Download from MongoDB**: Implement functionality to ingest data from MongoDB.
3. **DagsHub Training Experiments**: Enhance integration with DagsHub for tracking training experiments.
4. **Containerization and API Service**: Containerize the application and implement it as an API service for ease of deployment and scalability.

