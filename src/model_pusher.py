import mlflow
import mlflow.transformers
import logging
import os
import torch
from transformers import pipeline, Trainer
from utils.model_utils import get_best_f1
from Config.constant import training_args, model_name
import dagshub

class ModelPusher:
    def __init__(self): 
        dagshub.init(repo_owner='ogunladeesther01', 
                     repo_name='Shopease_Sentiment_Project', 
                     mlflow=True)
        
        self.experiment_name = "Shopease_Sentiment_Analysis"
        mlflow.set_experiment(self.experiment_name)

    def updated_model_pusher(self, trainer, metrics):
        try:
            new_f1 = metrics['eval_f1']
            old_f1 = get_best_f1(self.experiment_name)

            with mlflow.start_run() as run:

                # logging metrics and parameters
                if old_f1 is None or new_f1 > old_f1:
                    mlflow.log_metric("accuracy", metrics['eval_accuracy'])
                    mlflow.log_metric("loss", metrics['eval_loss'])
                    mlflow.log_metric("f1", new_f1)
                    mlflow.log_param("epochs", training_args.num_train_epochs)
                    mlflow.log_param("learning_rate", training_args.learning_rate)
                    mlflow.log_param("batch_size", training_args.per_device_train_batch_size)

                    # creating a hugging face pipeline for sentiment analysis. Which is how it would be stored.
                    sentiment_pipeline = pipeline(
                        task="text-classification",
                        model=trainer.model,
                        tokenizer=model_name,
                        return_all_scores=True,
                    )

                    # logging the pipeline using mlflow
                    mlflow.transformers.log_model(
                        transformers_model=sentiment_pipeline,
                        artifact_path="model",
                        registered_model_name="sentiment_model"
                    )

                    logging.info("New model + Tokenizer has been logged and registered into mlflow")
                else:
                    logging.info("New model did not outperform the existing model...")
        except Exception as e:
            logging.error(f"Error occured during model pushing: {e}")