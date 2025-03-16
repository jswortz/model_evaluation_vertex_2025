import joblib
import numpy as np
import pickle

from google.cloud import storage
from google.cloud.aiplatform.prediction.predictor import Predictor
from google.cloud.aiplatform.utils import prediction_utils

import json

class CprPredictor(Predictor):

    def __init__(self):
        return

    def load(self, gcs_artifacts_uri: str):
        """Loads the preprocessor artifacts."""
        prediction_utils.download_model_artifacts(gcs_artifacts_uri)
        # gcs_client = storage.Client()
        # with open("model.joblib", 'wb') as gcs_model:
        #     gcs_client.download_blob_to_file(
        #         gcs_artifacts_uri + "/model.joblib", gcs_model
        #     )

        with open("model.joblib", "rb") as f:
            self._model = joblib.load("model.joblib")


    def predict(self, instances):
        outputs = self._model.predict(instances) 
        outputs = [list(output) for output in outputs] #convert array to list
        return {'predictions': outputs}
