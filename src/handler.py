
import csv
from io import StringIO
import json

from fastapi import Response

from google.cloud.aiplatform.prediction.handler import PredictionHandler

class CprHandler(PredictionHandler):
    """Default prediction handler for the prediction requests sent to the application."""

    async def handle(self, request):
        """Handles a prediction request."""
        request_body = await request.body()
        request_body_dict = json.loads(request_body)
        instances=request_body_dict["instances"]
        prediction_results = self._predictor.predict(instances)

        return Response(content=json.dumps(prediction_results))

