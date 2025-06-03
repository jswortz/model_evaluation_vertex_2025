import sys

# argument structure
# [app.py project_id location endpoint_id bucket prediction_blob destination_blob batch_size]
import time

from pyspark.sql.functions import pandas_udf
import pandas as pd
from pyspark.sql.types import FloatType, ArrayType
import logging
import numpy as np
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType
from pyspark.sql import SparkSession

bucket = "model_experimentation_2025"
blob = "prediction_data/test.csv"


def main(
    project_id,
    location,
    endpoint_id,
    bucket,
    prediction_blob,
    destination_blob,
    batch_size,
):
    spark = SparkSession.builder.getOrCreate()
    spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", f"{batch_size}")

    df = spark.read.option("header", True).csv(f"gs://{bucket}/{prediction_blob}")

    prediction_data = df.drop("label")

    cols = prediction_data.columns

    predictions_formatted_data = prediction_data.withColumn(
        "data", F.array(*cols)
    ).drop(*cols)

    bp_args = {
        "project": project_id,
        "location": location,
        "endpoint_name": endpoint_id,
    }

    @pandas_udf(ArrayType(FloatType()))
    def make_vertex_batch_predict_fn(input_data: pd.Series) -> pd.Series:
        from google.cloud import aiplatform

        aiplatform.init(project=bp_args["project"], location=bp_args["location"])
        logging.info("aiplatform client established")

        model = aiplatform.Endpoint(bp_args["endpoint_name"])
        logging.info(f"Endpoint established: {model}")
        if input_data is None:
            return None
        float_input = input_data.apply(
            lambda string_features: string_features.astype(float)
        )
        reshaped_input = float_input.apply(
            lambda features: features.reshape(32, 32, 3) / 255.0
        )  # reshape and scape per the training
        list_typed_inputs = reshaped_input.apply(
            lambda reshaped_arrays: reshaped_arrays.tolist()
        )
        response = model.predict(list_typed_inputs)
        return pd.Series(response.predictions)

    predictions_df = predictions_formatted_data.withColumn(
        "predictions", make_vertex_batch_predict_fn("data")
    )
    epoch_time = time.time()

    output_blob = f"output_data/predictions_{epoch_time}.jsonl"

    # write the predictions to gcs
    predictions_df.write.option("lineSep", "\n").json(f"gs://{bucket}/{output_blob}")


if __name__ == "__main__":
    project_id = sys.argv[1]
    location = sys.argv[2]
    endpoint_id = sys.argv[3]
    bucket = sys.argv[4]
    prediction_blob = sys.argv[5]
    destination_blob = sys.argv[6]
    batch_size = sys.argv[7]

    main(
        project_id,
        location,
        endpoint_id,
        bucket,
        prediction_blob,
        destination_blob,
        batch_size,
    )
