# Vertex Guide to Training and Deployment
This guide will walk through the process of training a custom model, evaluating it, and deploying it to Vertex AI.
## Overview

This repository contains a set of notebooks that demonstrate how to train, evaluate, and deploy models using Vertex AI. The notebooks are designed to be run sequentially, starting with the training notebook and progressing through the evaluation and deployment notebooks.

## Notebooks

The following notebooks are included in this repository:
* 00 - Custom Training
* 01 - Evaluation with Vertex SDK

## 00 - Custom Training

* [Notebook Link](00_TRAIN_pandas_sklearn_custom_container.ipynb)
* Guide to a simple model creation, used for evaluation downstream

## 01 - Evaluation with Vertex SDK

* [Notebook Link](01_Versioning_and_Evaluations_with_Vertex.ipynb)
* Runs through now to list models
* How to get model versions
* How to evaluate a version using the SDK
* How to extract info from evaluations
* How to compare evaluations
* How to get the evaluation metrics
