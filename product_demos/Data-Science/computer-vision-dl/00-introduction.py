# Databricks notebook source
# MAGIC %md
# MAGIC # TODO
# MAGIC - [X]: Update flow and notebooks
# MAGIC - [X]: Speed up data download / files
# MAGIC - [ ]: Add GIF to show setup of compute
# MAGIC - [ ]: Update bundle config prior to publish

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC # Computer Vision
# MAGIC
# MAGIC In this demo, we’ll show how Databricks can help you build, train, and serve a computer vision model for inspecting Printed Circuit Board (PCB) quality. Starting from the [Visual Anomaly (VisA)](https://registry.opendata.aws/visa/) detection dataset, you’ll see how to construct a complete pipeline that detects defects in PCB images—from raw data ingestion all the way to real-time inference.
# MAGIC
# MAGIC <div style="margin-top: 20px; text-align: left;">
# MAGIC <img width="300px" src="https://raw.githubusercontent.com/databricks-industry-solutions/cv-quality-inspection/main/images/PCB1.png">
# MAGIC <p style="margin-top: 5px;">Example of a Printed Circuit Board (PCB)</p>
# MAGIC </div>
# MAGIC
# MAGIC Computer vision has rapidly advanced thanks to larger datasets, powerful GPUs, pre-trained deep learning models, transfer learning, and user-friendly frameworks. However, deploying robust computer vision models involves several challenges:
# MAGIC
# MAGIC - Scalable data ingestion and preprocessing: Handling large volumes of image data efficiently.
# MAGIC
# MAGIC - Distributed training: Leveraging multiple GPUs to speed up complex model training.
# MAGIC
# MAGIC - MLOps and governance: Ensuring production readiness, traceability, and lifecycle management for models.
# MAGIC
# MAGIC - Streaming and real-time inference: Meeting strict SLAs for instant model predictions.
# MAGIC
# MAGIC Databricks Lakehouse simplifies each step by unifying data storage, scalable compute, and MLOps under one platform—enabling data scientists and engineers to focus on model development and inspection logic, rather than infrastructure or operational overhead.
# MAGIC
# MAGIC
# MAGIC <!-- Collect usage data (view). Remove it to disable collection. View README for more details.  -->
# MAGIC <img width="1px" src="https://www.google-analytics.com/collect?v=1&gtm=GTM-NKQ8TT7&tid=UA-163989034-1&cid=555&aip=1&t=event&ec=field_demos&ea=display&dp=%2F42_field_demos%2Ffeatures%2Fcomputer-vision-dl%2Fintro&dt=ML">

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Overview
# MAGIC
# MAGIC This notebook provides an overview of the demo. To build and serve a model to solve the challenges above, we will implement the following steps, which correspond closely with The Big Book of MLOps:
# MAGIC
# MAGIC 1. **Ingest** Use declarative pipelines to ingest the images into a silve table.  
# MAGIC
# MAGIC 2. **Explore** Quantify image quality and augment the images to increase the number of anomalous cases.
# MAGIC
# MAGIC 3. **Train** Use Serverless GPUs or Pytorch Lightning to train our models via transfer learning.
# MAGIC
# MAGIC 4. **Evaluate** Reload our logged model and quantify accuracy, batch inference latency, and improvements versus the champion. Use explainability methods to see how the model is making inferences and that it matches common sense.
# MAGIC
# MAGIC 5. **Infer** See how we can reload our model for batch inference. Set up a robust, concurrent serving endpoint and use it for real-time inference.
# MAGIC
# MAGIC Extra. **Distribute** This is a bonus notebook to show you how to distribute deep learning training on Databricks.
# MAGIC
# MAGIC Let's explain each component a bit more below.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## 1 - Ingest
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/computer-vision/deeplearning-cv-pcb-1.png?raw=true" width="500px" style="float: right" />
# MAGIC
# MAGIC Our first step is to ingest the images. We will leverage Databricks Autoloader to ingest the images and the labels (as csv file). Our data will be saved as a Delta Table, allowing easy governance with Databricks Unity Catalog.

# COMMAND ----------

# MAGIC %md 
# MAGIC ## 2 - Explore
# MAGIC
# MAGIC While deep learning has reduced the need for a large amount of exploratory data analysis (EDA) and feature engineering, it doesn't eliminate it altogether. This notebook goes through some simple quality assessment techniques for images. It then augments the images to increase the training data.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## 3 - Training
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/computer-vision/deeplearning-cv-pcb-2.png?raw=true" width="500px" style="float: right" />
# MAGIC
# MAGIC Now that our data is ready, we can leverage huggingface transformer library and do fine-tuning in an existing state-of-the art model.
# MAGIC
# MAGIC This is a very efficient first approach to quickly deliver results without having to go into pytorch details.
# MAGIC
# MAGIC We present two approaches for training - the first using Serverless GPUs to reduce training time and the second using a classic cluster with Pytorch Lightning.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## 4 - Evaluate
# MAGIC Having inferences and score on each image is great, but our operators will need to know which part is considered as damaged for manual verification and potential fix.
# MAGIC
# MAGIC We will be using SHAP as explainer to highlight pixels having the most influence on the model prediction. 
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/computer-vision/deeplearning-cv-pcb-explainer.png?raw=true" width="500px" style="float: right" />
# MAGIC
# MAGIC Now that our model is created and available in our MLFlow registry, we'll be able to use it. First we validate model performance and test it in batch inference.

# COMMAND ----------

# MAGIC %md
# MAGIC ##6 - Deploy
# MAGIC This notebook walks through three paths for deployment - the first being a severless GPU job, the second being realtime model serving, and the third being edge deployment using ONNX.
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/computer-vision/deeplearning-cv-pcb-3.png?raw=true" width="500px" style="float: right" />
# MAGIC
# MAGIC Open the [06-deploy]($./06-deploy) to see how to train your model & run distributed and realtime inferences.

# COMMAND ----------

# MAGIC %md 
# MAGIC ##7 - Distribute
# MAGIC
# MAGIC In this notebook we go further, using Ray and pytorch lightning for distributed training.
# MAGIC
# MAGIC More advanced use-cases might require leveraging other libraries such as pytorch or lightning. In this example, we will show how to implement a model fine tuning and inference with pytorch lightning.
# MAGIC
# MAGIC We will leverage delta-torch to easily build our torch dataset from the Delta tables.
# MAGIC
# MAGIC In addition, we'll also demonstrate how Databricks makes it easy to distribute the training on multiple GPUs using `TorchDistributor`.
# MAGIC
# MAGIC Open the [05-torch-lightning-training-and-inference]($./05-torch-lightning-training-and-inference) to see how to train your model & run distributed and realtime inferences.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC
# MAGIC This demo covers how Databricks is uniquely positioned to solve deep learning challenges using:
# MAGIC
# MAGIC - Spark to ingest and transform data at scale
# MAGIC - MLflow and Unity Catalog to simplify model training and governance
# MAGIC - Model Serving for one-click deployment for all use-cases, from batch to realtime and edge deployments
# MAGIC - Serverless GPUs to lower total cost of ownership for training pipelines
# MAGIC - Ray for distributed training across multiple GPUs and accelerating time to value
# MAGIC
# MAGIC Ready to Dive In? Let's Get Start with Ingestion!
