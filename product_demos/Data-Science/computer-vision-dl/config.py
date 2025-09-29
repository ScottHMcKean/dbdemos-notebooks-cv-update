# Databricks notebook source
## Configuration file
# This file can be used to update default settings. We do not recommend to change the catalog here as it won't impact all the demo resources. Intead, please re-install the demo with a specific catalog and schema using dbdemos.install("computer-vision", catalog="..", schema="...")

# COMMAND ----------

CATALOG = "shm"
SCHEMA = DBNAME = DB = "dbdemos_computer_vision"
VOLUME_NAME = "pcb_training_data"
VOLUME_FOLDER =  f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_NAME}"

# COMMAND ----------

SHARPNESS_THRESHOLD = 150
IMAGE_RESIZE = 256

# COMMAND ----------

TEST_SIZE = 0.2
SEED = 42
