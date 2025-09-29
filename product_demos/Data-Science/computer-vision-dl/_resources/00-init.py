# Databricks notebook source
## Initialization notebook

# This notebook sets up the backend. Please do not edit the notebook, it contains import and helpers for the demo

# COMMAND ----------

# DBTITLE 1,Configure the catalog, schema, and volume names
# MAGIC %run ../config

# COMMAND ----------

# DBTITLE 1,Common dbdemo setup
# MAGIC %run ../../../../_resources/00-global-setup-v2

# COMMAND ----------

dbutils.widgets.dropdown("add_pcb2_data", "false", ["true", "false"], "Add Second Data Tranche")

# COMMAND ----------

# DBTITLE 1,Setup the demo catalog, schema, and volume context
reset_all_data = dbutils.widgets.get("reset_all_data") == 'true'
add_pcb2_data = dbutils.widgets.get("add_pcb2_data") == 'true'
DBDemos.setup_schema(CATALOG, SCHEMA, reset_all_data, VOLUME_NAME)

# COMMAND ----------

# DBTITLE 1,Install data into volume
from pathlib import Path
import urllib.request
import tarfile

if reset_all_data or DBDemos.is_folder_empty(VOLUME_FOLDER):
  print(f"Loading raw data under {VOLUME_FOLDER}, please wait a few minutes as we extract all images...")

  # Clean volume
  dbutils.fs.rm(VOLUME_FOLDER, recurse=True)

  # Download the file
  url = "https://amazon-visual-anomaly.s3.amazonaws.com/VisA_20220922.tar"
  urllib.request.urlretrieve(url, f"{VOLUME_FOLDER}/VisA_20220922.tar")

  # Extract the tar file directly into the volume
  with tarfile.open(f"{VOLUME_FOLDER}/VisA_20220922.tar") as tar:
    pcb1_members = [m for m in tar.getmembers() if m.name.startswith("pcb1/")]
    tar.extractall(path=VOLUME_FOLDER, members=pcb1_members)
else:
  print("Data exists. Run with reset_all_data=true to force a data cleanup for your local demo.")

# COMMAND ----------

if add_pcb2_data:
  print(f"Loading second PCB dataset under {VOLUME_FOLDER}, please wait a few minutes as we extract all images...")
  
  assert (Path(VOLUME_FOLDER)/"VisA_20220922.tar").exists()

  # Clean volume
  if (Path(VOLUME_FOLDER) / 'pcb2').exists():
    dbutils.fs.rm(f"{VOLUME_FOLDER}/pcb2", recurse=True)

  # Extract the tar file directly into the volume
  with tarfile.open(f"{VOLUME_FOLDER}/VisA_20220922.tar") as tar:
    pcb2_members = [m for m in tar.getmembers() if m.name.startswith("pcb2/")]
    tar.extractall(path=VOLUME_FOLDER, members=pcb2_members)
