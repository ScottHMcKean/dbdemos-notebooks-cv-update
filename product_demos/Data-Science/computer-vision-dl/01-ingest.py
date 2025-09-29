# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC # Ingestion
# MAGIC
# MAGIC This is the pipeline we will be building. We ingest 2 datasets, namely:
# MAGIC
# MAGIC * The raw images (jpg) containing PCB
# MAGIC * The label, the type of anomalies saved as CSV files
# MAGIC
# MAGIC We will first focus on building a data pipeline to incrementally load this data and create a final Gold table.
# MAGIC
# MAGIC This table will then be used to train a ML Classification model to learn to detect anomalies in our images in real time!
# MAGIC
# MAGIC *Note that this demo leverages the standard spark API. You could also implement this same pipeline in pure SQL leveraging [Declarative Pipelines](https://www.databricks.com/product/data-engineering/lakeflow-declarative-pipelines). For more details on DLT, install `dbdemos.install('dlt-loans')`*
# MAGIC
# MAGIC <div style="background-color: #d9f0ff; border-radius: 10px; padding: 15px; margin: 10px 0; font-family: Arial, sans-serif;">
# MAGIC   <strong>Note:</strong> This notebook has been tested on non-GPU accelerated serverless v2. <br/>
# MAGIC </div>
# MAGIC
# MAGIC <!-- Collect usage data (view). Remove it to disable collection. View README for more details.  -->
# MAGIC <img width="1px" src="https://www.google-analytics.com/collect?v=1&gtm=GTM-NKQ8TT7&tid=UA-163989034-1&cid=555&aip=1&t=event&ec=field_demos&ea=display&dp=%2F42_field_demos%2Ffeatures%2Fcomputer-vision-dl%2Fetl&dt=ML">

# COMMAND ----------

# MAGIC %run ./_resources/00-init $reset_all_data=false

# COMMAND ----------

print(f"Training data has been installed in the volume {VOLUME_FOLDER}")

# COMMAND ----------

from pyspark.sql import functions as F

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Reviewing the incoming dataset
# MAGIC
# MAGIC The dataset was downloaded for you automatically and is available in cloud your dbfs storage folder. Let's explore the data:

# COMMAND ----------

# Our ingestion path
display(dbutils.fs.ls(f"{VOLUME_FOLDER}/pcb1/Data/Images/Normal"))

# COMMAND ----------

# Our labels
display(dbutils.fs.head(f"{VOLUME_FOLDER}/pcb1/image_anno.csv"))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Ingesting raw images with Databricks Autoloader
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/computer-vision/deeplearning-cv-pcb-flow-1.png?raw=true" width="700px" style="float: right"/>
# MAGIC
# MAGIC The first step is to load the individual JPG images. This can be quite challenging at scale, especially for incremental load (consume only the new ones).
# MAGIC
# MAGIC Databricks Autoloader can easily handle all type of formats and make it very easy to ingest new datasets.
# MAGIC
# MAGIC Autoloader will guarantee that only new files are being processed while scaling with millions of individual images. 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load binary files with Auto Loader
# MAGIC
# MAGIC We can now use the Auto Loader to load images, and a spark function to create the label column. Autoloader will automatically create the table and tune it accordingly, disabling compression for binary among other.
# MAGIC
# MAGIC We can also very easily display the content of the images and the labels as a table.

# COMMAND ----------

dbutils.fs.rm(f"{VOLUME_FOLDER}/stream/labels_checkpoint", True) 

# COMMAND ----------

(
    spark.readStream.format("cloudFiles")
    .option("cloudFiles.format", "binaryFile")
    .option("pathGlobFilter", "*.JPG")
    .option("recursiveFileLookup", "true")
    .option("cloudFiles.schemaLocation", f"{VOLUME_FOLDER}/stream/pcb_schema")
    .option("cloudFiles.maxFilesPerTrigger", 200)
    .load(f"{VOLUME_FOLDER}/pcb1/Data/Images")
    .withColumn("filename", F.substring_index(F.col("path"), "/", -1))
    .writeStream.trigger(availableNow=True)
    .option("checkpointLocation", f"{VOLUME_FOLDER}/stream/pcb_checkpoint")
    .toTable(f"{CATALOG}.{SCHEMA}.images_bronze")
    .awaitTermination()
)

display(spark.table(f"{CATALOG}.{SCHEMA}.images_bronze").limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load CSV label files with Auto Loader
# MAGIC CSV files can easily be loaded using Databricks [Auto Loader](https://docs.databricks.com/ingestion/auto-loader/index.html), including schema inference and evolution. We can immediately start to see a common problem with real world classification too - there are only a few examples of anomalous circuit boards!

# COMMAND ----------

(
    spark.readStream.format("cloudFiles")
    .option("cloudFiles.format", "csv")
    .option("header", True)
    .option("cloudFiles.schemaLocation", f"{VOLUME_FOLDER}/stream/labels_schema")
    .load(f"{VOLUME_FOLDER}/pcb1/image_anno.csv")
    .withColumn("filename", F.substring_index(F.col("image"), "/", -1))
    .select("filename", "label")
    .withColumnRenamed("label", "labelDetail")
    .writeStream.trigger(availableNow=True)
    .option("checkpointLocation", f"{VOLUME_FOLDER}/stream/labels_checkpoint")
    .toTable(f"{CATALOG}.{SCHEMA}.labels_bronze")
    .awaitTermination()
)

display(spark.table(f"{CATALOG}.{SCHEMA}.labels_bronze"
    ).groupBy("labelDetail").count())

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Merge the labels and the images tables
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/computer-vision/deeplearning-cv-pcb-flow-2.png?raw=true" width="700px" style="float: right"/>
# MAGIC
# MAGIC Note that we're working with delta tables to make the ingestion simple. You don't have to worry about individual small images anymore. We can do the join operation either in python or SQL. Additionally, because we have so few labels and want to get to value quickly, we compress the detailed labels into a single 'damaged' class. This is common practice when getting started - get high accuracy on a simpler problem and then move into more detail as we get more data.

# COMMAND ----------

(
    spark.table(f"{CATALOG}.{SCHEMA}.images_bronze")
    .join(
        spark.table(f"{CATALOG}.{SCHEMA}.labels_bronze"),
        on="filename",
        how="inner"
    )
    .withColumn(
        "label",
        F.when(F.col("labelDetail") == "normal", "normal").otherwise("damaged")
    )
    .write.mode("overwrite")
    .saveAsTable(f"{CATALOG}.{SCHEMA}.images_silv")
)
