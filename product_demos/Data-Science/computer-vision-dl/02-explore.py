# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC # Exploration and Feature Engineering
# MAGIC
# MAGIC This notebook starts to inspect the data, augmenting it with a focus on the anomalies, and generating a gold table for model training. This gold table will then be used to train a classification model for real time anomaly detection.
# MAGIC
# MAGIC In the last step, we ingested a silver table with images and labels, with a compressed binary label ('normal' vs 'damaged'). In this step, we are going to explore the data and apply some common image augmentation techniques to generate a gold table that is ready for training.
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/computer-vision/deeplearning-cv-pcb-flow-3.png?raw=true" width="700px" style="float: right"/>
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

# MAGIC %pip install opencv-python

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exploratory Data Analysis
# MAGIC Even with deep learning, data exploration can make or break a project. This section of the notebook runs through some basic analyses that are supercharged with Spark for scale.
# MAGIC
# MAGIC - Visualize the data
# MAGIC - Use descriptive statistics

# COMMAND ----------

from PIL import Image
import io

import pyspark.sql.functions as F
from pyspark.sql.window import Window
import matplotlib.pyplot as plt

# COMMAND ----------

df = spark.table(f"{CATALOG}.{SCHEMA}.images_silv")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Visualize Your Data!
# MAGIC First thing we is too look at the images. This code generates a grid of three random normal and damaged images. It can be run multiple times to see various normal and damaged examples with no worries about scale since we are randomly selecting using Spark. This process could easily be supercharged using Databricks Apps.

# COMMAND ----------

window = Window.partitionBy("label").orderBy(F.rand())
random_images = df.withColumn("row_num", F.row_number().over(window)).filter(F.col("row_num") <= 3).toPandas()

imgs = []
labels = []

for i, row in random_images.iterrows():
    img = Image.open(io.BytesIO(row["content"]))  # Use your byte column name
    imgs.append(img)
    labels.append(row["labelDetail"])

plt.figure(figsize=(12, 8))
for idx, (img, label) in enumerate(zip(imgs, labels)):
    ax = plt.subplot(2, 3, idx + 1)
    plt.imshow(img)
    plt.title(label)
    plt.axis("off")

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Statistical Analysis
# MAGIC Our deep learning solution operates on raw number values and it is worth doing some statistical analysis on brightness and contrast and seeing if normalization is required.

# COMMAND ----------

# MAGIC %md
# MAGIC We use OpenCV to calculate statistics on our color images, parallelized over our spark cluster and the bytes column in our table. We measure per-channel mean, standard deviation, contrast, and Laplacian variance (sharpness) per class to help understand dataset quality, balance, and feature discriminability before augmentation.
# MAGIC
# MAGIC How to Interpret the Results:
# MAGIC
# MAGIC Per-channel Mean (mean_r, mean_g, mean_b):
# MAGIC Reveals the average color intensity in each channel. Significant differences between classes might mean that color features are predictive (e.g., melted components appear darker or redder than normal ones).
# MAGIC
# MAGIC Per-channel Standard Deviation (std_r, std_g, std_b):
# MAGIC Shows how much color fluctuates within each image/class. Low stddev can indicate low diversity (e.g., backgrounds) or potentially low-contrast images. High stddev might reveal class-distinguishing texture or color variation.
# MAGIC
# MAGIC Contrast (pixel intensity standard deviation):
# MAGIC Higher values mean more dynamic range and maybe better visibility of features. Low values may identify poor-quality, washed-out, or out-of-focus images, which can hinder learning.
# MAGIC
# MAGIC Laplacian Variance (laplacian_var):
# MAGIC Measures image sharpness. Low values could flag blurry images; these samples might need cleaning or upsampling, while high values often indicate relevant details for defect detection.
# MAGIC
# MAGIC

# COMMAND ----------

import cv2
import numpy as np
from pyspark.sql.functions import udf
from pyspark.sql.types import StructType, StructField, FloatType

# Define UDF for OpenCV-based statistics
def opencv_img_stats(img_bytes):
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    # Per-channel mean/std
    means, stddevs = cv2.meanStdDev(img)
    means = means.flatten()
    stddevs = stddevs.flatten()
    # Grayscale for sharpness/focus metrics
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var()) # Focus/sharpness measure (Laplacian)
    contrast = float(np.std(gray))  # Pixel intensity standard deviation (contrast)
    return float(means[0]), float(means[1]), float(means[2]), float(stddevs[0]), float(stddevs[1]), float(stddevs[2]), contrast, sharpness

schema = StructType([
    StructField("mean_b", FloatType()),
    StructField("mean_g", FloatType()),
    StructField("mean_r", FloatType()),
    StructField("std_b", FloatType()),
    StructField("std_g", FloatType()),
    StructField("std_r", FloatType()),
    StructField("contrast", FloatType()),
    StructField("sharpness", FloatType()),
])

opencv_stats_udf = udf(opencv_img_stats, schema)

df = df.withColumn("stats", opencv_stats_udf(df["content"]))

# Unpack stats columns
for name in schema.names:
    df = df.withColumn(name, df["stats." + name])

# Aggregate class-level statistics
stat_cols = schema.names
agg_exprs = [F.mean(c).alias(f"{c}") for c in stat_cols]
display(df.groupBy("label").agg(*agg_exprs))

# COMMAND ----------

# MAGIC %md
# MAGIC Our statistical analysis shows in color means and contrasts between "normal" and "damaged" images, but a notable drop in mean Laplacian variance (sharpness) for damaged samples.
# MAGIC
# MAGIC Normal and damaged color and contrast are similar, which is good and suggests that the model shouldn't overfit on color or basic intensity features (this would be undesireable).
# MAGIC
# MAGIC The sharpness is noticeably lower in damaged images (323) compared to normal (411). This could reflect a propensity for damaged samples to be blurrier, possibly due to defocused imaging, motion blur, or inherent difficulty in capturing damaged features.
# MAGIC
# MAGIC This leads us to the following actions:
# MAGIC - Flag extreme blurry images, especially among the damaged class, to prevent the model from learning on low-quality, ambiguous samples, improving generalization.
# MAGIC - Use augmentation to synthetically increase sharpness in damaged examples, such as unsharp masking filters, or counter case with blur augmentations for normal images to ensure that your model doesn’t learn to simply equate sharpness with “normal.” This prevents spurious cues from dominating learning.

# COMMAND ----------

# MAGIC %md
# MAGIC We can also use these features to visualize potential image clustering. We use the features extracted above to identify. It shows that there is some seperability of the images solely based on the color and contrast. This might be okay, but might also cause overfitting.

# COMMAND ----------

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt

# Select sample and extract features for visualization
sampled = df.select(['label', 'mean_b', 'mean_g', 'mean_r', 'std_b', 'std_g', 'std_r', 'contrast', 'sharpness']).toPandas()
X = sampled.drop('label', axis=1).values
y = sampled['label'].values

# Map labels to colors: damaged=red, normal=blue
label_to_color = {'damaged': 'red', 'normal': 'blue'}
colors = [label_to_color[label] for label in y]

# PCA for 2D projection
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
for label, color in label_to_color.items():
    idx = (y == label)
    plt.scatter(X_pca[idx,0], X_pca[idx,1], c=color, label=label, alpha=0.6)
plt.title("PCA on image features")
plt.legend()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC We can plot the importance of each image feature in the first two principal components, and it shows how sharpness dominates the variance of the dataset, accounting for 99% of the whole dataset variance.

# COMMAND ----------

import numpy as np

explained_var = pca.explained_variance_ratio_
print("Variance explained by each component:", explained_var)

feature_names = sampled.drop('label', axis=1).columns
loadings = pd.DataFrame(
    pca.components_.T,
    columns=['PC1', 'PC2'],
    index=feature_names
)

plt.figure(figsize=(10,5))
loadings[['PC1', 'PC2']].plot(kind='bar')
plt.title('PCA Feature Loadings')
plt.ylabel('Weight')
plt.xticks(rotation=45)
plt.axhline(0, color='grey', linewidth=0.8)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Image Augmentation
# MAGIC
# MAGIC Now that we've explored some of the flaws in the dataset, let's mitigate some of them using some common `OpenCV` techniques

# COMMAND ----------

# MAGIC %md
# MAGIC ### Filter Extreme Blurry Images & Increase Sharpness
# MAGIC Add a column with Laplacian variance (sharpness metric) and filter/flag blurry samples. In this case, there is only one image with a sharpness below 150, which we remove as an outlier.
# MAGIC
# MAGIC We then augment Damaged Images with Unsharp Masking to synthetically increase image sharpness) using OpenCV as a Spark UDF.

# COMMAND ----------

@F.pandas_udf("binary")
def unsharp_mask_udf(content_series):
    def unsharp_mask(content):
        arr = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        # Gaussian blur
        blurred = cv2.GaussianBlur(img, (0, 0), 2)
        # Unsharp mask: original + weighted difference
        sharpened = cv2.addWeighted(img, 1.15, blurred, -0.15, 0)
        _, buf = cv2.imencode('.jpg', sharpened)
        return buf.tobytes()
    return content_series.apply(unsharp_mask)

df = spark.table(f"{CATALOG}.{SCHEMA}.images_silv")

df_damaged_sharp = (
    df
    .filter("label == 'damaged'")
    .withColumn("content", unsharp_mask_udf(F.col("content")))
)

df = df.filter("label != 'damaged'").union(df_damaged_sharp)

# COMMAND ----------

# MAGIC %md
# MAGIC Now let's check our stats again

# COMMAND ----------

df = df.withColumn("stats", opencv_stats_udf(df["content"]))

# Unpack stats columns
for name in schema.names:
    df = df.withColumn(name, df["stats." + name])

df = df.filter(
    F.col("sharpness") > SHARPNESS_THRESHOLD
)

# Aggregate class-level statistics
stat_cols = schema.names
agg_exprs = [F.mean(c).alias(f"{c}") for c in stat_cols]
display(df.groupBy("label").agg(*agg_exprs))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC Note that we're working with tables. We can do this transformation in python or SQL.
# MAGIC
# MAGIC Some transformation on image can be expensive. We can leverage Spark to distribute some image pre-processing first.
# MAGIC
# MAGIC In this example, we will do the following:
# MAGIC - crop the image in the center to make them square (the model we use for fine-tuning take square images)
# MAGIC - resize our image to smaller a resolution (256x256) as our models won't use images with high resolution. 
# MAGIC
# MAGIC We will also augment our dataset to add more "damaged" items as we have here something fairly imbalanced (only 1 on 10 item has an anomaly). <br/>
# MAGIC It looks like our system takes pcb pictures upside/down without preference and that's how our inferences will be. Let's then flip all the damaged images horizontally and add them back in our dataset.

# COMMAND ----------

import cv2
import numpy as np

@F.pandas_udf("binary")
def opencv_resize_udf(content_series):
    def resize_img(content):
        arr = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        new_size = min(h, w)
        # Center crop
        top = (h - new_size) // 2
        left = (w - new_size) // 2
        img_cropped = img[top:top+new_size, left:left+new_size]
        img_resized = cv2.resize(img_cropped, (IMAGE_RESIZE, IMAGE_RESIZE), interpolation=cv2.INTER_NEAREST)
        _, buf = cv2.imencode('.jpg', img_resized)
        return buf.tobytes()
    return content_series.apply(resize_img)

@pandas_udf("binary")
def opencv_flip_udf(content_series):
    def flip_img(content):
        arr = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        flipped = cv2.flip(img, 0)  # 0 = vertical, 1 = horizontal
        _, buf = cv2.imencode('.jpg', flipped)
        return buf.tobytes()
    return content_series.apply(flip_img)

image_meta = {"spark.contentAnnotation": '{"mimeType": "image/jpeg"}'}

# Resize and crop all images (overwrite table)
df = (df.withColumn("sort", F.rand()).orderBy("sort").drop('sort')
    .withColumn("content", opencv_resize_udf(F.col("content")).alias("content", metadata=image_meta)))

# Flip only damaged images (append to table)
added_images = (df
    .filter("label == 'damaged'")
    .withColumn("content", opencv_flip_udf(F.col("content")).alias("content", metadata=image_meta))
)

df = df.union(added_images)

# COMMAND ----------

# MAGIC %md
# MAGIC We can now write out our gold dataset and move to training

# COMMAND ----------

df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.images_gold")

# COMMAND ----------

df.drop('stats').write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.images_gold")
