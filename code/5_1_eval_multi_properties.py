import itertools, uuid, pathlib
import pandas as pd, tqdm, sklearn.metrics, torch, numpy as np, os
import cvae.tokenizer, cvae.models.multitask_transformer as mt, cvae.utils, cvae.models.mixture_experts as me
from cvae.tokenizer import SelfiesPropertyValTokenizer
from pyspark.sql.functions import col, when, countDistinct
from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from pyspark.sql.functions import split, col, when
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, log_loss

# Create all necessary directories
outdir = pathlib.Path("cache/eval_multi_properties")
outdir.mkdir(exist_ok=True, parents=True)

tqdm.tqdm.pandas()

tokenizer : SelfiesPropertyValTokenizer = me.MoE.load("brick/moe").tokenizer
spark = cvae.utils.get_spark_session()

# GENERATE STRATIFIED EVALUATIONS FOR POSITION 0-5 ===============================
outdf = spark.read.parquet("cache/generate_evaluations/multitask_predictions.parquet")

# Calculate metrics
value_indexes = list(tokenizer.value_indexes().values())
val0_index, val1_index = value_indexes[0], value_indexes[1]

def calculate_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_true_binary = (y_true != val0_index).astype(int)
    y_pred = np.array(y_pred)
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    auc = float(roc_auc_score(y_true_binary, y_pred))
    acc = float(accuracy_score(y_true_binary, y_pred_binary))
    bac = float(balanced_accuracy_score(y_true_binary, y_pred_binary))
    ce_loss = float(log_loss(y_true_binary, y_pred))
    return auc, acc, bac, ce_loss


calculate_metrics_udf = F.udf(calculate_metrics, "struct<AUC:double, ACC:double, BAC:double, cross_entropy_loss:double>")
large_properties_df = outdf.groupBy('nprops', 'assay').agg(
    F.collect_list('value').alias('y_true'),
    F.collect_list('probs').alias('y_pred'),
    countDistinct('chemical_id').alias('nchem'),
    F.sum(when(col('value') == val1_index, 1).otherwise(0)).alias('NUM_POS'),
    F.sum(when(col('value') == val0_index, 1).otherwise(0)).alias('NUM_NEG')) \
    .filter((col('NUM_POS') >= 10) & (col('NUM_NEG') >= 10) & (col('nchem') >= 20)).cache()

metrics_df = large_properties_df.repartition(800) \
    .withColumn('metrics', calculate_metrics_udf(F.col('y_true'), F.col('y_pred'))) \
    .select('nprops', 'assay', col('metrics.AUC').alias('AUC'), col('metrics.ACC').alias('ACC'), col('metrics.BAC').alias('BAC'), col('metrics.cross_entropy_loss').alias('cross_entropy_loss'), 'NUM_POS', 'NUM_NEG')

df = metrics_df.toPandas()
df['AUC'].median()

metrics_df.write.parquet((outdir / "multitask_metrics.parquet").as_posix(), mode="overwrite")