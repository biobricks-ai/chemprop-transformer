# cmd: 
# spark-submit \
#   --master local[240] \
#   --driver-memory 512g \
#   --conf spark.eventLog.enabled=true \
#   --conf spark.eventLog.dir=file:///tmp/spark-events \
#   code/1_3_preprocess_activities.py

import biobricks
import cvae
import logging
import pyspark
import pyspark.ml.feature
import pyspark.sql.functions as F
import pathlib
import cvae.utils
import pandas as pd

# set up logging
logdir = pathlib.Path('cache/preprocess_activities/log')
logdir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=logdir / 'preprocess_activities.log', filemode='w')

outdir = pathlib.Path('cache/preprocess_activities')
outdir.mkdir(parents=True, exist_ok=True)

# set up spark session
spark = cvae.utils.get_spark_session()

# GET CHEMARHMONY ACTIVITIES ===========================================================================
logging.info("Loading ChemHarmony activities data...")
chemharmony = biobricks.assets('chemharmony')
activities = spark.read.parquet(chemharmony.activities_parquet).select(['source','smiles','pid','sid','binary_value'])
substances = spark.read.parquet('cache/preprocess_tokenizer/substances.parquet')
data = activities.join(substances.select('sid','selfies','encoded_selfies'), 'sid', 'inner')
data = data.withColumnRenamed('pid', 'assay').withColumnRenamed('binary_value', 'value')
data = data.orderBy(F.rand(52)) # Randomly shuffle data with seed 52

source_counts = data.groupBy('source').count().orderBy('count', ascending=False).toPandas()
logging.info(str(source_counts))
assert min(source_counts['count']) >= 1000

## REMOVE ASSAYS WITH < 20 `0` or `1` VALUES OR A CLASS IMBALANCE GREATER THAN 10% =====================
logging.info("Filtering assays based on class counts and balance...")
assay_counts = data.groupBy('assay', 'value').count()
pivot_cnts_1 = assay_counts.groupBy('assay').pivot('value').agg(F.first('count')).na.fill(0)
pivot_cnts_2 = pivot_cnts_1.withColumnRenamed("0", "count_0").withColumnRenamed("1","count_1")
pivot_filter = pivot_cnts_2.filter((F.col('count_0') >= 20) & (F.col('count_1') >= 20))

## imbalance filter
assay_counts = pivot_filter.withColumn('imbalance', F.abs((F.col('count_0') - F.col('count_1')) / (F.col('count_0') + F.col('count_1'))))
balanced_assays = assay_counts.filter((F.col('imbalance') <= 0.95)).select('assay')
logging.info(f"Number of balanced assays: {balanced_assays.count()}")

data = data.join(balanced_assays, 'assay', 'inner') # 5 898 498 
logging.info(f"Data size after filtering: {data.count()} rows")

## Map assay UUIDs to integers
logging.info("Converting assay IDs to indices...")
indexer = pyspark.ml.feature.StringIndexer(inputCol="assay", outputCol="assay_index")
data = indexer.fit(data).transform(data)

## write out the processed data, delete activities.parquet if it exists
logging.info("Writing processed data to parquet...")
data.write.parquet((outdir / 'activities.parquet').as_posix(), mode='overwrite')
logging.info(f"wrote {outdir / 'activities.parquet'}")

## TEST count activities ===========================================================
data = spark.read.parquet((outdir / 'activities.parquet').as_posix()).cache()
assert data.count() > 10e6 # should have more than 10m activities

source_counts = data.groupBy('source').count().orderBy('count', ascending=False).toPandas()
logging.info(str(source_counts))
assert min(source_counts['count']) >= 1000