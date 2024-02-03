import os, sys, pathlib, biobricks
import pyspark.sql, pyspark.sql.functions as F, pyspark.ml.feature

sys.path.insert(0, os.getcwd())
import cvae.tokenizer.selfies_tokenizer, cvae.utils, cvae.spark_helpers as H

spark = pyspark.sql.SparkSession.builder \
    .appName("ChemharmonyDataProcessing") \
    .config("spark.driver.memory", "32g") \
    .config("spark.driver.maxResultSize", "20g") \
    .getOrCreate()

pathlib.Path('data/processed').mkdir(parents=True, exist_ok=True)

# GET CHEMHARMONY SUBSTANCES ===============================================================
chemharmony = biobricks.assets('chemharmony')
rawsubstances = spark.read.parquet(chemharmony.substances_parquet).select("sid","source","data")

## Extract INCHI and SMILES from the data json column. It has a few different names for the same thing
substances = rawsubstances \
    .withColumn("rawinchi", F.get_json_object("data", "$.inchi")) \
    .withColumn("ligand_inchi", F.get_json_object("data", "$.Ligand InChI")) \
    .withColumn("rawsmiles", F.get_json_object("data", "$.SMILES")) \
    .withColumn("ligand_smiles", F.get_json_object("data", "$.Ligand SMILES")) \
    .withColumn("inchi", F.coalesce("rawinchi", "ligand_inchi")) \
    .withColumn("smiles", F.coalesce("rawsmiles", "ligand_smiles")) \
    .select("sid","source","inchi","smiles")
    
substances = substances.withColumn("smiles", F.coalesce("smiles", H.inchi_to_smiles_udf("inchi")))
substances = substances.filter(substances.smiles.isNotNull())
substances = substances.withColumn("selfies", H.smiles_to_selfies_udf("smiles"))
substances = substances.select('sid', 'inchi', 'smiles', 'selfies').distinct()
substances = substances.filter(substances.selfies.isNotNull())

# Transform selfies to indices
tokenizer = cvae.tokenizer.selfies_tokenizer.SelfiesTokenizer().fit(substances, 'selfies')
substances = tokenizer.transform(substances, 'selfies', 'encoded_selfies')

tokenizer.save('data/processed/selfies_tokenizer.json')
substances.write.parquet('data/processed/substances.parquet', mode='overwrite')

spark.read.parquet('data/processed/substances.parquet').count() # 118 197 880
spark.read.parquet('data/processed/substances.parquet').show(100)

# GET CHEMARHMONY ACTIVITIES ===========================================================================
chemharmony = biobricks.assets('chemharmony')
activities = spark.read.parquet(chemharmony.activities_parquet).select(['smiles','pid','sid','binary_value'])
substances = spark.read.parquet('data/processed/substances.parquet')
data = activities.join(substances.select('sid','selfies','encoded_selfies'), 'sid', 'inner')
data = data.withColumnRenamed('pid', 'assay').withColumnRenamed('binary_value', 'value')
data = data.orderBy(F.rand(52)) # Randomly shuffle data with seed 52

## REMOVE ROWS WITH ASSAYS THAT HAVE LESS THAN 100 `0` and `1` values =======
zero_counts = data.filter(F.col('value') == 0).groupBy('assay').count().withColumnRenamed('count', 'zero_count')
valid_assays = zero_counts.filter(F.col('zero_count') >= 100).select('assay')
data = data.join(valid_assays, 'assay', 'inner')

one_counts = data.filter(F.col('value') == 1).groupBy('assay').count().withColumnRenamed('count', 'one_count')
valid_assays = one_counts.filter(F.col('one_count') >= 100).select('assay')
data = data.join(valid_assays, 'assay', 'inner')

## REMOVE ASSAYS WITH A CLASS IMBALANCE GREATER THAN 10% =====================
assay_counts = data.groupBy('assay', 'value').count()
pivot_counts_1 = assay_counts.groupBy('assay').pivot('value').agg(F.count('count')).na.fill(0)
pivot_counts_2 = pivot_counts_1.withColumnRenamed("0.0", "count_0").withColumnRenamed("1.0", "count_1")
assay_counts = pivot_counts_2.withColumn('imbalance', F.abs((F.col('count_0') - F.col('count_1')) / (F.col('count_0') + F.col('count_1'))))
balanced_assays = assay_counts.filter((F.col('imbalance') <= 0.1) | (F.col('imbalance') >= 0.9)).select('assay')
data = data.join(balanced_assays, 'assay', 'inner')

## Map assay UUIDs to integers
indexer = pyspark.ml.feature.StringIndexer(inputCol="assay", outputCol="assay_index")
data = indexer.fit(data).transform(data)

## Convert SMILES to Morgan fingerprints
data = data.withColumn("morgan_fingerprint", H.smiles_to_morgan_udf(F.col("smiles")))

## write out the processed data, delete activities.parquet if it exists
cvae.utils.mk_empty_directory('data/processed/activities.parquet', overwrite=True)
data.write.parquet('data/processed/activities.parquet', mode='overwrite')

spark.read.parquet('data/processed/activities.parquet').count() # 14 613 322