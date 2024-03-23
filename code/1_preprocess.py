# purpose: preprocess chemharmony data to create SELFIES, a tokenizer, and a Parquet file for conversion to a TensorDataset.
# dependencies: chemharmony
# outputs: 
#     - data/processed/substances.parquet - Parquet file containing processed substance data with SELFIES, encoded SELFIES, InChI, and SMILES.
#     - data/processed/selfies_tokenizer.json - JSON file containing the trained SELFIES tokenizer for encoding and decoding SELFIES strings.
#     - data/processed/activities.parquet - Parquet file containing processed activity data with assay IDs, substance IDs, binary values, and Morgan fingerprints, ready for conversion to a TensorDataset.
import biobricks
import pyspark.sql, pyspark.sql.functions as F, pyspark.ml.feature
import cvae.tokenizer.selfies_tokenizer, cvae.utils, cvae.spark_helpers as H

outdir = cvae.utils.mk_empty_directory('data/processed', overwrite=True)
spark = pyspark.sql.SparkSession.builder \
    .appName("ChemharmonyDataProcessing") \
    .config("spark.driver.memory", "64g") \
    .config("spark.driver.maxResultSize", "48g") \
    .config("spark.executor.memory", "64g") \
    .getOrCreate()

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

spark.read.parquet('data/processed/substances.parquet').count() # 115 153 470
spark.read.parquet('data/processed/substances.parquet').show(100)

# GET CHEMARHMONY ACTIVITIES ===========================================================================
chemharmony = biobricks.assets('chemharmony')
activities = spark.read.parquet(chemharmony.activities_parquet).select(['smiles','pid','sid','binary_value'])
substances = spark.read.parquet('data/processed/substances.parquet')
data = activities.join(substances.select('sid','selfies','encoded_selfies'), 'sid', 'inner')
data = data.withColumnRenamed('pid', 'assay').withColumnRenamed('binary_value', 'value')
data = data.orderBy(F.rand(52)) # Randomly shuffle data with seed 52

## REMOVE ASSAYS WITH < 100 `0` or `1` VALUES OR A CLASS IMBALANCE GREATER THAN 10% =====================
assay_counts = data.groupBy('assay', 'value').count()
pivot_cnts_1 = assay_counts.groupBy('assay').pivot('value').agg(F.first('count')).na.fill(0)
pivot_cnts_2 = pivot_cnts_1.withColumnRenamed("0", "count_0").withColumnRenamed("1","count_1")
pivot_filter = pivot_cnts_2.filter((F.col('count_0') >= 100) & (F.col('count_1') >= 100))

## imbalance filter
assay_counts = pivot_filter.withColumn('imbalance', F.abs((F.col('count_0') - F.col('count_1')) / (F.col('count_0') + F.col('count_1'))))
balanced_assays = assay_counts.filter((F.col('imbalance') <= 0.9)).select('assay')

data = data.join(balanced_assays, 'assay', 'inner') # 5 898 498 

## Map assay UUIDs to integers
indexer = pyspark.ml.feature.StringIndexer(inputCol="assay", outputCol="assay_index")
data = indexer.fit(data).transform(data)

## write out the processed data, delete activities.parquet if it exists
cvae.utils.mk_empty_directory('data/processed/activities.parquet', overwrite=True)
data.write.parquet('data/processed/activities.parquet', mode='overwrite')