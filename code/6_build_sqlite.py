import pyspark.sql, pyspark.sql.functions as F
from pyspark.sql.functions import col
import biobricks as bb, pandas as pd

spark = pyspark.sql.SparkSession.builder \
    .appName("ChemharmonyDataProcessing") \
    .config("spark.driver.memory", "64g") \
    .config("spark.driver.maxResultSize", "100g") \
    .getOrCreate()
    
ch = bb.assets('chemharmony')

pytorch_property_id = spark.read.parquet("data/processed/activities.parquet")\
    .withColumnRenamed('assay','property_id')\
    .withColumnRenamed('assay_index','property_pytorch_index')\
    .withColumn("property_pytorch_index", col("property_pytorch_index").cast("int")) \
    .select("property_id","property_pytorch_index").distinct()

raw_activities = spark.read.parquet(ch.activities_parquet)\
    .withColumnRenamed('pid', 'property_id')\
    .withColumnRenamed('sid', 'substance_id')\
    .withColumnRenamed('aid', 'activity_id')\
    .select('source','activity_id','property_id','substance_id','inchi','smiles','value','binary_value')

prop = spark.read.parquet(ch.properties_parquet)
prop = prop.withColumnRenamed('pid', 'property_id')
prop = prop.join(pytorch_property_id, on='property_id')

raw_prop_cat = spark.read.parquet(ch.property_categories_parquet)
raw_prop_cat = raw_prop_cat.withColumnRenamed('pid', 'property_id')

# CREATE SOME NORMALIZED TABLES PROPERTY, CATEGORY, SOURCE =========================

## categories and property_category
cat = raw_prop_cat.select('category').distinct()
cat = cat.withColumn('category_id', F.monotonically_increasing_id())
prop_cat = raw_prop_cat.join(cat, on='category').select('property_id', 'category_id','reason','strength')

## sources and property_source
src = prop.select('source').distinct()
src = src.withColumn('source_id', F.monotonically_increasing_id())
prop = prop.join(src, on='source').select('property_id','property_pytorch_index','source_id','data')

## activities and activity_source 
activities = raw_activities\
    .select('source','activity_id','property_id','substance_id','inchi','smiles','value','binary_value')\
    .join(src, on='source')\
    .select('source_id','activity_id','property_id','substance_id','inchi','smiles','value','binary_value')


# WRITE TABLE TO SQLITE =============================================================
import sqlite3
conn = sqlite3.connect('data/processed/cvae.sqlite')

prop.toPandas().to_sql('property', conn, if_exists='replace', index=False)

cat.toPandas().to_sql('category', conn, if_exists='replace', index=False)
prop_cat.toPandas().to_sql('property_category', conn, if_exists='replace', index=False)

src.toPandas().to_sql('source', conn, if_exists='replace', index=False)
activities.toPandas().to_sql('activity', conn, if_exists='replace', index=False)

conn.close()

# DO A SIMPLE TEST QUERY =============================================================
# query the graph
import sqlite3
conn = sqlite3.connect('data/processed/cvae.sqlite')

query = """
SELECT * 
FROM property pr 
INNER JOIN property_category pc ON pr.property_id = pc.property_id
INNER JOIN category c ON pc.category_id = c.category_id
WHERE c.category = 'endocrine disruption' 
ORDER BY strength DESC
"""

df = pd.read_sql_query(query, conn)

a = df[['data','category','reason','strength']][df['category'] == 'endocrine disruption']
# iterate over a printing a human readable version of the row
for row in a.iterrows():
    print(f"{row[1]['data']} is an {row[1]['category']} because {row[1]['reason']} with strength {row[1]['strength']}")

assert df['data'].isnull().sum() == 0, "Null values found in 'data' column"
assert df['reason'].isnull().sum() == 0, "Null values found in 'reason' column"

assert pd.api.types.is_string_dtype(df['data']), "'data' column should be of type string"
assert pd.api.types.is_string_dtype(df['reason']), "'reason' column should be of type string"
assert pd.api.types.is_numeric_dtype(df['strength']), "'reason' column should be of type string"

conn.close()