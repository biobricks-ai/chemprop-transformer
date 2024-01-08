import os, sys, uuid, pathlib
import torch, torch.nn.utils.rnn
import pyspark.sql, pyspark.sql.functions as F
sys.path.insert(0, os.getcwd())
import cvae.utils, cvae.tokenizer.selfies_tokenizer, cvae.tokenizer.selfies_property_val_tokenizer

spark = pyspark.sql.SparkSession.builder \
    .appName("ChemharmonyDataProcessing") \
    .config("spark.driver.memory", "64g") \
    .config("spark.local.dir", "/mnt/ssd/spark/local") \
    .getOrCreate()

# BUILD UNSUPERVISED TRAINING SET ====================================================
data = spark.read.parquet("data/processed/substances.parquet")
def process_selfies(partition,outdir):
    tensor = torch.tensor([row['encoded_selfies'] for row in partition], dtype=torch.long)
    torch.save(tensor, outdir / f'partition_{uuid.uuid4()}.pt')

vaedir = cvae.utils.mk_empty_directory('data/processed/all_selfies', overwrite=True)
selfies = data.select('encoded_selfies') # selfies are already distinct
selfies.rdd.foreachPartition(lambda partition: process_selfies(partition, vaedir))

# BUILD SUPERVISED DATA SET =============================================
data = spark.read.parquet("data/processed/activities.parquet")
train, test, holdout = data.randomSplit([0.8, 0.1, 0.1], seed=37) # TODO stratify by assay and value

def process_partition(partition, outdir):
    
    get_vals = lambda field: [row[field] for row in partition]

    partition = list(partition)
    selfie = torch.tensor(get_vals('encoded_selfies'), dtype=torch.long)
    assays = torch.tensor([int(x) for x in get_vals('assay_index')], dtype=torch.long)
    morgan = torch.tensor(get_vals('morgan_fingerprint'), dtype=torch.float)
    values = torch.tensor(get_vals('value'), dtype=torch.float)    
    
    outobj = {'tselfies': selfie, 'tmorgan': morgan, 'tass': assays, 'tval': values}
    torch.save(outobj, outdir / f'partition_{uuid.uuid4()}.pt')

outdir = cvae.utils.mk_empty_directory('data/processed/activity_tensors', overwrite=True)

def build_partition_tensors(df, subdir):
    cvae.utils.mk_empty_directory(subdir, overwrite=True)
    df = df.repartition(df.count() // 10000) # 10k rows per partition
    df.foreachPartition(lambda partition: process_partition(partition, subdir))

build_partition_tensors(train, outdir / 'train.pt')
build_partition_tensors(test, outdir / 'test.pt')
build_partition_tensors(holdout, outdir / 'holdout.pt')

# BUILD MULTITASKSUPERVISED DATA SET =============================================
def build_multitask_supervised_data(spark):
    data = spark.read.parquet("data/processed/activities.parquet").select('selfies','assay_index','encoded_selfies', 'value')

    # exclude any examples in data where a given selfies has distinct values for a given assay
    data = data \
        .groupby('encoded_selfies', 'assay_index') \
        .agg(F.collect_set('value').alias('values')) \
        .filter(F.size('values') == 1) \
        .select('encoded_selfies', 'assay_index', F.element_at('values', 1).alias('value'))
        
        
    selfies_tok = cvae.tokenizer.selfies_tokenizer.SelfiesTokenizer().load('data/processed/selfies_tokenizer.json')
    num_assays = int(data.agg(F.max('assay_index')).collect()[0][0] + 1)
    num_values = int(data.agg(F.max('value')).collect()[0][0] + 1) # TODO this assumes an index identity for values
    tokenizer = cvae.tokenizer.SelfiesPropertyValTokenizer(selfies_tok, num_assays, num_values)

    savepath = cvae.utils.mk_empty_directory('data/processed/selfies_property_val_tokenizer', overwrite=True)
    tokenizer.save(savepath)

    def create_tensors(partition, outdir):
        partition = list(partition)
        selfies = torch.stack([torch.LongTensor(r.encoded_selfies) for r in partition])
        assay_vals = [tokenizer.tokenize_assay_values(r.assay_val_pairs) for r in partition]
        assay_vals = torch.nn.utils.rnn.pad_sequence(assay_vals, batch_first=True, padding_value=tokenizer.pad_idx)
        torch.save({'selfies': selfies, 'assay_vals': assay_vals}, outdir / f"{uuid.uuid4()}.pt")

    gdata = data \
        .select("encoded_selfies", "assay_index", "value").distinct() \
        .groupby("encoded_selfies").agg(F.collect_list(F.struct("assay_index", "value")).alias("assay_val_pairs"))
        
    trn, tst, hld = gdata.randomSplit([0.8, 0.1, 0.1], seed=37)

    # df, path = trn, 'trn'
    for df, path in zip([trn, tst, hld], ['trn', 'tst', 'hld']):
        cvae.utils.mk_empty_directory(f'data/processed/multitask_tensors/{path}', overwrite=True)
        df.foreachPartition(lambda partition: create_tensors(partition, pathlib.Path(f'data/processed/multitask_tensors/{path}')))
        
build_multitask_supervised_data(spark)