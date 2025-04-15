import uuid, torch, torch.nn.utils.rnn
import pyspark.sql, pyspark.sql.functions as F
import cvae.utils, cvae.tokenizer.selfies_tokenizer, cvae.tokenizer.selfies_property_val_tokenizer
import logging

cvae.utils.setup_logging('log/build_tensordataset.log', logging)
spark = cvae.utils.get_spark_session()

# BUILD UNSUPERVISED TRAINING SET ====================================================
data = spark.read.parquet('cache/preprocess_tokenizer/substances.parquet')
def process_selfies(partition,outdir):
    tensor = torch.tensor([row['encoded_selfies'] for row in partition], dtype=torch.long)
    torch.save(tensor, outdir / f'partition_{uuid.uuid4()}.pt')

vaedir = cvae.utils.mk_empty_directory('cache/build_tensordataset/all_selfies', overwrite=True)
selfies = data.select('encoded_selfies') # selfies are already distinct
selfies.rdd.foreachPartition(lambda partition: process_selfies(partition, vaedir))
logging.info('Unsupervised training set built')

# BUILD MULTITASK SUPERVISED DATA SET =============================================
def build_multitask_supervised_data(spark):
    data = spark.read.parquet("cache/preprocess_activities/activities.parquet").select('selfies','assay_index','encoded_selfies', 'value')
    data = data \
        .groupby('encoded_selfies', 'assay_index') \
        .agg(F.collect_set('value').alias('values')) \
        .filter(F.size('values') == 1) \
        .select('encoded_selfies', 'assay_index', F.element_at('values', 1).alias('value')) \
        .cache()
        
    selfies_tok = cvae.tokenizer.selfies_tokenizer.SelfiesTokenizer().load('cache/preprocess_tokenizer/selfies_tokenizer.json')
    num_assays = int(data.agg(F.max('assay_index')).collect()[0][0] + 1)
    num_values = int(data.agg(F.max('value')).collect()[0][0] + 1) # TODO this assumes an index identity for values
    tokenizer = cvae.tokenizer.SelfiesPropertyValTokenizer(selfies_tok, num_assays, num_values)

    tokenizer.save(cvae.utils.mk_empty_directory('brick/selfies_property_val_tokenizer', overwrite=True))

    def create_tensors(partition, outdir):
        partition = list(partition)
        selfies = torch.stack([torch.LongTensor(r.encoded_selfies) for r in partition])
        assay_vals = [tokenizer.tokenize_assay_values(r.assay_val_pairs) for r in partition]
        assay_vals = torch.nn.utils.rnn.pad_sequence(assay_vals, batch_first=True, padding_value=tokenizer.pad_idx)
        torch.save({'selfies': selfies, 'assay_vals': assay_vals}, (outdir / f"{uuid.uuid4()}.pt").as_posix())

    gdata = data \
        .select("encoded_selfies", "assay_index", "value").distinct() \
        .groupby("encoded_selfies") \
        .agg(F.collect_list(F.struct("assay_index", "value")).alias("assay_val_pairs")) \
        .cache()
        
    trn, tst, hld = gdata.randomSplit([0.8, 0.1, 0.1], seed=37)

    # df, path = trn, 'trn'
    for df, path in zip([trn, tst, hld], ['trn', 'tst', 'hld']):
        logging.info(f'Building {path} multitask supervised training set')
        outdir = cvae.utils.mk_empty_directory(f'cache/build_tensordataset/multitask_tensors/{path}', overwrite=True)
        df.foreachPartition(lambda partition: create_tensors(partition, outdir))
        
build_multitask_supervised_data(spark)
logging.info('Multitask supervised training set built')