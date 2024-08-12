import torch
import pdb
import cvae.models.mixture_experts as moe
import cvae.spark_helpers as H
import cvae.tokenizer
import logging
import pandas as pd, biobricks as bb

from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, DataLoader

tqdm.pandas()# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s %(levelname)s:%(message)s',
                    handlers=[logging.StreamHandler()])

DEVICE = torch.device(f'cuda:0')

# Load the model ==============================================================================
model = moe.MoE.load("brick/moe").to(DEVICE)
model = torch.nn.DataParallel(model)
tokenizer = cvae.tokenizer.SelfiesPropertyValTokenizer.load('brick/selfies_property_val_tokenizer')
sftok = tokenizer.selfies_tokenizer
logging.info("Model loaded and moved to device.")

# Preprocess the data ==========================================================================
df = pd.read_parquet(bb.assets('bayer-dili').bayer_dili_data_parquet)
label_encoder = LabelEncoder()
# TODO we need to assign a new assay_index, not reuse old ones
df['assay_index'] = label_encoder.fit_transform(df['variable'])
df['value'] = df['value'].apply(lambda x: 0 if x == 'negative' else 1)
df['selfies'] = df['canonical_smiles'].progress_apply(lambda x: H.smiles_to_selfies_safe(x))
df.shape
df = df.dropna(subset=['selfies', 'value', 'assay_index'])

finetune_dir = cvae.utils.mk_empty_directory('data/finetune', overwrite=True)
df.to_parquet(f'{finetune_dir}/finetune_activities.parquet', index=False)
logging.info("finetune_activities.parquet written to data/finetune directory.")

# Build the TensorDataset ======================================================================
import uuid, torch, torch.nn.utils.rnn
import pyspark.sql, pyspark.sql.functions as F
import cvae.utils, cvae.tokenizer.selfies_tokenizer, cvae.tokenizer.selfies_property_val_tokenizer

spark = pyspark.sql.SparkSession.builder \
    .appName("ChemharmonyDataProcessing") \
    .config("spark.driver.memory", "64g") \
    .getOrCreate()

data = spark.read.parquet("data/finetune/finetune_activities.parquet")\
    .select('selfies','assay_index', 'value')
data = tokenizer.selfies_tokenizer.transform(data, 'selfies', 'encoded_selfies')

# exclude any examples in data where a given selfies has distinct values for a given assay
data = data \
    .groupby('encoded_selfies', 'assay_index') \
    .agg(F.collect_set('value').alias('values')) \
    .filter(F.size('values') == 1) \
    .select('encoded_selfies', 'assay_index', F.element_at('values', 1).alias('value'))

data = data.cache()

num_assays = int(data.agg(F.max('assay_index')).collect()[0][0] + 1)
# TODO this assumes an index identity for values
num_values = int(data.agg(F.max('value')).collect()[0][0] + 1) 

def create_tensors(partition, outdir):
    partition_list = list(partition)
    selfies = torch.stack([torch.LongTensor(r.encoded_selfies) for r in partition_list])
    assay_vals = [tokenizer.tokenize_assay_values(r.assay_val_pairs) for r in partition_list]
    assay_vals = torch.nn.utils.rnn.pad_sequence(assay_vals, batch_first=True, padding_value=tokenizer.pad_idx)
    torch.save({'selfies': selfies, 'assay_vals': assay_vals}, (outdir / f"{uuid.uuid4()}.pt").as_posix())

gdata = data \
    .select("encoded_selfies", "assay_index", "value").distinct() \
    .groupby("encoded_selfies").agg(F.collect_list(F.struct("assay_index", "value")).alias("assay_val_pairs"))
    
trn, tst, hld = gdata.randomSplit([0.8, 0.1, 0.1], seed=37)

for df, path in zip([trn, tst, hld], ['trn', 'tst', 'hld']):
    outdir = cvae.utils.mk_empty_directory(f'data/finetune/multitask_tensors/{path}', overwrite=True)
    partition = df.rdd.glom().filter(lambda x: len(x) > 0).first()
    df.foreachPartition(lambda partition: create_tensors(partition, outdir))
        
