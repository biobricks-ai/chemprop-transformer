from pyspark.sql.functions import udf
from pyspark.sql.types import BooleanType, ArrayType, IntegerType, FloatType, StringType
import pyspark.sql.functions as F
import selfies
from rdkit import Chem
import cvae.models.tokenizer as tokenizer  # Assuming this is your tokenizer module

def is_valid_smiles(smiles):
    """Validate SMILES string using RDKit."""
    # Disable RDKit warnings
    from rdkit import RDLogger
    logger = RDLogger.logger()
    logger.setLevel(RDLogger.CRITICAL)
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None and mol.GetNumAtoms() > 0
    except:
        return False

# Register the UDF
is_valid_smiles_udf = udf(is_valid_smiles, BooleanType())


def smiles_to_morgan(smiles, nBits=2048):
    # Assuming tokenizer.smiles_to_morgan_fingerprint returns a numpy array of floats
    morgan_fingerprint = tokenizer.smiles_to_morgan_fingerprint(smiles, nBits=nBits)
    return morgan_fingerprint.tolist() if morgan_fingerprint is not None else []

# Register the UDF
smiles_to_morgan_udf = udf(smiles_to_morgan, ArrayType(IntegerType()))


# INCHI TO SMILES UDF ============================================================================
def inchi_to_smiles_safe(inchi):
    try:
        if inchi:
            mol = Chem.MolFromInchi(inchi)
            if mol:
                return Chem.MolToSmiles(mol)
    except Exception as e:
        print(f"Error in converting InChI to SMILES: {e}")
    return None

inchi_to_smiles_udf = udf(inchi_to_smiles_safe, StringType())

# SMILES TO SELFIES UDF ===========================================================================
def smiles_to_selfies_safe(smiles):
    try:
        if smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                return selfies.encoder(smiles)
    except Exception as e:
        print(f"Error in converting SMILES to SELFIES: {e}")
    return None

smiles_to_selfies_udf = udf(smiles_to_selfies_safe, StringType())

# SELFIES SYMBOL SPLITTER UDF ====================================================================
def split_selfies(selfies_string):
    return selfies.split_selfies(selfies_string) if selfies_string else []

split_selfies_udf = udf(split_selfies, ArrayType(StringType()))