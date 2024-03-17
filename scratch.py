# TODO: look at low AUC properties of bindingdb and toxvaldb and extract to json and look at the properties


import sqlite3
import pandas as pd

# Path to your SQLite database
db_file = 'brick/cvae.sqlite'
metrics_df = pd.read_csv('metrics.csv')
# Connect to the SQLite database
conn = sqlite3.connect(db_file)

# Increase the number of columns shown when printing the DataFrame
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Read the 'activity' table into a pandas DataFrame
query = "SELECT * FROM activity;"
activity_df = pd.read_sql_query(query, conn)
property_df = pd.read_sql_query("SELECT * FROM property inner join source on property.source_id = source.source_id;", conn)
# how many unique properties for each source_id
property_df['source'].value_counts()
merged_df = pd.merge(property_df, metrics_df, left_on='property_token', right_on='assay', how='inner')
imbalanceness = merged_df.apply(
    lambda x: abs(x['NUM_POS'] - x['NUM_NEG']) / (x['NUM_POS'] + x['NUM_NEG']) if (x['NUM_POS'] + x['NUM_NEG']) > 0 else None,
    axis=1
)
merged_df['Imbalanceness'] = imbalanceness
low_auc_df = merged_df[merged_df['AUC'] < 0.6]
high_auc_df = merged_df[merged_df['AUC'] > 0.9]
# get the median, and mean, and variance of imbalanceness for low_auc_df and high_auc_df
low_auc_df['Imbalanceness'].median()
low_auc_df['Imbalanceness'].mean()
low_auc_df['Imbalanceness'].var()
high_auc_df['Imbalanceness'].median()
high_auc_df['Imbalanceness'].mean()
high_auc_df['Imbalanceness'].var()

import json
# count the source_ids
low_auc_df['source'].value_counts()
counts = {condition: merged_df[merged_df['AUC'].between(*range)].groupby('source').size() for condition, range in [('low_auc', (0, 0.6)), ('high_auc', (0.8, 1)), ('total', (0, 1))]}
# Filter for 'bindingdb' source
bindingdb_df = merged_df[merged_df['source'] == 'bindingdb']
low_auc_bindingdb_df = bindingdb_df[bindingdb_df['AUC'] < 0.6]
# Optional: Parse JSON data in 'data' column
bindingdb_df['data_parsed'] = bindingdb_df['data'].apply(lambda x: json.loads(x))

def parse_json(json_string):
    try:
        json_data = json.loads(json_string)
        return json_data
    except json.JSONDecodeError:
        return {}  # Return an empty dictionary in case of JSON parsing error

# Apply the parse_json function to each row in the 'data' column
bindingdb_df_expanded = bindingdb_df['data'].apply(parse_json).apply(pd.Series)
toxvaldb_df = merged_df[merged_df['source'] == 'toxvaldb']
low_auc_toxvaldb_df = toxvaldb_df[toxvaldb_df['AUC'] < 0.6]
# expand the json data
toxvaldb_df_expanded = toxvaldb_df['data'].apply(parse_json).apply(pd.Series)

low_auc_bindingdb_df_expanded = low_auc_bindingdb_df['data'].apply(parse_json).apply(pd.Series)
# Concatenate the expanded data with the original DataFrame
bindingdb_df_combined = pd.concat([bindingdb_df, bindingdb_df_expanded], axis=1)

# Display the DataFrame
print(bindingdb_df_combined.head())
bindingdb_df_expanded['metric'].value_counts()
low_auc_bindingdb_df_expanded['metric'].value_counts()



def parse_json(json_string):
    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        return {}  # Return an empty dictionary in case of a JSON parsing error

def filter_and_expand(df, source, auc_low, auc_high):
    filtered_df = df[(df['source'] == source) & (df['AUC'] >= auc_low) & (df['AUC'] < auc_high)]
    expanded_data = filtered_df['data'].apply(parse_json).apply(pd.Series)
    return pd.concat([filtered_df.reset_index(drop=True), expanded_data.reset_index(drop=True)], axis=1)

# Filter and expand data for BindingDB and ToxvalDB
low_auc_bindingdb_df = filter_and_expand(merged_df, 'bindingdb', 0, 0.6)
high_auc_bindingdb_df = filter_and_expand(merged_df, 'bindingdb', 0.8, 1)
low_auc_toxvaldb_df = filter_and_expand(merged_df, 'toxvaldb', 0, 0.6)
high_auc_toxvaldb_df = filter_and_expand(merged_df, 'toxvaldb', 0.8, 1)

# Example: Use this if you need value counts for a specific metric
# low_auc_bindingdb_df['metric'].value_counts()

# Displaying sample data
print(low_auc_bindingdb_df.head())
print(high_auc_bindingdb_df.head())
print(low_auc_toxvaldb_df.head())
print(high_auc_toxvaldb_df.head())
# save them to csv
low_auc_bindingdb_df.to_csv('low_auc_bindingdb.csv')
high_auc_bindingdb_df.to_csv('high_auc_bindingdb.csv')
low_auc_toxvaldb_df.to_csv('low_auc_toxvaldb.csv')
high_auc_toxvaldb_df.to_csv('high_auc_toxvaldb.csv')


# ===============================================
def parse_json(row):
    try:
        # Convert JSON data into a dictionary
        dic = json.loads(row['data'])
        # Add 'index' as a key
        dic['index'] = row.name
        return dic
    except json.JSONDecodeError:
        # In case of JSON decode error, return None or an empty dict
        return None

merged_df_expanded = merged_df.apply(parse_json, axis=1).apply(pd.Series)
# save to csv
merged_df_expanded.to_csv('merged_df_expanded.csv')

# Filtering based on 'source' and 'AUC'
toxvaldb_df = merged_df_expanded[merged_df_expanded['source'] == 'toxvaldb']
low_auc_toxvaldb_df = toxvaldb_df[toxvaldb_df['AUC'] < 0.6]

# ===============================================
import json
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

def parse_json_string(json_string: str) -> dict:
    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        return {}  # Return an empty dictionary in case of a JSON parsing error

def expand_data_json(df: pd.DataFrame, json_column: str) -> pd.DataFrame:
    """Expand the JSON data in the specified column into DataFrame columns."""
    if json_column not in df.columns or df[json_column].isna().all():
        return df  # Return original DataFrame if column doesn't exist or is all NaN
    
    json_strings = df[json_column].dropna().tolist()
    if len(json_strings) == 0:
        return df

    with ProcessPoolExecutor(cpu_count()) as pool:
        # Map the parse_json_string function over the JSON strings using multiprocessing
        results = list(pool.map(parse_json_string, json_strings))
    
    expanded_df = pd.DataFrame(results)
    if expanded_df.empty:
        return df

    # Avoid column name overlap
    col_no_overlap = expanded_df.columns.difference(df.columns)
    expanded_df = df.join(expanded_df[col_no_overlap])

    return expanded_df

# Example usage
bindingdb_df_expanded = expand_data_json(bindingdb_df, 'data')
# save to csv
bindingdb_df_expanded.to_csv('bindingdb_expanded.csv')
bindingdb_df_expanded




# Add this as a new column to low_auc_df
low_auc_df['Imbalanceness'] = imbalanceness
# Sort and select specific columns

ranked_df = low_auc_df.sort_values(by='AUC', ascending=False)[['assay', 'AUC', 'Imbalanceness','source_id','NUM_POS','NUM_NEG']]
# print over 100 rows of rank_df
pd.set_option('display.max_rows', 50)
print(ranked_df.head(25))
print(ranked_df.tail(25))

# ===========================




# Print the content of the 'activity' table
print("Table: activity")
print(activity_df)
print("\n")

# Close the connection
conn.close()

# Count the number of unique source_ids in filtered_activity_info
unique_source_ids = filtered_activity_info['source_id'].unique()

print(f"Number of unique source_ids: {unique_source_ids}")

# ============================================
unique_low_auc_prob_assays = [2641 2445 2453 2449 2462 2578 2450 2486 2446 2444 2455 2859 2467 2458
 2537 2548 4219 3041 2507 2497 5292 2460 3045 2625 2603 2654 3730 2979
 2489 6561 4941 2609 2475 2619 3112 2646 3243 3207 5482 4473 3797 2935
 2653 2683 3700 3515 3101 5092 3247 4857 6612 4146 3522 6064 2912 5109
 4659 3257 5212 6676 3356 3693 2472 4689 5491 2685 6189 2926 5293 3065
 2640 6121 3811 5112 4409 6272 6441 4632 2905 2676 6336 6061 2666 4740
 2669 4959 3088 6192 2678 5287 2673 6715 6368 6507 5105 4737 4678 4474
 3279 3636 6746 5101 2672 2675 5588]

# Filter the activity_df for these prob_assays
filtered_activity_info = activity_df[activity_df['property_token'].isin(unique_low_auc_prob_assays)]
# group by source_id, and count distinct property_tokens
source_id_counts = filtered_activity_info.groupby('source_id')['property_token'].nunique()
print(source_id_counts)



# Display or process the filtered activity information
print(filtered_activity_info)

unique_source_ids = filtered_activity_info['source_id'].unique()
print(f"unique source_ids: {unique_source_ids}")

# Count occurrences of each source_id in the filtered_activity_info DataFrame
source_id_counts = filtered_activity_info['source_id'].value_counts()

# Print the counts for each source_id
print(source_id_counts)

# Count occurrences of each source_id in the original activity_df DataFrame
source_id_counts_original = activity_df['source_id'].value_counts()

# Print the counts for each source_id in the original DataFrame
print(source_id_counts_original)
