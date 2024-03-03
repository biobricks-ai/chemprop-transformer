import sqlite3
import pandas as pd

# Path to your SQLite database
db_path = 'brick/cvae.sqlite'

# Create a connection
conn = sqlite3.connect(db_path)

# SQL query to map property tokens to property IDs
query = '''
    SELECT 
        p.property_id,
        p.property_token,
        p.title,
        p.source_id,
        p.data
    FROM 
        property p;
'''

# Use Pandas to execute the query and load the results into a DataFrame
df = pd.read_sql_query(query, conn)

# Display the DataFrame
print(df)

# Close the connection
conn.close()
