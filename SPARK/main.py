import pinecone
import pandas as pd

# Initialize Pinecone
pinecone.init(api_key='d5edf74b-d4c7-4160-9628-389de98a9e02', environment='gcp-starter')

# Specify the index name
index_name = 'spark'

# Check if the index exists, if not create it
if index_name not in pinecone.list_indexes():
    pinecone.create_index(name=index_name, dimension=768)
    
# Connect to your Pinecone index
index = pinecone.Index(index_name=index_name)

# Load your CSV file using pandas
df = pd.read_csv('~/Downloads/merged_file.csv')

# Format your data for Pinecone's upsert operation
# Assuming 'text' column to be used as vector and 'target' as metadata
# Replace the vectorization step with your actual vectorization logic
# For demonstration, let's just use dummy vectors and assume 'text' is the ID
vectors = [(str(row['text']), [0]*768, {"target": row['target']}) for index, row in df.iterrows()]

# Upsert data into Pinecone
index.upsert(vectors=vectors)

# Check if upsert was successful
print("Upsert completed. Current index size:", index.describe_index_stats())
