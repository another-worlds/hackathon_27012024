import pandas as pd
from pinecone import Pinecone
from pinecone import Index
from pinecone import ServerlessSpec, PodSpec
from dotenv import load_dotenv
import time

load_dotenv()

def get_df()-> pd.DataFrame: 
    df = pd.read_csv('tickers.csv', index_col=0)
    return df

def init_pinecone(index_name='rag', use_serverless=True) -> Index:
    # configure client
    pc = Pinecone()
    if use_serverless:
        spec = ServerlessSpec(cloud='aws', region='us-west-2')
    else:
        # if not using a starter index, you should specify a pod_type too
        spec = PodSpec()

    # check for and delete index if already exists
    if index_name in pc.list_indexes().names():
        pc.delete_index(index_name)

    # we create a new index
    pc.create_index(
            index_name,
            dimension=1536,  # dimensionality of text-embedding-ada-002
            metric='dotproduct',
            spec=spec
        )

    # wait for index to be initialized
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)
        
    # create or connect to  index
    index = pc.Index(index_name)
    index.describe_index_stats()
    return index

if __name__ == '__main__':
    print(get_df())