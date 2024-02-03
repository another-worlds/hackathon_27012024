import pandas as pd

def get_df()-> pd.DataFrame: 
    df = pd.read_csv('tickers.csv', index_col=0)
    return df

if __name__ == '__main__':
    print(get_df())