import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def create_rfm_features(df):
  df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
  snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)


  rfm = df.groupby('CustomerId').agg({
    'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,
    'TransactionId': 'count',
    'Amount': 'sum'
  }).reset_index()

  rfm.columns = ['CustomerId', 'Recency', 'Frequency', 'Monetary']
  return rfm
  