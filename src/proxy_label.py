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


def create_high_risk_label(rfm_df, random_state=42):
  scaler = StandardScaler()
  rfm_scaled = scaler.fit_transform(rfm_df[['Recency', 'Frequency', 'Monetary']])

  kmeans = KMeans(n_clusters=3, random_state=random_state)
  clusters = kmeans.fit_predict(rfm_scaled)
  rfm_df['cluster'] = clusters

  cluster_profiles = rfm_df.groupby('cluster')[['Recency', 'Frequency', 'Monetary']].mean()
  print(cluster_profiles)

  high_risk_cluster = cluster_profiles['Recency'].idxmax()

  rfm_df['is_high_risk'] = (rfm_df['cluster'] == high_risk_cluster).astype(int)
  return rfm_df[['CustomerId', 'is_high_risk']]


if __name__ == "__main__":
  df = pd.read_csv("../../data/raw/data.csv")
  rfm_df = create_rfm_features(df)
  proxy_labels = create_high_risk_label(rfm_df)

  processed_df = pd.read_csv("../../data/processed/model_ready_data.csv")
  final_df = pd.merge(processed_df, proxy_labels, on="CustomerId", how="inner")

  final_df.to_csv("../../data/processed/final_labeled_data.csv", index=False)
