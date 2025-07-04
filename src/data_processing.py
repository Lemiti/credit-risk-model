import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


class DatetimeFeatureExtractor(BaseEstimator, TransformerMixin):
  def __init__(self, datetime_col='TransactionStartTime'):
    self.datetime_col = datetime_col

  def fit(self, X, y=None):
    return self
  

  def transform(self, X):
    X = X.copy()
    X[self.datetime_col] = pd.to_datetime(X[self.datetime_col])
    X['hour'] = X[self.datetime_col].dt.hour
    X['day'] = X[self.datetime_col].dt.day
    X['month'] = X[self.datetime_col].dt.month
    X['year'] = X[self.datetime_col].dt.year
    return X.drop(columns=[self.datetime_col])



class CustomerAggregateFeatures(BaseEstimator, TransformerMixin):
  def __init__(self, customer_id_col='CustomerId'):
    self.customer_id_col = customer_id_col
  
  def fit(self, X, y=None):
    return self
  
  def transform(self, X):
    X = X.copy()

    numeric_agg = X.groupby(self.customer_id_col).agg({
      'Amount': ['sum', 'mean', 'std'],
      'Value': ['sum', 'mean', 'std'],
      'TransactionId': 'count',
    })

    cat_agg = X.groupby(self.customer_id_col).agg({
      'ChannelId': lambda x: x.mode().iloc[0]  if not x.mode().empty else np.nan,
      'ProductCategory': lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
    })

    grouped = pd.concat([numeric_agg,cat_agg], axis=1).reset_index()

    grouped.columns = [self.customer_id_col] + [
      f'{i}_{j}' if j else i for i, j in grouped.columns[1:-2]
    ] + ['ChannelId', 'ProductCategory']

    return grouped

def build_pipeline():
    numeric_features = [
      'Amount_sum', 'Amount_mean', 'Amount_std',
      'Value_sum', 'Value_mean', 'Value_std',
      'TransactionId_count'
    ]
    catagorical_features = ['ChannelId', 'ProductCategory']

    numeric_transformer = Pipeline(steps=[
      ('imputer', SimpleImputer(strategy='median')),
      ('scaler', StandardScaler())
      ])

    categorical_transformer = Pipeline(steps=[
      ('imputer', SimpleImputer(strategy='most_frequent')), 
      ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
      ('num', numeric_transformer, numeric_features),
      ('cat', categorical_transformer, catagorical_features)
    ])

    pipeline = Pipeline(steps=[
      ('datetime', DatetimeFeatureExtractor()),
      ('aggregate', CustomerAggregateFeatures()),
      ('preprocessor', preprocessor)
    ])

    return pipeline

if __name__ == "__main__":
  # After loading data
  data = pd.read_csv("../../data/raw/data.csv")

  # Save CustomerId before pipeline removes it
  customer_ids = data[['CustomerId']].drop_duplicates()

  # Run pipeline
  pipeline = build_pipeline()
  processed = pipeline.fit_transform(data)

  # Rebuild DataFrame from output
  # (Optional: get column names too, as we did earlier)
  processed_df = pd.DataFrame(processed)

  # Concatenate CustomerId back
  final_model_ready_df = pd.concat([customer_ids.reset_index(drop=True), processed_df], axis=1)

  # Save to disk
  final_model_ready_df.to_csv("../../data/processed/model_ready_data.csv", index=False)

  print(processed.shape)