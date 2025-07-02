import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


class DatetimeFeatureExtractor(BaseEstimator, TransformerMixin):
  def __init__(self, datetime_col):
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



  def build_pipeline():
    numeric_features = ['Amount', 'Value']
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
      ('datetime', DatetimeFeatureExtractor('TransactionStartTime')),
      ('preprocessor', preprocessor)
    ])

    return pipeline