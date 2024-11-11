import pandas as pd
import numpy as np


df = pd.read_csv('WineQT.csv')
df.head(10)

df.shape

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X=df.drop(['Id', 'quality'], axis=1)
y=df['quality']
X.info()

#remove outliers
def remove_outliers(df2, col):
  Q1 = df2[col].quantile(0.25)
  Q3 = df2[col].quantile(0.75)
  IQR = Q3 - Q1
  lower_bound = Q1 - 1.5*IQR
  upper_bound = Q3 + 1.5*IQR
  df2 = df2[df2[col] >= lower_bound]
  df2 = df2[df2[col] <= upper_bound]
  return df2

for cols in X.columns:
  df = remove_outliers(df, cols)


X.shape

# import pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

preprocessor = Pipeline([
    ('scaler', StandardScaler())
])

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
pipeline1 = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])
pipeline2 = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])


import mlflow

def log_model_with_mlflow(model, model_name):
    with mlflow.start_run():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)

        mlflow.sklearn.log_model(model, model_name)
        print(f'{model_name} logged with accuracy={accuracy}')
    mlflow.end_run()

log_model_with_mlflow(pipeline1, "Logistic Regression")
log_model_with_mlflow(pipeline2, "Random Forest Classifier")
