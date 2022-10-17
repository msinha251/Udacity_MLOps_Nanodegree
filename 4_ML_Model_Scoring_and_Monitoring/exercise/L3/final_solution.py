import pickle
from re import X
import pandas as pd
import logging
import ast
import numpy as np
from sklearn.metrics import mean_squared_error

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

#load model:
model = pickle.load(open('./l3model.pkl', 'rb'))
logging.info(f"Model: {model}")

#load data:
data = pd.read_csv('./testdatafinal.csv')
logging.info(f"Data shape: {data.shape}")

X = data.loc[:,['timeperiod']].values.reshape(-1, 1)
y = data['sales'].values.reshape(-1, 1)

# make predictions
predictions = model.predict(X)
logging.info(f"Predictions: {predictions}")

# calculate accuracy
sse = mean_squared_error(y, predictions)
logging.info(f"SSE: {sse}")

# load previous scores
with open('./l3finalscores.txt', 'r') as f:
    previous_scores = ast.literal_eval(f.read())

logging.info(f"Previous scores: {previous_scores}")

#calculate nonparametric significance test
iqr = np.percentile(previous_scores, 75) - np.percentile(previous_scores, 25)
logging.info(f"iqr: {iqr.round(2)}")
nonparametrics_outlier_test = sse > np.quantile(previous_scores, 0.75) + 1.5*iqr
logging.info(f"Nonparametric significance test: {nonparametrics_outlier_test}")