import pandas as pd
import numpy as np
import os
import logging
import pickle
from sklearn.metrics import f1_score

logging.basicConfig(level=logging.INFO)

#load model
model = pickle.load(open('./samplemodel.pkl', 'rb'))
logging.info(f"Model loaded")

# load test data:
testdata = pd.read_csv('./testdata.csv')
logging.info(f"Test data shape: {testdata.shape}")

X = testdata.loc[:,['col1','col2']].values.reshape(-1, 2)
y = testdata['col3'].values.reshape(-1, 1)

# make predictions
predictions = model.predict(X)
logging.info(f"Predictions: {predictions}")

# calculate accuracy
f1_score = f1_score(y, predictions)
logging.info(f"F1 score: {f1_score}")


