import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression
import os
import logging

logging.basicConfig(level=logging.INFO)

#Read csv file
sales = pd.read_csv('./sales.csv')
logging.info(f"Data shape: {sales.shape}")

X = sales['timeperiod'].values.reshape(-1, 1)
y = sales['sales'].values.reshape(-1, 1)

#Train model
model = LinearRegression()
model.fit(X, y)
logging.info(f"Model trained")

#Save model
pickle.dump(model, open('./production/model.pkl', 'wb'))
logging.info(f"Saved model to production/model.pkl")