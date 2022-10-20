import ast
from asyncio.log import logger
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# load historic means:
with open('historicmeans.txt', 'r') as f:
    historicmean = ast.literal_eval(f.read())
logging.info(f"historicmean: {historicmean}")


# load data:
df = pd.read_csv('samplefile2.csv')
current_mean = df.mean().values
logging.info(f"current_mean: {current_mean}")

# find the percentage difference between the historic mean and the current mean
final_stability_output = [(current_mean[i] - historicmean[i])/historicmean[i] for i in range(len(historicmean))]
logging.info(f"final_stability_output: {final_stability_output}")

df.isnull().sum()
# calculate % of null values in each column
final_integrity_output = [df[col].isnull().sum()/len(df) for col in df.columns]
logging.info(f"final_integrity_output: {final_integrity_output}")
