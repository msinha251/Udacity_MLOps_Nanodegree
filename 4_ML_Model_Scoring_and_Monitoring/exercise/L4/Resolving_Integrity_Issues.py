import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# load data:
df = pd.read_csv('samplefile3.csv')
logging.info(f"df.isnull().sum(): \n{df.isnull().sum()}")
logging.info(f"df: \n{df}")

logging.info(f"mean() of df: \n{df.mean()}")

# fill nan with mean
df = df.fillna(df.mean())
logging.info(f"df.isnull().sum(): \n{df.isnull().sum()}")
logging.info(f"df: \n{df}")