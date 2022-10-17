import ast
import pandas as pd
import numpy as np
import logging

recent_r2=0.6
recent_sse=52938

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

previous_scores = pd.read_csv('./previousscores.csv')
logging.info(f"Previous scores: {previous_scores}")

# maximum version:
thisversion = previous_scores['version'].max() + 1
logging.info(f"This version: {thisversion}")

# append new scores to previous scores
new_r2 = pd.DataFrame({'metric': 'r2', 'version': thisversion, 'score': recent_r2}, index=[0])
new_sse = pd.DataFrame({'metric': 'sse', 'version': thisversion, 'score': recent_sse}, index=[0])

logging.info('recent_sse: %s', recent_sse)
if recent_sse < previous_scores[previous_scores['metric']=='sse']['score'].min():
    logging.info(f"New model is better than previous model")
    new_scores = previous_scores.append(new_r2).append(new_sse)
    new_scores.to_csv('./newscores.csv', index=False)
