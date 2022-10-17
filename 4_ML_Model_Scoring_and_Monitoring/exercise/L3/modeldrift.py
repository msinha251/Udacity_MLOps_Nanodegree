import ast
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

newr2=0.3625

###### PANDAS START ######
# #load previous scores
# previous_scores = pd.read_csv('./previousscores.csv')
# logging.info(f"Previous scores: {previous_scores.head()}")

# r2_scores = previous_scores[previous_scores['metric']=='r2']['score'].values
# logging.info(f"R2 scores: {r2_scores}")

###### PANDAS END ######

# load previousscores.txt
with open('./previousscores.txt', 'r') as f:
    previous_scores = ast.literal_eval(f.read())

#lowest of previous scores
logging.info(f"lowest of previsious scores: {min(previous_scores)}")

#std of previous scores
logging.info(f"std of previsious scores: {np.std(previous_scores).round(2)}")

###### RAW COMPARISON TEST ######

raw_comparison_test = newr2 < np.min(previous_scores)
logging.info(f"Model drift with raw_comparison_test: {raw_comparison_test}")


###### PARAMETRIC SIGNIFICANCE TEST ######
parametric_significance_test = newr2 < np.mean(previous_scores) - 2*np.std(previous_scores)
logging.info(f"Model drift with parametric_significance_test: {parametric_significance_test}")


###### NON-PARAMETRIC SIGNIFICANCE TEST ######
iqr = np.percentile(previous_scores, 75) - np.percentile(previous_scores, 25)
logging.info(f"iqr: {iqr.round(2)}")
nonparametric_significance_test = newr2 < np.quantile(previous_scores, 0.25) - 1.5*iqr
logging.info(f"Model drift with nonparametric_significance_test: {nonparametric_significance_test}")



