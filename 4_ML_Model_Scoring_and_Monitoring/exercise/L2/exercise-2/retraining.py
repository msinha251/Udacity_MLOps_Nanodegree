import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
import os
import logging

logging.basicConfig(level=logging.INFO)

###################Reading Records#############
with open('datalocation.txt', 'r') as f:
    datalocation = f.read()
logging.info(f"Reading data from {datalocation}")

with open('deployedmodelname.txt', 'r') as f:
    deployedname = f.read()

##################Re-training a Model#############

trainingdata = pd.read_csv('./'+datalocation)

X=trainingdata.loc[:,['col1','col2']].values.reshape(-1, 2)
y=trainingdata['col3'].values.reshape(-1, 1)

logit=LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                    multi_class='auto', n_jobs=None, penalty='l2',
                    random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                    warm_start=False)
                    
model = logit.fit(X,y)

############Pushing to Production###################

pickle.dump(model, open('./production/'+deployedname, 'wb'))
logging.info(f"Saved model to production/{deployedname}")






