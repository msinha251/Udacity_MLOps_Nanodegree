
## Recap Lesson 1: Introduction to Deploying a Scalable ML Pipeline in Production

In this lesson, we covered:
* The big picture – what is this course about and why does it matter?
* The project you'll build at the end of the course.
* The prerequisites you'll need to have before you take this course.
* The business stakeholders you'll interact with as a professional in this field.
* The tools and environment you'll need and learn in this course.<br>
  * Python 
  * DVC for Data and Model versioning
  * Github actions / Heroku for CI-CD
  * FastAPI for API
  * Aequitas Package for checking Model bias

-----

## Recap Lesson 2: Performance Testing and Preparing a Model for Production

**Data Slicing**: It is when we compute the metrics of our model holding certain fixed features.

Typical model validation such as validation sets and **K-Fold Cross-Validation** can be thought of as looking at **horizontal slices of the data**, i.e. an overall view of the data and performance. **Data slicing** can be thought of as looking at **vertical slices of the data**. 
Data Slicing is generally used to validate the model or in order words used to find the areas of poor model performance.

![Overall VS data_slice](./images/data_slice.png)

**Data Bias**: This is not to be confused with the bias in "bias-variance trade-off" which is part of the model under or overfitting and model generalization.
Data bias can come from a multitude of sources such as human error. A few examples are

* *sampling error* - when there is a mismatch between the sample and the intended population, one cause can be too small of a sample or using a biased method of collection.
* *exclusion bias* - exclusion of a group from a survey, it could arise from survey methods (such as only using in-person surveys) or perhaps only collecting data from a platform that certain age-groups frequent when instead an all-age sample is desired.
* *recall bias* - the human error that occurs when people are asked to recall events from the past. Data could be unreliable or clouded by external perspective.

Data Bias tools such as **What-If Tool**, **FairLearn**, **FairML**, and **Aequitas**.

**Model Cards**: Model card is an approach for documenting all the details about the model like creation, data, use, shortcomings etc and make it available along within the model directory.<br>

There is no one way to write a model card! Suggested sections include:

**Model Details** such as who made it, type of model, training/hyperparameter details, and links to any additional documentation like a paper reference.
**Intended use** for the model and the intended users.
**Metrics** of how the model performs. Include overall performance and also key slices. A figure or two can convey a lot.
**Data** including the training and validation data. How it was acquired and processed.
**Bias** inherent either in data or model. This could also be included in the metrics or data section.
**Caveats**, if there are any.

Model Card Sample: 
```
Model Details
Justin C Smith created the model. It is logistic regression using the default hyperparameters in scikit-learn 0.24.2.

Intended Use
This model should be used to predict the acceptability of a car based off a handful of attributes. The users are prospective car buyers.

Metrics
The model was evaluated using F1 score. The value is 0.8960.

Data
The data was obtained from the UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/Car+Evaluation). The target class was modified from four categories down to two: "unacc" and "acc", where "good" and "vgood" were mapped to "acc".

The original data set has 1728 rows, and a 75-25 split was used to break this into a train and test set. No stratification was done. To use the data for training a One Hot Encoder was used on the features and a label binarizer was used on the labels.

Bias
According to Aequitas bias is present at the unsupervised and supervised level. This implies an unfairness in the underlying data and also unfairness in the model. From Aequitas summary plot we see bias is present in only some of the features and is not consistent across metrics.

```

-----

## Recap Lesson 3: Data and Model Versioning

**Data provenance** is the complete *origin*, *movement*, and *manipulation* of data.

*Origin*: It mean how the data was gathered.
*Movement*: It's the origin + any other jumps the data has made, e.g. data pulled from an API and moved to a company's S3 bucket for storage only to later be moved to HDFS for analysis.
*Manipulation*: Manipulation will be any kind of transformation or alternation on data. e.g. In NLP is documentation on how a data set is processed such as by changing the case or removing numbers or punctuation

**Recommanded way** is to add *Data provenance* in model card under data.


**Data Version Control (DVC)**: DVC is a complete solution for managing data, models and process of going from data to the model. 
DVC uses git for version control and remote storage like s3.

DVC commands are similar to git:
*dvc init* (git init) - initialiye dvc project
*dvc add (git add)* -  to add files.
*dvc pull /  dvc push* (git pull / git push) - Donwloading and uploading data from remote storage specified in dvc config.

Install DVC:
`brew install dvc`

