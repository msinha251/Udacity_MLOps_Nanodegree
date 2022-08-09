# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This projects is all about identifying the credit card customers who are most likely to churn. It contains a python package that follows coding standard (PEP8) and engineering best practises. Follow below steps for running it interactively. 

## Files and data description
File and folder structure is as below:
```
    .
    ├── README.md
    ├── churn_library.py
    ├── churn_script_logging_and_tests.py
    ├── data
        ├── bank_data.csv
    ├── logs --> Folder for logs
    └── models --> Folder for generate models
```


## Running Files

#### Run Using conda virtual environment:
Create conda virtual environment:
`conda create --name udacity-ml-devops-nano python=3.6`
`conda activate udacity-ml-devops-nano`

Install required packages:
`python3.6 -m pip install -r requirements_py3.6.txt`

Run package:
`python3.6 churn_script_logging_and_tests.py`


#### Run with dockerfile:
```
docker build -t churn-prediction:0.1.0 .
docker run -it churn-prediction:0.1.0 bash
python3.6 churn_script_logging_and_tests.py
```



