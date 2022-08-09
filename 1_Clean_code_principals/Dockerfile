FROM python:3.6-slim

LABEL description="Dockerfile to build a python package for churn prediction"
LABEL license="MIT"
LABEL maintainer="Mahesh Sinha"
LABEL name="churn-prediction"
LABEL version="0.1.0"

COPY . .

RUN pip install -r requirements_py3.6.txt