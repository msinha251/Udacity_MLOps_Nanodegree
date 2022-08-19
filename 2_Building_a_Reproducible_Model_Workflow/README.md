Prerequisite:
* conda 

### Create conda environment for this course:
`conda create --name udacity-ml-devops-nano-2 python=3.8 mlflow jupyter pandas matplotlib requests -c conda-forge`

### activate created environment:
`conda activate udacity-ml-devops-nano-2`
                OR
`source activate udacity-ml-devops-nano-2`

### Since weight/biases is not available on conda, install it via pip:
`pip install wandb`

### wandb login:
`wandb login`

### test wandb:
```
source activate udacity
echo "wandb test" > wandb_test
wandb artifact put -n testing/artifact_test wandb_test
```


Machine learning pipeline:

![image.png](ml-pipeline.png)

## Recap Lesson 1: Introduction to Reproducible Model Workflows

**Machine Learning Operations**: MLops is a set of best practices and methods for an efficient end-to-end development and operation of performant, scalable, reliable, automated and reproducible ML solutions in a real production setting.

**Reproducible Workflow**: An orchestrated, tracked and versioned workflow that can be reproduced and inspected.

-----------------

