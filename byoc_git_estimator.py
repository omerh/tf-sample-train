import sagemaker
from sagemaker.estimator import Estimator

project_name = 'byoc-tensorflow'
hyperparameters = {'epochs': 100, 'batch_size': 128, 'learning_rate': 0.01 }

github_config = {
    'repo': 'https://github.com/omerh/tf-sample-train.git',
    'branch': 'main'
}

sess = sagemaker.session.Session()
bucket = sess.default_bucket()

account_id = sess.account_id()
region = sess.boto_region_name

tensorflow_estimator = Estimator(
    source_dir='src',
    entry_point='train.py',
    git_config=github_config,
    instance_type='ml.c5.xlarge',
    instance_count=1,
    hyperparameters=hyperparameters,
    role=sagemaker.get_execution_role(),
    base_job_name='byoc-tf2',
    image_uri=f'{account_id}.dkr.ecr.{region}.amazonaws.com/byoc-tf2:v2.5.0')

# inputs = {'train': 'file://./data/train', 'test': 'file://./data/test'}
inputs = {'train': f's3://{bucket}/{project_name}/train', 'test': f's3://{bucket}/{project_name}/test'}

tensorflow_estimator.fit(inputs, logs=True)
