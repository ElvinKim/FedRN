#nsml: floydhub/pytorch:1.7.1-gpu.cuda10cudnn7-py3.57
from distutils.core import setup

setup(
    name='FED LNL for NSML',
    version='0.1',
    description='FED LNL for NSML',
    install_requires=[
        'scikit-learn',
        'numpy',
        'torch==1.7.0',
        'torchvision==0.8.1',
    ],
)
