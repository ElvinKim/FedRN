#nsml: pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime
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
