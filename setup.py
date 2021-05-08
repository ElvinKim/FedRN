#nsml: nsml/default_ml:cuda9
from distutils.core import setup

setup(
    name='FED LNL for NSML',
    version='0.1',
    description='FED LNL for NSML',
    install_requires=[
        'scikit-learn',
        'numpy',
        'torch==0.4.1',
        'torchvision==0.2.1',
    ],
)
