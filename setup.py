
from setuptools import setup, find_packages

setup(
    name='Bhaang',
    version='0.3',
    packages=find_packages(exclude=['tests*']),
    description='A general purpose package for medical imaging project and Lab Work',
    install_requires=['numpy', "torch", "torchvision", "pillow","transformers","tqdm", "tabulate", "medmnist", "albumentations", "pycocotools"],
    author='Harsh kumar',
    author_email = 'harshkumar1@iisc.ac.in'
)