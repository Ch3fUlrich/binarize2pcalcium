from setuptools import setup

setup(
    name='binarize2pcalcium',
    version='0.1.4',
    packages=['binarize2pcalcium'],
    install_requires=[
        'numpy',
        'tqdm',
        'scipy',
        'matplotlib',
        'pyyaml',
        'networkx',
        'scikit-learn',
        'pandas',
        'opencv-python',
        'parmap',
    ],
)

