from setuptools import setup

setup(
    name='bardensr',
    author='Jackson Loper',
    version='0.1',
    include_package_data=True,
    description='barcode demixing through nonnegative spatial regression',
    packages=['bardensr','bardensr.singlefov'],
    package_data={
        '': ['*.pkl']
    },
    install_requires=[
        'tensorflow >=2.1.0',
        'scikit-image',
        'tqdm',
        'matplotlib',
        'scipy',
        'pandas',
    ]
)
