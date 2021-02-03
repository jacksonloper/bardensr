from setuptools import setup

setup(
    name='bardensr',
    author='Jackson Loper',
    version='0.2',
    include_package_data=True,
    description='barcode demixing through nonnegative spatial regression',
    packages=['bardensr','bardensr.singlefov','bardensr.benchmarks',
                'bardensr.barcodediscovery','bardensr.meshes'],
    package_data={
        '': ['*.hdf5']
    },
    install_requires=[
        'tensorflow >=2.3.0',
        'scikit-image',
        'tqdm',
        'matplotlib',
        'scipy',
        'pandas',
        'dataclasses',
    ]
)
