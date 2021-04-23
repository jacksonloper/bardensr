from setuptools import setup

setup(
    name='bardensr',
    author='Jackson Loper',
    version='0.3',
    include_package_data=True,
    description='barcode demixing through nonnegative spatial regression',
    packages=[
        'bardensr',
        'bardensr.barcodediscovery',
        'bardensr.benchmarks',
        'bardensr.meshes',
        'bardensr.optimization',
        'bardensr.plotting',
        'bardensr.reconstruction',
        'bardensr.rectangles',
        'bardensr.registration',
        'bardensr.spot_calling',
    ],
    package_data={
        '': ['*.hdf5']
    },
    install_requires=[
        'tensorflow >=2.4.0',
        'scikit-image',
        'tqdm',
        'matplotlib',
        'scipy',
        'pandas',
        'dataclasses',
    ]
)
