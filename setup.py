from setuptools import setup, find_packages

setup(
    name='lion-tf',
    packages=find_packages(),
    version='0.0.1',
    license='MIT',
    description='Lion in TensorFlow 2',
    author='OrigamiDream',
    author_email='sdy36071@naver.com',
    url='https://github.com/OrigamiDream/lion-tf',
    install_requires=[
        'tensorflow>=2.11'
    ],
    keywords=[
        'machine learning',
        'deep learning',
        'tensorflow',
        'optimizers'
    ]
)
