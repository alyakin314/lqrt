import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='lqrt',
    version='0.3.2',
    author='Anton Alyakin',
    author_email='aalyaki1@jhu.edu',
    description='Robust Hypothesis Testing of Location Parameters using Lq-Likelihood-Ratio-Type Test in Python',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/alyakin314/lqrt',
    packages=['lqrt'],
    install_requires=['numpy', 'scipy'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
