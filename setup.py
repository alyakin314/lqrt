import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='lqrt',
    version='1.0',
    author='Anton Alyakin',
    author_email='aalyaki1@jhu.edu',
    description='Robust Hypothesis testing via Lq-likelihood',
    long_description=long_description,
    url='https://github.com/alyakin314/lqrt',
    packages=['lqrt'],
    install_requires=['numpy', 'scipy', 'tqdm'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
