import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='lqrt',
    version='0.3.1',
    author='Anton Alyakin',
    author_email='aalyaki1@jhu.edu',
    description='Robust Hypothesis testing via Lqlikelihood',
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
