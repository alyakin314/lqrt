[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

# Robust Testing using Lq-Likelihood

This package implements a robust hypothesis testing procedure: the 
Lq-likelihood-ratio-type test (LqRT), introduced in Qin and Priebe (2017).
The code replicates and extends the R package which can be found here
[here](http://homepages.uc.edu/~qinyn/LqLR/).

## Installation

In order to install the package one needs to have python 3.x installed, then
either clone the repository from github and install using pip:
```
git clone https://github.com/alyakin314/lqrt
cd lqrt
pip install .
```
on install directly via pip:
```
pip install lqrt
```



## Import
The recommended import line is 
```python
>>> import lqrt
```
All examples below are performed using this import line.

(All examples also assume the usage of numpy and scipy.stats imported as:
```python
>>> import numpy as np
>>> from scipy import stats
```
This is unnecessary for the actual usage of the package, only for examples)


## Usage
There are three tests implemented in this repostory: Single sample, related 
samples (also known as paired) and independent samples (also known as 
unpaired). 

### Single sample
The single sample test performs the Lq-Likelihood-test for the mean of one group
of scores.
It is a robust two-sided test for the null hypothesis that the expected 
value (mean) of a sample of independent observations is equal to the given
population mean.
It can be thought of the as a robust version of the single sample t-test
(`scipy.stats.ttest_1samp`).

#### Example:

Test if mean of random sample is equal to true mean, and different mean. We
reject the null hypothesis in the second case and don’t reject it in the first
case.
```python
>>> np.random.seed(314)

>>> rvs1 = stats.multivariate_normal.rvs(0, 1, 50)

>>> lqrt.lqrtest_1samp(rvs1, 0)
Lqrtest_1sampResult(statistic=0.02388120731922072, pvalue=0.85)
>>> lqrt.lqrtest_1samp(rvs1, 1)
Lqrtest_1sampResult(statistic=35.13171144154751, pvalue=0.0)
```

### Related samples
The related samples test performs the Lq-Likelihood-test for the mean of 
two related samples of scores.
It is a robust two-sided test for the null hypothesis that 2 independent 
samples have identical average (expected) values.
It can be thought of as a robust version of the paired t-test 
(`scipy.stats.ttest_rel`).

#### Example:
A related samples test between two samples with identical means:
```python
>>> np.random.seed(314)

>>> rvs1 = stats.multivariate_normal.rvs(0, 1, 50)
>>> rvs2 = stats.multivariate_normal.rvs(0, 1, 50)

>>> lqrt.lqrtest_rel(rvs1, rvs2)
Lqrtest_relResult(statistic=0.22769245832813567, pvalue=0.66)
```

A related samples test between two samples with different means:

```python
>>> rvs3 = stats.multivariate_normal.rvs(1, 1, 50)
>>> lqrt.lqrtest_rel(rvs1, rvs3)
Lqrtest_relResult(statistic=27.827284933987784, pvalue=0.0)
```

### Independent samples
The independent samples tests performes the Lq-Likelihood-test for the mean
 of two independent samples of scores.
It is a robust two-sided test for the null hypothesis that 2 independent
samples have identical average (expected) values. 
it can be thought of as a robust version of the unparied t-test 
(`scipy.stats.ttest_ind`).
One can perform the test with or without the assumption that the samples have
 equal variance, which is estimated together. 
This is accomplished by setting the equal\_variance flag, similar to scipy's
t-test.

#### Example:
Test with samples with identical means with and without the equal variance
assumption. Note that in the unpaired set-up the samples need not to have
the same size:
```python
>>> np.random.seed(314)

>>> rvs1 = stats.multivariate_normal.rvs(0, 1, 50)
>>> rvs2 = stats.multivariate_normal.rvs(0, 1, 70)

>>> lqrt.lqrtest_ind(rvs1, rvs2)
LqRtest_indResult(statistic=0.00046542438241203854, pvalue=0.99)
>>> lqrt.lqrtest_ind(rvs1, rvs2, equal_var=False)
LqRtest_indResult(statistic=0.00047040017227573117, pvalue=0.97)
```

Test with samples with different means with and without the equal variance
assumption:

```python
>>> rvs3 = stats.multivariate_normal.rvs(1, 1, 70)

>>> lqrt.lqrtest_ind(rvs1, rvs3)
LqRtest_indResult(statistic=31.09168298440227, pvalue=0.0)
>>> lqrt.lqrtest_ind(rvs1, rvs3, equal_var=False)
LLqRtest_indResult(statistic=31.251454446588696, pvalue=0.0)
```

### q parameter
All test functions have an argument q which specifies the q parameter of 
the Lq-likelihood. The q should typically be within [0.5, 1.0] and the 
lower value is associated with a more robust test. If left unspecified of
set to None, the q is estimated using the empirical approximation to the
trace of the assymptotic covariance matrix procedure specified in Qin and
Priebe (2017).

#### Example:
```python
>>> x_true = np.random.normal(0.34, 1, 40)
>>> x_contamination = np.random.normal(0.34, 1, 10)
>>> x_sample = np.concatenate([x_true, x_contamination])

>>> lqrt.lqrtest_1samp(x_sample, 0, q = 0.9)
Lqrtest_1sampResult(statistic=1.1440379636073885, pvalue=0.28)
>>> lqrt.lqrtest_1samp(x_sample, 0, q = 0.6)
Lqrtest_1sampResult(statistic=3.710699836358458, pvalue=0.08)
>>> lqrt.lqrtest_1samp(x_sample, 0)
Lqrtest_1sampResult(statistic=5.5937088664291394, pvalue=0.06)
```

### Critical Value Bootstrap
The critical value for the tests is obtained via a bootstrap procedure, 
outlined in Qin and Priebe (2017). By default - 100 resamples are used,
but the number can be changed. Increasing the number of samples increases
the precision of the p-value, but adds on computational work.

#### Example:
```python
>>> x = np.random.normal(0, 1, 50)
>>> lqrt.lqrtest_1samp(x, 0) # takes ~0.25s
Lqrtest_1sampResult(statistic=0.36665186821102225, pvalue=0.58)
>>> lqrt.lqrtest_1samp(x, 0, bootstrap=1000) # takes ~1.5s
Lqrtest_1sampResult(statistic=0.36665186821102225, pvalue=0.541)
>>> lqrt.lqrtest_1samp(x, 0, bootstrap=10000) # takes ~15s
Lqrtest_1sampResult(statistic=0.36665186821102225, pvalue=0.5483)
```

## References
Qin, Yichen & E. Priebe, Carey. (2017). Robust Hypothesis Testing via Lq-Likelihood. Statistica Sinica. 27. 10.5705/ss.202015.0441. 
