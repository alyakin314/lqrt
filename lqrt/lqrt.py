# Copyright 2019 Anton Alyakin (alyakin314.github.io)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import numpy as np
from scipy.stats import norm
from collections import namedtuple

Lqrtest_1sampResult = namedtuple('Lqrtest_1sampResult',
                                 ('statistic', 'pvalue'))

def lqrtest_1samp(x, u,
                  q=None, bootstrap=100,
                  random_state=None):
    """
    Calculates the Lq-Likelihood-ratio-type test for the location of a single
    sample from a normal distribution.

    This is a robust two-sided test for the null hypothesis that the expected
    value (mean) of a sample of independent observations `x` is equal to the
    given population mean, `u`.

    Parameters
    ----------
    x : array_like
        Sample observation. Must be one-dimensional.
    u : float
        expected value in null hypothesis.
    q : float, optional (default=None)
        Parameter of the Lq-Likelihood. If None - it is estimated.
    bootstrap : integer, optional (default=100)
        Number of bootstrap iterations performed for the p-value estimation.
    random_state : int or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If None, the random number generator is the default of np.random.

    Returns
    -------
    statistic: float
        LqR Test statistic
    p-value : float
        Two-tailed p-value

    References
    ----------
    .. [1] Qin, Yichen & E. Priebe, Carey. (2017). Robust Hypothesis Testing
    via Lq-Likelihood. Statistica Sinica. 27. 10.5705/ss.202015.0441.

    .. [2] http://homepages.uc.edu/~qinyn/LqLR/

    Examples
    --------
    >>> import lqrt
    >>> import numpy as np
    >>> from scipy import stats

    >>> np.random.seed(314)
    >>> rvs1 = stats.multivariate_normal.rvs(0, 1, 50)
    
    Test if mean of random sample is equal to true mean, and different mean.
    We reject the null hypothesis in the second case and don’t reject it in
    the first case.
    
    >>> lqrt.lqrtest_1samp(rvs1, 0)
    Lqrtest_1sampResult(statistic=0.02388120731922072, pvalue=0.85)
    >>> lqrt.lqrtest_1samp(rvs1, 1)
    Lqrtest_1sampResult(statistic=35.13171144154751, pvalue=0.0)

    The argument q specifies the q parameter of the Lq-likelihod. The q is 
    typically within [0.5, 1.0] adn the lower associated with a more robust 
    test. It is estimated if unspecified or set to None.

    >>> rvs2 = np.concatenate([stats.multivariate_normal.rvs(0.32, 1, 45),
    ...                        stats.multivariate_normal.rvs(0.32, 50, 5)])
    >>> lqrt.lqrtest_1samp(rvs2, 0, q=0.9)
    Lqrtest_1sampResult(statistic=2.239547159197258, pvalue=0.09)
    >>> lqrt.lqrtest_1samp(rvs2, 0, q=0.6)
    Lqrtest_1sampResult(statistic=3.4268748448623256, pvalue=0.02)
    >>> lqrt.lqrtest_1samp(rvs2, 0)
    Lqrtest_1sampResult(statistic=2.7337572196229587, pvalue=0.03)

    The critical value for the tests is obtained via a bootstrap procedure.
    Increasing the number of samples increases the precision of the p-value,
    but adds on computational work.

    >>> lqrt.lqrtest_1samp(rvs1, 0, bootstrap=100) # takes ~0.3s
    Lqrtest_1sampResult(statistic=0.02388120731922072, pvalue=0.85)
    >>> lqrt.lqrtest_1samp(rvs1, 0, bootstrap=1000) # takes ~1.5s
    Lqrtest_1sampResult(statistic=0.02388120731922072, pvalue=0.875)
    >>> lqrt.lqrtest_1samp(rvs1, 0, bootstrap=10000) # takes ~15s
    Lqrtest_1sampResult(statistic=0.02388120731922072, pvalue=0.8743)

    """
    # Initialize Random Number Generator
    if random_state is not None:
        np.random.seed(random_state)
    # Shape wrangling
    x = np.asarray(x)
    if x.ndim > 1:
        raise ValueError('Sample x must be one-dimensional.')
    # Estimate q
    if q is None:
        q = _lqrt_select_q(x)
    # Estimating the Critical Value
    D_qs = _critical_values_1samp(x, u, q, bootstrap=bootstrap)
    # Lq-likelihood Test
    D_q_test = _lqr_test_statistic_1samp(x, u, q)
    # p-value
    p_value = np.sum(D_q_test < D_qs) / bootstrap
    return Lqrtest_1sampResult(D_q_test, p_value)


Lqrtest_relResult = namedtuple('Lqrtest_relResult', ('statistic', 'pvalue'))


def lqrtest_rel(x_1, x_2,
                q=None, bootstrap=100,
                random_state=None):
    """
    Calculates the Lq-Likelihood-ratio-type test for the equivalence of the
    location parameters of two paired/matched samples from a noraml
    dstribution.

    This is a robust two-sided test with the null hypothesis that 2 independent
    samples have identical average (expected) values.

    This function is a wrapper of a one-sample test.

    Parameters
    ----------
    x_1 : array_like
        First sample data; must be one-dimensional, same size as x_2.
    x_2 : array_like
        Second sample data; must be one-dimensional, same size as x_1.
    q : float, optional (default=None)
        Parameter of the Lq-Likelihood. If None - it is estimated.
    bootstrap : integer, optional (default=100)
        Number of bootstrap iterations performed for the p-value estimation.
    random_state : int or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If None, the random number generator is the default of np.random.

    Returns
    -------
    statistic: float
        LqR Test statistic
    p-value : float
        Two-tailed p-value

    References
    ----------
    .. [1] Qin, Yichen & E. Priebe, Carey. (2017). Robust Hypothesis Testing
    via Lq-Likelihood. Statistica Sinica. 27. 10.5705/ss.202015.0441.

    .. [2] http://homepages.uc.edu/~qinyn/LqLR/

    Examples
    --------
    >>> import lqrt
    >>> from scipy import stats
    >>> import numpy as np
    >>> np.random.seed(314)

    Test with samples with identical means:

    >>> rvs1 = stats.multivariate_normal.rvs(0, 1, 50)
    >>> rvs2 = stats.multivariate_normal.rvs(0, 1, 50)
    >>> lqrt.lqrtest_rel(rvs1, rvs2)
    Lqrtest_relResult(statistic=0.22769245832813567, pvalue=0.66)

    Test with samples with different means:

    >>> rvs3 = stats.multivariate_normal.rvs(1, 1, 50)
    >>> lqrt.lqrtest_rel(rvs1, rvs3)
    Lqrtest_relResult(statistic=27.827284933987784, pvalue=0.0)

    The argument q specifies the q parameter of the Lq-likelihod. The q is 
    typically within [0.5, 1.0] adn the lower associated with a more robust 
    test. It is estimated if unspecified or set to None.

    >>> rvs4 = np.concatenate([stats.multivariate_normal.rvs(0, 1, 45),
    ...                        stats.multivariate_normal.rvs(0, 50, 5)])
    >>> rvs5 = np.concatenate([stats.multivariate_normal.rvs(0.32, 1, 45),
    ...                        stats.multivariate_normal.rvs(0.32, 50, 5)])
    >>> lqrt.lqrtest_rel(rvs4, rvs5, q=0.9)
    Lqrtest_relResult(statistic=1.0115551154702587, pvalue=0.16)
    >>> lqrt.lqrtest_rel(rvs4, rvs5, q=0.6)
    Lqrtest_relResult(statistic=0.9560522020834696, pvalue=0.21)
    >>> lqrt.lqrtest_rel(rvs4, rvs5)
    Lqrtest_relResult(statistic=1.2020064196410374, pvalue=0.18)

    The critical value for the tests is obtained via a bootstrap procedure.
    Increasing the number of samples increases the precision of the p-value,
    but adds on computational work.

    >>> lqrt.lqrtest_rel(rvs1, rvs2, bootstrap=100) # takes ~0.35s
    Lqrtest_relResult(statistic=0.22769245832813567, pvalue=0.61)
    >>> lqrt.lqrtest_rel(rvs1, rvs2, bootstrap=1000) # takes ~1.7s
    Lqrtest_relResult(statistic=0.22769245832813567, pvalue=0.646)
    >>> lqrt.lqrtest_rel(rvs1, rvs2, bootstrap=10000) # takes ~17s
    Lqrtest_relResult(statistic=0.22769245832813567, pvalue=0.6361)

    """
    # Ensure Dimensions
    x_1 = np.asarray(x_1)
    x_2 = np.asarray(x_2)
    if x_1.ndim > 1 or x_2.ndim > 1:
        raise ValueError('Samples x_1 and x_2 must be one-dimensional.')
    if x_1.shape != x_2.shape:
        raise ValueError('Samples x_1 and x_2 must have same lengths.')
    result = lqrtest_1samp(x_1-x_2, 0,
                           q=q, bootstrap=bootstrap,
                           random_state=random_state)
    return Lqrtest_relResult(*result)


Lqrtest_indResult = namedtuple('LqRtest_indResult', ('statistic', 'pvalue'))


def lqrtest_ind(x_1, x_2, equal_var=True,
                q=None, bootstrap=100,
                random_state=None):
    """
    Calculates the Lq-Likelihood-test for the mean of TWO INDEPENDENT samples
    of scores.

    This is a robust two-sided test for the null hypothesis that 2 independent
    samples have identical average (expected) values.

    Parameters
    ----------
    x_1 : array_like
        First sample data; must be one-dimensional.
    x_2 : array_like
        Second sample data; must be one-dimensional.
    equal_var: boolean, optional (default=True)
        If True - makes the equal population variance assumptions.
        If False - estimates them independently.
    q : float, optional (default=None)
        Parameter of the Lq-Likelihood. If None - it is estimated.
    bootstrap : integer, optional (default=100)
        Number of bootstrap iterations performed for the p-value estimation.
    random_state : int or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If None, the random number generator is the default of np.random.

    Returns
    -------
    statistic: float
        LqR Test statistic
    p-value : float
        Two-tailed p-value

    References
    ----------
    .. [1] Qin, Yichen & E. Priebe, Carey. (2017). Robust Hypothesis Testing
    via Lq-Likelihood. Statistica Sinica. 27. 10.5705/ss.202015.0441.

    .. [2] http://homepages.uc.edu/~qinyn/LqLR/

    Examples
    --------

    >>> import lqrt
    >>> from scipy import stats
    >>> import numpy as np
    >>> np.random.seed(314)

    Test with samples with identical means with and without the equal variance
    assumption. Note that in the unpaired set-up the samples need not to have
    the same size:

    >>> rvs1 = stats.multivariate_normal.rvs(0, 1, 50)
    >>> rvs2 = stats.multivariate_normal.rvs(0, 1, 70)
    >>> lqrt.lqrtest_ind(rvs1, rvs2)
    LqRtest_indResult(statistic=0.00046542438241203854, pvalue=0.99)
    >>> lqrt.lqrtest_ind(rvs1, rvs2, equal_var=False)
    LqRtest_indResult(statistic=0.00047040017227573117, pvalue=0.97)

    Test with samples with different means with and without the equal variance
    assumption:

    >>> rvs3 = stats.multivariate_normal.rvs(1, 1, 70)
    >>> lqrt.lqrtest_ind(rvs1, rvs3)
    LqRtest_indResult(statistic=31.09168298440227, pvalue=0.0)
    >>> lqrt.lqrtest_ind(rvs1, rvs3, equal_var=False)
    LLqRtest_indResult(statistic=31.251454446588696, pvalue=0.0)

    The argument q specifies the q parameter of the Lq-likelihod. The q is 
    typically within [0.5, 1.0] adn the lower associated with a more robust 
    test. It is estimated if unspecified or set to None.

    >>> rvs4 = np.concatenate([stats.multivariate_normal.rvs(0, 1, 45), 
    ...                        stats.multivariate_normal.rvs(0, 50, 5)])
    >>> rvs5 = np.concatenate([stats.multivariate_normal.rvs(0.32, 1, 65),
    ...                        stats.multivariate_normal.rvs(0.32, 50, 5)])
    >>> lqrt.lqrtest_ind(rvs4, rvs5, q=0.9, equal_var=False)
    LqRtest_indResult(statistic=2.5523157340379043, pvalue=0.08)
    >>> lqrt.lqrtest_ind(rvs4, rvs5, q=0.6, equal_var=False)
    LqRtest_indResult(statistic=2.8616655080245437, pvalue=0.11)
    >>> lqrt.lqrtest_ind(rvs4, rvs5, equal_var=False)
    LqRtest_indResult(statistic=3.3919342023621084, pvalue=0.01)

    The critical value for the tests is obtained via a bootstrap procedure.
    Increasing the number of samples increases the precision of the p-value,
    but adds on computational work.

    >>> lqrt.lqrtest_ind(rvs1, rvs2, bootstrap=100) # takes ~0.5s
    LqRtest_indResult(statistic=0.00046542438241203854, pvalue=1.0)
    >>> lqrt.lqrtest_ind(rvs1, rvs2, bootstrap=1000) # takes ~2.5s
    LqRtest_indResult(statistic=0.00046542438241203854, pvalue=0.979)
    >>> lqrt.lqrtest_ind(rvs1, rvs2, bootstrap=10000) # takes ~22.5s
    LqRtest_indResult(statistic=0.00046542438241203854, pvalue=0.9848)

    """
    # Initialize Random Number Generator
    if random_state is not None:
        np.random.seed(random_state)
    # Ensure Dimensions
    x_1 = np.asarray(x_1)
    x_2 = np.asarray(x_2)
    if x_1.ndim > 1 or x_2.ndim > 1:
        raise ValueError('Samples x_1 and x_2 must be one-dimensional.')
    # Estimate q
    if q is None:
        q = _lqrt_select_q(x_1, x_2, equal_var=equal_var)
    # Estimating the critical value
    D_qs = _critical_values_ind(x_1, x_2, q,
                                bootstrap=bootstrap,
                                equal_var=equal_var)
    # Lq-likelihood Ratio Test
    if equal_var:
        D_q_test = _lqr_test_statistic_ind_equal_var(x_1, x_2, q)
    else:
        D_q_test = _lqr_test_statistic_ind_unequal_var(x_1, x_2, q)
    # p-value
    p_value = np.sum(D_q_test < D_qs) / bootstrap
    return Lqrtest_indResult(D_q_test, p_value)


def _lq_function(x, q=1):
    """
    Computes the Lq function (reparametrized Box-Cox transformation).
    Effectively equivalent to scipy.stats.boxcox(x, 1-q).

    Parameters
    ----------
    x : array_like
        Input to the Lq function.
    q : int, optional
        Parameter of the Lq. Defaults to 1.

    Returns
    -------
    lq : float
        Evaluated Lq function

    """
    if q == 1:
        lq = np.log(x)
    else:
        lq = (np.array(x) ** (1 - q) - 1)/(1 - q)
    return lq


def _lql_normal(x, u, var, q=1):
    """
    Computes the Lq-likelihood of a sample from a one-dimensional Gaussian.

    Parameters
    ----------
    x : array_like
        Sample.
    u : float
        Mean parameter of the Gaussian distribution.
    var : float
        Variance parameter of the Gaussian distribution.
    q : int, optional
        Parameter of the Lq. Defaults to 1.

    Returns
    -------
    lql: float
        Lq-likelihood of the sample under the Gaussian model.

    """
    densities = norm.pdf(x, u, np.sqrt(var))
    lql = np.sum(_lq_function(densities, q))
    return lql


def _mlqe_normal(x, q=1, true_u=None, tol=0.00000001, iterations=1000):
    """
    Computes the Maximum Lq-Likelihood Estimatior for mean and variance
    parameters of a univariate normal distribution (or only variance if mean
    is provided) using an iterative reweighting algorithm.

    Parameters
    ----------
    x : array_like
        Data; must be one-dimensional.
    q : int, optional
        Parameter of the MLqE. Defaults to 1.
    true_u : float, optional
        Mean parameter if such is known a priori. If None - it is estimated.
        Defaults to None.
    tol: float, optional
        Tolerance parameter for optimization.
        Defaults to 0.0000001.
    iterations: integer, optional
        Number of iterations performed if convergence is not reached.
        Defaults to 1000.

    Returns
    -------
    u : float
        Estimated means parameter of the sample.
    var : float
        Estimated variance parameter of the sample, NOT bias-corrected.

    """
    # Ensure Dimensions
    x = np.asarray(x)
    if x.ndim > 1:
        raise ValueError('Sample x must be one-dimensional.')
    # Compute initial parameters
    if true_u is None:
        u = np.mean(x)
        estimate_mean = True
    else:
        old_u = u = true_u
        estimate_mean = False
    var = np.sum((x - u) ** 2) / len(x)
    # Iterative Reweighting algorithm
    for i in range(iterations):
        # Compute weights
        w = norm.pdf(x, u, np.sqrt(var)) ** (1-q)
        # Update the mean
        if estimate_mean:
            old_u = u
            u = np.sum(x * w) / np.sum(w)
        # Update the variance
        old_var = var
        var = np.sum((x - u) ** 2 * w) / np.sum(w)
        # Prevents the zero variance
        var = np.max([var, np.finfo(np.float64).eps])
        # Convergence Condition
        if np.abs(u - old_u) + np.abs(var - old_var) < tol:
            break
    return u, var


def _mlqe_normal_2samp_equal_var(x_1, x_2, q=1,
                                 tol=0.0000001, iterations=1000):
    """
    Computes the Maximum Lq-Likelihood Estimatior for mean and variance
    parameters of two samples univariate normal distributions that are known
    to have a shared variance using an iterative reweighting algorithm.
    (If the variance is not shared - use two separate one-sampled mlqe
    instead)

    Parameters
    ----------
    x_1 : array_like
        First sample data; must be one-dimensional.
    x_2 : array_like
        Second sample data; must be one-dimensional.
    q : float, optional
        Parameter of the MLqE. Defaults to 1.
    tol: float, optional
        Tolerance parameter for optimization.
        Defaults to 0.0000001.
    iterations: integer, optional
        Number of iterations performed if convergence is not reached.
        Defaults to 1000.

    Returns
    -------
    u_1 : float
        Estimated mean parameter (mu) of the first sample.
    u_2 : float
        Estimated mean parmaeter (mu) of the second sample.
    var : float
        Estimated variance parameter for both samples, NOT bias-corrected.
    """
    # Ensure Dimensions
    x_1 = np.asarray(x_1)
    x_2 = np.asarray(x_2)
    if x_1.ndim > 1 or x_2.ndim > 2:
        raise ValueError('Samples x_1 and x_2 must be one-dimensional.')
    # Compute initial parameters
    u_1 = np.mean(x_1)
    u_2 = np.mean(x_2)
    x_combined_centered = np.concatenate([x_1 - u_1, x_2 - u_2])
    var = np.sum((x_combined_centered) ** 2) / len(x_combined_centered)
    # Iterative Reweighting algorithm
    for i in range(iterations):
        # Compute Weights
        w_1 = norm.pdf(x_1, u_1, np.sqrt(var)) ** (1-q)
        w_2 = norm.pdf(x_2, u_2, np.sqrt(var)) ** (1-q)
        w = np.concatenate([w_1, w_2])
        # Store old mean and variance
        old_u_1 = u_1
        old_u_2 = u_2
        old_var = var
        # Compute new mean and variance
        u_1 = np.sum(x_1 * w_1) / np.sum(w_1)
        u_2 = np.sum(x_2 * w_2) / np.sum(w_2)
        x_combined_centered = np.concatenate([x_1 - u_1, x_2 - u_2])
        var = np.sum(x_combined_centered ** 2 * w) / np.sum(w)
        # Prevents the zero variance
        var = np.max([var, np.finfo(np.float64).eps])
        # The following case happens when one sample converges to a singularity,
        # but the other doesn't, or not as quickly.
        # Breaking the loop prevents the nan's.
        if (np.sum(norm.pdf(x_1, u_1, np.sqrt(var)) ** (1-q)) == 0
            or np.sum(norm.pdf(x_2, u_2, np.sqrt(var)) ** (1-q)) == 0):
            var = np.finfo(np.float64).eps
            break
        # Convergence Condition
        if (np.abs(u_1 - old_u_1) + np.abs(u_2 - old_u_2) +
                np.abs(var - old_var) < tol):
            break
    return u_1, u_2, var




def _mlqe_normal_2samp_equal_mean(x_1, x_2, q=1,
                                  tol=0.0000001, iterations=1000):

    """
    Computes the Maximum Lq-Likelihood Estimatior for mean and variance
    parameters of two samples univariate normal distributions that are known
    to have a shared mean, using an iterative reweighting algorithm.
    (If the mean is not shared - use two separate one-sampled mlqe instead)

    Parameters
    ----------
    x_1 : array_like
        First sample data; must be one-dimensional.
    x_2 : array_like
        Second sample data; must be one-dimensional.
    q : float, optional
        Parameter of the MLqE. Defaults to 1.
    tol: float, optional
        Tolerance parameter for optimization.
        Defaults to 0.0000001.
    iterations: integer, optional
        Number of iterations performed if convergence is not reached.
        Defaults to 1000.

    Returns
    -------
    u_1 : float
        Estimated mean parameter (mu) of both samples
    var_1 : float
        Estimated variance parameter for the first sample, NOT bias-corrected.
    var_2 : float
        Estimated variance parameter for the second sample, NOT bias-corrected.

    """
    # Ensure Dimensions
    x_1 = np.asarray(x_1)
    x_2 = np.asarray(x_2)
    if x_1.ndim > 1 or x_2.ndim > 1:
        raise ValueError('Samples x_1 and x_2 must be one-dimensional.')
    # Compute initial parameters
    x_combined = np.concatenate([x_1, x_2])
    u = np.mean(x_combined)
    var_1 = np.sum((x_1 - u) ** 2) / len(x_1)
    var_2 = np.sum((x_2 - u) ** 2) / len(x_2)
    # Iterative Reweighting algorithm
    for i in range(iterations):
        # Compute weights
        w_1 = norm.pdf(x_1, u, np.sqrt(var_1)) ** (1-q)
        w_2 = norm.pdf(x_2, u, np.sqrt(var_2)) ** (1-q)
        w = np.concatenate([w_1, w_2])
        # Store old mean and variance
        old_u = u
        old_var_1 = var_1
        old_var_2 = var_2
        # Compute new mean and variance
        u = np.sum(x_combined * w) / np.sum(w)
        var_1 = np.sum((x_1 - u) ** 2 * w_1) / np.sum(w_1)
        var_2 = np.sum((x_2 - u) ** 2 * w_2) / np.sum(w_2)
        # Prevents the zero variance
        var_1 = np.max([var_1, np.finfo(np.float64).eps])
        var_2 = np.max([var_2, np.finfo(np.float64).eps])
        # Convergence Condition
        if (np.abs(u - old_u) + np.abs(var_1 - old_var_1) +
                np.abs(var_2 - old_var_2) < tol):
            break
    return u, var_1, var_2


def _lqrt_empirical_variance(x, u, var, q):
    """
    Computes the empirical approximation to the assymptotic variance of the
    mean estimator of the provided sample under the Maximum Lq-likelihood
    estimation technique and 1-dimensional Gaussian model.

    Parameters
    ----------
    x : array_like
        Sample.
    u : float
        Mean parameter of the Gaussian distribution.
    var : float
        Variance parameter of the Gaussian distribution.
    q : int, optional
        Parameter of the Lq.

    Returns
    -------
    empirical_var: float
        Empirical approximation to the assymptotic variance.

    """
    # Ensure dimensions
    x = np.asarray(x)
    if x.ndim > 1:
        raise ValueError('Sample x must be one-dimensional')
    # Compute weights
    w = norm.pdf(x, u, np.sqrt(var)) ** (1-q)
    e_phi2 = np.mean(((x - u) / var * w) ** 2)
    e_dphi = np.mean((- 1 / var + (1 - q) * (x - u) ** 2 / (var ** 2)) * w)
    empirical_var = e_phi2 / (e_dphi ** 2)

    return empirical_var


def _lqrt_select_q(x_1, x_2=None,
                   low_bound=0.5, high_bound=1.0, granularity=0.01,
                   equal_var=True):
    """
    Adaptively selects q via minimizing trace of assymptotic variance.
    Handles both one- and two-sample cases.

    Parameters
    ----------
    x_1 : array_like
        First sample data; shape convention (n, d)
    x_2: array_like, optional
        Second sample data; shape convention (n, d)
        If provided - performs the two-sample q selection.
        If None - performs the one-sample q selection. Defaults to None.
    low_bound : float, optional
        Smallest q to test. Defaults to 0.5.
    high_bound : float, optional
        Largest q to test. Defaults to 1.0.
    granularity : float, optional
        The increments in which to test the q's. Defaults to 0.01
    equal_var: boolean, optional
        If True - makes the equal population variance assumptions.
        If False - estimates them independently.
        Only matters when the second sample is provided.
        Defaults to True.

    Returns
    -------
    q_estimated: float
        The q that minimizes the empirical variance.
    """
    # Decisde whether we are doing one or two-sample
    if x_2 is None:
        two_sample = False
    else:
        two_sample = True
    # Ensure the dimensions
    if two_sample:
        x_1 = np.asarray(x_1)
        x_2 = np.asarray(x_2)
        if (x_1.ndim > 1) or (x_2.ndim > 1):
            raise ValueError('Sample x_1 and x_2 must be one-dimensional')
    else:
        x_1 = np.asarray(x_1)
        if x_1.ndim > 1:
            raise ValueError('Sample x_1 must be one-dimensional')
    # Initialize the iteratable arrays
    qs = np.arange(low_bound, high_bound, granularity)
    assymptotic_variances = np.zeros(qs.shape)
    # Compute the empirical variance for each q option
    for i, q in enumerate(qs):
        if two_sample:
            if equal_var:
                u_1, u_2, var = _mlqe_normal_2samp_equal_var(x_1, x_2, q=q)
                var_1 = var_2 = var
            else:
                u_1, var_1 = _mlqe_normal(x_1, q=q)
                u_2, var_2 = _mlqe_normal(x_2, q=q)
            ass_var_1 = _lqrt_empirical_variance(x_1, u_1, var_1, q)
            ass_var_2 = _lqrt_empirical_variance(x_2, u_2, var_2, q)
            assymptotic_variances[i] = ass_var_1 + ass_var_2
        else:
            u_1, var_1 = _mlqe_normal(x_1, q=q)
            ass_var_1 = _lqrt_empirical_variance(x_1, u_1, var_1, q)
            assymptotic_variances[i] = ass_var_1
    q_estimated = qs[np.argmin(assymptotic_variances)]
    return q_estimated


def _lqr_test_statistic_1samp(x, u, q=1):
    """
    Compute the Lq-likelihood-ratio-like test statistic for ONE group of
    scores (can also be used for two related samples if x = x1 - x2.)

    Parameters
    ----------
    x : array_like
        Data; must be one-dimensional.
    u : float
        expected value in null hypothesis.
    q : float, optional
        Parameter of the Lq-Likelihood.
        Defaults to 1

    Returns
    -------
    test_statistic: float
        LqRT statistic

    """
    # Ensure Dimensions
    x = np.asarray(x)
    if x.ndim > 1:
        raise ValueError('Sample x must be one-dimensional.')
    # parameters under the null hypothesis
    u_h0, cov_h0 = _mlqe_normal(x, q=q, true_u=u)
    # Lq-likelihood under the null hypothesis
    u_h1, cov_h1 = _mlqe_normal(x, q=q)
    # parameters under the alternative hypothesis
    lql_null = _lql_normal(x, u_h0, cov_h0, q)
    # Lq-likelihood under the alternative hypothesis
    lql_alt = _lql_normal(x, u_h1, cov_h1, q)
    # test statistic
    test_statistic = - 2 * lql_null + 2 * lql_alt
    return test_statistic


def _lqr_test_statistic_ind_equal_var(x_1, x_2, q=1):
    """
    Compute the Lq-likelihood-ratio-like test statistic for two independent
    samples with the equal variance assumption.

    Parameters
    ----------
    x_1 : array_like
        First sample data; must be one-dimensional.
    x_2 : array_like
        Second sample data; must be one-dimensional.
    q : float, optional
        Parameter of the Lq-Likelihood.
        Defaults to 1.

    Returns
    -------
    test_statistic: float
        LqRT statistic

    """
    # Ensure Dimensions
    x_1 = np.asarray(x_1)
    x_2 = np.asarray(x_2)
    if x_1.ndim > 1 or x_2.ndim > 1:
        raise ValueError('Samples x_1 and x_2 must be one-dimensional.')
    # parameters under the null_hypothesis
    x_combined = np.concatenate([x_1, x_2])
    u_h0, var_h0 = _mlqe_normal(x_combined, q=q)
    # Lq-likelihood under the null hypothesis
    lql_null = _lql_normal(x_combined, u_h0, var_h0, q)
    # parameters under the alternative hypothesis
    u_h1_1, u_h1_2, cov_h1 = _mlqe_normal_2samp_equal_var(x_1, x_2, q=q)
    # Lq-likelihood under the alternative hypothesis
    lql_alt = _lql_normal(x_1, u_h1_1, cov_h1, q) + \
        _lql_normal(x_2, u_h1_2, cov_h1, q)
    # test statistic
    test_statistic = - 2 * lql_null + 2 * lql_alt
    return test_statistic


def _lqr_test_statistic_ind_unequal_var(x_1, x_2, q=1):
    """
    Compute the Lq-likelihood-ratio-like test statistic for two independent
    samples with no equal variance assumption.

    Parameters
    ----------
    x_1 : array_like
        First sample data; must be one-dimensional.
    x_2 : array_like
        Second sample data; must be one-dimensional.
    q : float, optional
        Parameter of the Lq-Likelihood.

    Returns
    -------
    test_statistic: float
        LqRT statistic

    """
    # Ensure Dimensions
    x_1 = np.asarray(x_1)
    x_2 = np.asarray(x_2)
    if x_1.ndim > 1 or x_2.ndim > 2:
        raise ValueError('Samples x_1 and x_2 must be one-dimensional.')
    # parmaeters under the null hypothesis
    u_h0, cov_h0_1, cov_h0_2 = _mlqe_normal_2samp_equal_mean(x_1, x_2, q=q)
    # Lq-likelihood under the null hypothesis
    lql_null = _lql_normal(x_1, u_h0, cov_h0_1, q) + \
        _lql_normal(x_2, u_h0, cov_h0_2, q)
    # parameters under the alternative hypthesis
    u_h1_1, cov_h1_1 = _mlqe_normal(x_1, q=q)
    u_h1_2, cov_h1_2 = _mlqe_normal(x_2, q=q)
    # Lq-likelihood  under the alterantive hypothesis
    lql_alt = _lql_normal(x_1, u_h1_1, cov_h1_1, q) + \
        _lql_normal(x_2, u_h1_2, cov_h1_2, q)
    # test statistic
    test_statistic = - 2 * lql_null + 2 * lql_alt
    return test_statistic


def _critical_values_1samp(x, u, q, bootstrap=100):
    """
    Bootstraps the critical values distribution for the one-sample
    Lq-likelihood-ratio-type test using the bootstrap method provided in
    Qin and Priebe (2017).

    Parameters
    ----------
    x : array_like
        Data; must be one-dimensional.
    u : float
        expected value in null hypothesis.
    q : float, optional
        Parameter of the Lq-Likelihood.
        Defaults to 1
    bootstrap: int, optional
        number of samples to boostrap in order to estimate the Critical Value

    Returns
    -------
    D_qs: ndarray
        Bootstrapped vector of the sample of the test statistic under the null.

    """
    # Ensure Dimensions
    x = np.asarray(x)
    if x.ndim > 1:
        raise ValueError('Sample x must be one-dimensional.')
    n = len(x)

    # estimate the mean
    initial_est_mean, _ = _mlqe_normal(x, q=q)
    # shift the sample
    x_shifted = x - initial_est_mean + u
    # bootstrap
    D_qs = np.zeros(bootstrap)
    for i in range(bootstrap):
        # bootstraped sample
        idx_boot = np.random.randint(0, n, n)
        x_boot = x_shifted[idx_boot]
        # bootstrapped test statistic
        D_qs[i] = _lqr_test_statistic_1samp(x_boot, u, q)
    return D_qs


def _critical_values_ind(x_1, x_2, q, bootstrap=100, equal_var=True):
    """
    Bootstraps the critical values distribution for the two-sample
    Lq-likelihood-ratio-type test using the bootstrap method.

    Parameters
    ----------
    x_1 : array_like
        First sample data; must be one-dimensional.
    x_2 : array_like
        Second sample data; must be one-dimensional.
    q : float
        Parameter of the Lq-Likelihood.
    bootstrap: int, optional
        number of samples to boostrap in order to estimate the Critical Value
    equal_var: boolean, optional
        If True - makes the equal population variance assumptions.
        If False - estimates them independently.
        Defaults to True.

    Returns
    -------
    D_qs: ndarray
        Bootstrapped vector of the sample of the test statistic under the null.

    """
    # Ensure Dimensions
    x_1 = np.asarray(x_1)
    x_2 = np.asarray(x_2)
    if x_1.ndim > 1 or x_2.ndim > 1:
        raise ValueError('Samples x_1 and x_2 must be one-dimensional.')
    n_1 = len(x_1)
    n_2 = len(x_2)
    # estimate the means
    if equal_var:
        initial_est_mean_1, initial_est_mean_2, _ = _mlqe_normal_2samp_equal_var(x_1, x_2, q=q)
    else:
        initial_est_mean_1, _ = _mlqe_normal(x_1, q=q)
        initial_est_mean_2, _ = _mlqe_normal(x_2, q=q)
    # shift the samples to zero mean
    x_shifted_1 = x_1 - initial_est_mean_1
    x_shifted_2 = x_2 - initial_est_mean_2
    # bootstrap
    D_qs = np.zeros(bootstrap)
    for i in range(bootstrap):
        # bootstraped samples
        idx_1_boot = np.random.randint(0, n_1, n_1)
        x_1_boot = x_shifted_1[idx_1_boot]
        idx_2_boot = np.random.randint(0, n_2, n_2)
        x_2_boot = x_shifted_2[idx_2_boot]
        #  bootstrapped test statistic
        if equal_var:
            D_qs[i] = _lqr_test_statistic_ind_equal_var(x_1_boot, x_2_boot, q)
        else:
            D_qs[i] = _lqr_test_statistic_ind_unequal_var(x_1_boot,
                                                          x_2_boot, q)
    return D_qs
