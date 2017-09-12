import os
import signal
from multiprocessing import Pool

import numpy as np
import pandas as pd
import scipy.optimize
from sklearn.base import BaseEstimator, RegressorMixin
import matplotlib.pyplot as plt

from .information import compute_ic


def add_constant(x, prepend=True):
    stacked = np.column_stack((x, np.ones((x.shape[0], 1))))
    return np.roll(stacked, 1, 1) if prepend else stacked


class NNLS(BaseEstimator, RegressorMixin):

    def __init__(self, fit_intercept=True):
        self.coef_ = None
        self.residues_ = None
        self.fit_intercept = fit_intercept
        self.intercept_ = None

    def fit(self, x_orig, y):
        x = add_constant(x_orig) if self.fit_intercept else x_orig
        coef, resid = scipy.optimize.nnls(x, y)
        if self.fit_intercept:
            self.intercept_ = coef[0]
            self.coef_ = coef[1:]
        else:
            self.coef_ = coef
        self.residues_ = resid

    def predict(self, x):
        prod = np.dot(x, self.coef_)
        if self.fit_intercept:
            prod += self.intercept_
        return prod


def linear_combine(features, target):
    """
    Could use a regular linear regression:
     from sklearn.linear_model import LinearRegression
     lr = LinearRegression()
    But I think the coefficients should be constrained to be non-negative
    because these are positively correlated features,
    and our combination strategies are additive in nature.
    NNLS effectively makes the combo a weighted average,
    with the weighting that best matches the target
    """
    lr = NNLS()
    lr.fit(features, target)
    return pd.Series(lr.predict(features), index=features.index)


def combine_features(features, target, are_binary, mode='linear'):
    if mode == 'auto':
        """
        Detect and max binary features, then linear combine this max along with the remaining features
        """
        bin_max = features.T[are_binary].max(axis=0)
        conc = pd.concat([features.T[~are_binary].T, bin_max], axis=1)
        return linear_combine(conc, target)
    if mode == 'linear':
        return linear_combine(features, target)
    elif mode == 'max':
        return features.max(axis=0)
    elif mode == 'mean':
        return features.mean(axis=0)
    else:
        raise ValueError('{} not a supported feature combination mode'.format(mode))


def is_binary(series):
    unique_values = set(series)
    if len(unique_values) == 2:
        return True
    return False


def compute_cic(args):
    x, y, z = args
    return compute_ic(x, y, z=z)


def compute_cics(target, feature_df, seed=None, parallel=True):
    if parallel:
        nprocs = os.cpu_count()
        pool = Pool(processes=nprocs)

        # so spawned processes get cleaned up on interrupt signal
        def sigint_handler(signum, frame):
            pool.close()
            pool.join()
        signal.signal(signal.SIGINT, sigint_handler)

        results = pool.map(compute_cic, [(target, feature_df.loc[:, feature], seed) for feature in feature_df.columns])
        ics = pd.Series(results, index=feature_df.columns)
        pool.close()
        pool.join()
    else:
        ics = pd.Series(index=feature_df.columns)
        for feature in feature_df.columns:
            ics.loc[feature] = compute_ic(target, feature_df.loc[:, feature], z=seed)
    return ics


def plot_matches(t, features, selected):
    t = t.sort_values(ascending=False)
    to_show = pd.concat([t, features.loc[t.index, selected]], axis=1).T
    plt.matshow(to_show)
    n, m = to_show.shape
    plt.tick_params(
        axis='x',
        which='both',
        bottom='off',
        top='off',
        labelbottom='off')
    plt.xticks([])
    plt.yticks(range(n), to_show.index, **{'fontsize':20})
    plt.ylabel("features", fontsize=20)
    plt.xlabel('samples', fontsize=20)


def revealer(target, features_df, seeds=None, max_iter=5, combine='auto', parallel=True):
    """
    REVEALER is a greedy iterative search for features matching a target.
    At each iteration, previously selected features are combined to a summary feature.
    The next feature selected has the highest conditional information coefficient (CIC),
    i.e. the highest (signed) mutual information with the target variable conditioned
    on the summary feature.

    Parameters
    ----------
    target : pandas Series, (n_samples,)
    features_df : pandas DataFrame, (n_samples, n_features)
    seeds : None or list, optional (default = None)
        names of seed features; must be columns in the features DataFrame
    max_iter : int, optional (default = 5)
        maximum number of iterations
    combine : str, optional (default = 'auto')
        Strategy for combining selected features. Possible values:
        - 'max': maximum, suitable for binary features
        - 'linear': a linear combination of the features is chosen that
            best matches the target. Suitable for continuous features
        - 'auto' : 'max' is taken of binary features, then 'linear' is
            used to combine this max with the remaining features
        - 'mean': the mean of the features
    parallel : boolean, optional (default = True)
        Whether to use multiple processes.


    Returns
    -------
    selected_features: list of features in order of selection


    Notes
    -----
    The algorithm is described in
    Kim et al. 2015 Characterizing genomic alterations in cancer by complementary functional associations
    https://www.ncbi.nlm.nih.gov/pubmed/27088724
    """
    selected_features = [] if seeds is None else seeds
    iter_count = 0
    n_samples, n_features = features_df.shape
    max_iter = min(max_iter, n_features)
    prev_summary_ic = -1
    are_binary = pd.Series([is_binary(features_df.loc[:, feature]) for feature in features_df.columns],
                           index=features_df.columns)
    """
    Since the bandwidth is selected for each individual variable, not jointly, we could save some unnecessary
     recomputation by pre-selecting all the bandwidths here and passing them to the IC computation.
    This is not the main bottleneck; the savings may not be large. But it would make sense to try it.
    """
    while iter_count < max_iter:
        if len(selected_features) > 0:
            summary_feature = combine_features(features_df.loc[:, selected_features], target,
                                               are_binary.loc[selected_features],
                                               mode=combine)
            summary_ic = compute_ic(target, summary_feature)
            if summary_ic < prev_summary_ic:
                selected_features = selected_features[:-1]
                break
            prev_summary_ic = summary_ic
            print('Summary IC: {:.3}'.format(summary_ic))
        else:
            summary_feature = None
        features_left = features_df.drop(selected_features, axis=1)
        print('iter {}'.format(iter_count + 1))
        sorted_cics = compute_cics(target, features_left, summary_feature, parallel=parallel).sort_values(ascending=False)
        """
        The NMF clustering of the top N features would go here.
        Also the FDR/pvalue computation with permutations for the top features.
        However, neither affect the greedy feature selection in any way.
        And Pablo hinted that some of that was included mainly because of
         reviewers' requests and may not be important functionality now.
        """
        best_feature = sorted_cics.index[0]
        selected_features.append(best_feature)
        iter_count += 1
    return selected_features
