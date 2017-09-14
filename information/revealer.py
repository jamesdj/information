import os
import signal
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import pandas as pd
import scipy.optimize
from sklearn.base import BaseEstimator, RegressorMixin
import matplotlib.pyplot as plt

from .information import compute_ic, keep_nonnan_overlap, compute_bandwidth, have_rpy


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
        summary_feature = linear_combine(conc, target)
    elif mode == 'linear':
        summary_feature = linear_combine(features, target)
    elif mode == 'max':
        summary_feature = features.max(axis=0)
    elif mode == 'mean':
        summary_feature = features.mean(axis=0)
    else:
        raise ValueError('{} not a supported feature combination mode'.format(mode))
    summary_feature.name = 'summary'
    return summary_feature


def is_binary(series):
    unique_values = set(series)
    if len(unique_values) == 2:
        return True
    return False


def compute_cic(args):
    x, y, z, bandwidths = args
    if bandwidths is not None:
        bws = [bandwidths[x.name], bandwidths[y.name]]
        if z is not None:
            bws.append(bandwidths[z.name])
    return compute_ic(x, y, z=z, bandwidths=bws)


def compute_cics(target, feature_df, seed=None, bandwidths=None, parallel=True):
    if parallel:
        nprocs = os.cpu_count()
        pool = Pool(processes=nprocs)

        # so spawned processes get cleaned up on interrupt signal
        def sigint_handler(signum, frame):
            pool.close()
            pool.join()
        signal.signal(signal.SIGINT, sigint_handler)

        results = pool.map(compute_cic, [(target, feature_df.loc[:, feature], seed, bandwidths) for feature in feature_df.columns])
        ics = pd.Series(results, index=feature_df.columns)
        pool.close()
        pool.join()
    else:
        ics = pd.Series(index=feature_df.columns)
        for feature in feature_df.columns:
            ics.loc[feature] = compute_ic(target, feature_df.loc[:, feature], z=seed)
    return ics


def compute_single_bandwidth(profile):
    return compute_bandwidth(keep_nonnan_overlap([profile.values])[0], rless=~have_rpy)

########################################################################################################################

# Feature preprocessing and normalization


def scale_features_between_zero_and_one(df):
    dfc = df.copy()
    dfc = dfc.subtract(dfc.min(axis=0))
    dfc = dfc.divide(dfc.max(axis=0))
    return dfc


def consolidate_identical_features(df):
    values_to_names = defaultdict(list)
    for col in df.columns:
        col_vals = df.loc[:, col]
        values_to_names[tuple(col_vals)].append(col)
    merged = {}
    for values, names in values_to_names.items():
        merged['|'.join(names)] = pd.Series(values, index=df.index)
    return pd.DataFrame(merged, index=df.index)

########################################################################################################################

# Visualization


def plot_matches(t, features, selected, out_file=None):
    t = t.sort_values(ascending=False)
    to_show = pd.concat([t, features.loc[t.index, selected]], axis=1)
    to_show = scale_features_between_zero_and_one(to_show).T
    plt.matshow(to_show)
    n, m = to_show.shape
    plt.tick_params(
        axis='x',
        which='both',
        bottom='off',
        top='off',
        labelbottom='off')
    plt.xticks([])
    plt.yticks(range(n), to_show.index, **{'fontsize': 20})
    plt.ylabel("features", fontsize=20)
    plt.xlabel('samples', fontsize=20)
    fig = plt.gcf()
    plt.show()
    if out_file is not None:
        plt.savefig(out_file)
    return fig

from matplotlib.backends.backend_pdf import PdfPages


def write_figures_to_pdf(figures, pdf_file):
    """
    Saves each figure to its own page in the pdf.
    """
    pdf = PdfPages(pdf_file)
    for fig in figures:
        pdf.savefig(fig, bbox_inches='tight')
    pdf.close()

########################################################################################################################


def revealer(target, features_df, seeds=None, max_iter=5, combine='auto', parallel=True,
             exclude_threshold=0.0, precompute_bandwidths=True):
    """
    REVEALER is a greedy iterative search for features matching a target.
    At each iteration, previously selected features are combined to a summary feature.
    The next feature selected has the highest conditional information coefficient (CIC),
    i.e. the highest (signed) mutual information with the target variable, conditioned
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
    exclude_threshold : float, optional (default = 0.0)
        Minimum IC or CIC for a feature to be included in a subsequent iteration.
         e.g. features that have negative IC w.r.t the target are highly unlikely
         to be selected later
    precompute_bandwidths : boolean, optional (default = True)
        Whether to precompute bandwidths. Saves some computation.
        May not be advisable if data has missing values.


    Returns
    -------
    selected_features: list of features in order of selection


    Notes
    -----
    The algorithm is described in
    Kim et al. 2015 Characterizing genomic alterations in cancer by complementary functional associations
    https://www.ncbi.nlm.nih.gov/pubmed/27088724
    """
    features_df = scale_features_between_zero_and_one(features_df)
    features_df = consolidate_identical_features(features_df)
    n_samples, n_features = features_df.shape
    are_binary = pd.Series([is_binary(features_df.loc[:, feature]) for feature in features_df.columns],
                           index=features_df.columns)

    """
    Bandwidth computation is not a main bottleneck; the savings are not large.
    It may or may not be problematic when the data has missing entries, because when two or three features are
     considered together for IC/CIC, only the samples at which all 2 or 3 are nonnan are considered, whereas
     for a precomputed bandwidth all nonnan samples for each individual feature are considered.
    """

    if precompute_bandwidths:
        bandwidths = {}
        for col in features_df.columns:
            bandwidths[col] = compute_single_bandwidth(features_df.loc[:, col])
        bandwidths[target.name] = compute_single_bandwidth(target)
    else:
        bandwidths = None

    """
    Missing functionality:
    - consolidation of very similar features
    - NMF clustering and FDR/p-values for top matches at each iteration
        - Neither affects the greedy feature selection.
        - Pablo hinted that some of that was included mainly because of reviewers' requests
           and may not be important functionality now. We should talk with him and decide details.
        - If we do NMF, I suggest using BIC instead of the cophenetic correlation to select k. It requires fitting
           only a single model at each k, and so is faster. Code for this below.
    """
    selected_features = [] if seeds is None else seeds
    excluded_features = []
    max_iter = min(max_iter, n_features)
    iter_count = 0
    prev_summary_ic = -1

    while iter_count < max_iter:
        if len(selected_features) > 0:
            summary_feature = combine_features(features_df.loc[:, selected_features],
                                               target,
                                               are_binary.loc[selected_features],
                                               mode=combine)
            if precompute_bandwidths:
                bandwidths[summary_feature.name] = compute_single_bandwidth(summary_feature)
            summary_ic = compute_ic(target, summary_feature)
            if summary_ic < prev_summary_ic:
                selected_features = selected_features[:-1]
                break
            prev_summary_ic = summary_ic
            print('Summary IC: {:.3}'.format(summary_ic))
        else:
            summary_feature = None
        features_left = features_df.drop(selected_features + excluded_features, axis=1)
        if features_left.shape[1] == 0:
            break
        print('iter {}'.format(iter_count + 1))
        sorted_cics = compute_cics(target,
                                   features_left,
                                   summary_feature,
                                   bandwidths=bandwidths,
                                   parallel=parallel).sort_values(ascending=False)
        """
        The NMF clustering and FDR computation for the top N features would go here.
        """
        best_feature = sorted_cics.index[0]
        selected_features.append(best_feature)
        excluded_features.extend(sorted_cics[sorted_cics < exclude_threshold].index)
        iter_count += 1
    return selected_features

########################################################################################################################

# NMF-related code

from sklearn.decomposition import NMF


def compute_log_likelihood(x, nmf):
    """
    This is based only on the errors. It assumes they have a zero-mean Gaussian distribution.
    Regular sklearn NMF does not estimate an error variance as e.g. a Bayesian version would,
     but we can just use as a stand-in the MLE estimate, the variance of the observed error.
    """
    nmf.bases_ = nmf.transform(x)
    nmf.variance_ = np.var(x - np.dot(nmf.bases_, nmf.components_))
    w, h, mu2 = [nmf.bases_,
                 nmf.components_,
                 nmf.variance_]
    x_model = np.dot(w, h)
    log_likelihood = scipy.stats.norm.logpdf(x, loc=x_model, scale=np.sqrt(mu2)).sum()
    return log_likelihood


def count_free_nmf_params(nmf):
    """
    The attribute nmf.bases_ is not defined in sklearn NMF normally. However, in bic()
     compute_log_likelihood() is called first, and it is added as an attribute there
    """
    return np.count_nonzero(nmf.bases_) + np.count_nonzero(nmf.components_) + 1


def bic(x, nmf):
    """
    In some formulations, the likelihood term is first, so it is maximized rather than minimized.
    I follow the convention in sklearn, where BIC is minimized.
    """
    log_likelihood = compute_log_likelihood(x, nmf)
    n_free_params = count_free_nmf_params(nmf)
    n, m = x.shape
    n_samples = n * m
    return np.log(n_samples) * n_free_params - 2 * log_likelihood


def select_k_by_bic(x, k_range, nmfs=None):
    # Pablo's code tries k from 2 to 10
    if nmfs is None:
        nmfs = [NMF(n_components=k).fit(x) for k in k_range]
    bics = [bic(x, nmf) for nmf in nmfs]
    min_bic_idx = np.argmin(bics)
    best_k = k_range[min_bic_idx]
    best_nmf = nmfs[min_bic_idx]
    return best_k, best_nmf