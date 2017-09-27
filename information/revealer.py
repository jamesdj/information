import os
import signal
from collections import defaultdict
from multiprocessing import Pool
import copy

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


def count_unique_values_in_columns(features_df):
    return pd.Series([len(set(features_df.loc[:, col])) for col in features_df.columns], index=features_df.columns)


def compute_cic(args):
    x, y, z, bandwidths = args
    if bandwidths is not None:
        bws = [bandwidths[x.name], bandwidths[y.name]]
        if z is not None:
            bws.append(bandwidths[z.name])
    else:
        bws = None
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
    return compute_bandwidth(keep_nonnan_overlap([profile.values])[0], rless=not have_rpy)

########################################################################################################################

# Feature preprocessing and normalization


def screen_features(features_df):
    ns_unique_vals = count_unique_values_in_columns(features_df)
    features_df = features_df.T[ns_unique_vals > 0].T
    features_df = scale_features_between_zero_and_one(features_df)
    features_df = consolidate_identical_features(features_df)
    return features_df


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


def add_rtexts_to_plot(ax, rtexts, x_offset):
    ymax, ymin = ax.get_ylim()
    ydiv = ymax - ymin
    ytick_ycoords = ax.get_yticks()[::-1]
    for i, label in enumerate(rtexts):
        ax.text(1 + x_offset, (ytick_ycoords[i] / ydiv) + 0.08, label, fontsize=20, **{'va':'center'}, transform=ax.transAxes)
    return ax

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


def permute_series(series):
    return pd.Series(np.random.permutation(series.values), name=series.name, index=series.index)


class Revealer:
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
    combine_mode : str, optional (default = 'auto')
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
    combine_first : boolean, optional (default = True)
        Whether to combine features and compute IC, rather than compute CIC and then combine.
        CIC is slower to compute, so this can result in a speedup
        Results should be identical if combine=='max',
        and similar but possibly not identical for other combination modes.


    Notes
    -----
    The algorithm is described in
    Kim et al. 2015 Characterizing genomic alterations in cancer by complementary functional associations
    https://www.ncbi.nlm.nih.gov/pubmed/27088724
    """

    def __init__(self,
                 target,
                 features_df,
                 seeds=None,
                 max_iter=5,
                 combine_mode='auto',
                 parallel=True,
                 exclude_threshold=0.0,
                 precompute_bandwidths=True,
                 combine_first=True):
        self.target = target
        self.features_df = screen_features(features_df).loc[target.index]
        # Todo: maybe include code for combining very similar features
        """
        May not need feature scaling inside the algorithm and could leave it in the plotting functions
         unless 1) we do NMF or 2) we think it's important that some variables could be identical after scaling
         i.e. one is a multiple of another--but I think that's a pretty remote possibility in most cases...
        """
        ns_unique_vals = count_unique_values_in_columns(features_df)
        self.n_samples, self.n_features = features_df.shape
        self.are_binary = ns_unique_vals == 2
        self.seeds = seeds
        self.max_iter = min(max_iter, self.n_features)
        if combine_mode not in 'auto linear max mean'.split():
            raise ValueError('{} not a supported feature combination mode'.format(combine_mode))
        self.combine_mode = combine_mode
        self.parallel = parallel
        self.exclude_threshold = exclude_threshold
        self.precompute_bandwidths = precompute_bandwidths
        self.combine_first = combine_first
        if self.precompute_bandwidths:
            self.bandwidths = {}
            for col in features_df.columns:
                self.bandwidths[col] = compute_single_bandwidth(features_df.loc[:, col])
            self.bandwidths[target.name] = compute_single_bandwidth(target)
        else:
            self.bandwidths = None
        """
        Bandwidth computation is not a main bottleneck; the savings are not large.
        It may or may not be problematic when the data has missing entries, because when two or three features are
         considered together for IC/CIC, only the samples at which all 2 or 3 are nonnan are considered, whereas
         for a precomputed bandwidth all nonnan samples for each individual feature are considered.
        """
        self.selected_features = None
        self.summary_ics = None
        self.cum_pvals = None

    def match(self):
        """

        """
        selected_features = [] if self.seeds is None else self.seeds
        summary_ics = []
        excluded_features = []
        iter_count = 0
        prev_summary_ic = -1

        if len(selected_features) > 0:
            summary_feature = self.combine_features(selected_features)
            if self.precompute_bandwidths:
                self.bandwidths[summary_feature.name] = compute_single_bandwidth(summary_feature)
        else:
            summary_feature = None

        while iter_count < self.max_iter:
            features_left = self.features_df.drop(selected_features + excluded_features, axis=1)
            if features_left.shape[1] == 0:
                break
            #print('iter {}'.format(iter_count + 1))

            if self.combine_first:
                combined = pd.concat([self.combine_features(selected_features + [feature])
                                      for feature in features_left.columns], axis=1)
                combined.columns = features_left.columns
                sorted_cics = compute_cics(self.target,
                                           combined,
                                           None,
                                           bandwidths=None,
                                           parallel=self.parallel).sort_values(ascending=False)
                best_feature = sorted_cics.index[0]
                summary_feature = combined.loc[:, best_feature]
                summary_feature.name = 'summary'
                summary_ic = sorted_cics.loc[best_feature]
                selected_features.append(best_feature)
            else:
                sorted_cics = compute_cics(self.target,
                                           features_left,
                                           summary_feature,
                                           bandwidths=self.bandwidths,
                                           parallel=self.parallel).sort_values(ascending=False)
                best_feature = sorted_cics.index[0]
                selected_features.append(best_feature)
                summary_feature = self.combine_features(selected_features)
                summary_ic = compute_ic(self.target, summary_feature)
            """
            The NMF clustering and/or FDR computation for the top N features would go here.
            - Neither affects the greedy feature selection.
            - Pablo hinted that some of that was included mainly because of reviewers' requests
               and may not be important functionality now. We should talk with him and decide details.
            - If we do NMF, I suggest using BIC instead of the cophenetic correlation to select k.
               It requires fitting only a single model at each k, and so is faster. Code for this below.
            """
            excluded_features.extend(sorted_cics[sorted_cics < self.exclude_threshold].index)
            if self.precompute_bandwidths:
                self.bandwidths[summary_feature.name] = compute_single_bandwidth(summary_feature)
            if summary_ic - prev_summary_ic < 10E-8:
                selected_features = self.selected_features[:-1]
                break
            else:
                summary_ics.append(summary_ic)

            prev_summary_ic = summary_ic
            #print('Summary IC: {:.3}'.format(summary_ic))
            iter_count += 1
        self.selected_features = selected_features
        self.summary_ics = summary_ics
        return self

    def combine_features(self, selected_features):
        features_subdf = self.features_df.loc[:, selected_features]
        are_binary = self.are_binary.loc[selected_features]
        if features_subdf.ndim == 1:
            summary_feature = features_subdf.copy()
        else:
            n, m = features_subdf.shape
            if m == 1:
                summary_feature = features_subdf.copy().iloc[:, 0]
            else:
                if self.combine_mode == 'auto':
                    """
                    Detect and max binary features, then linear combine this max along with the remaining features
                    """
                    n_binary = self.are_binary.sum()
                    if n_binary == 0:
                        summary_feature = linear_combine(features_subdf, self.target)
                    elif n_binary == len(are_binary):
                        summary_feature = features_subdf.max(axis=1)
                    else:
                        bin_max = features_subdf.T[are_binary].max(axis=0)
                        conc = pd.concat([features_subdf.T[~are_binary].T, bin_max], axis=1)
                        summary_feature = linear_combine(conc, self.target)
                elif self.combine_mode == 'linear':
                    summary_feature = linear_combine(features_subdf, self.target)
                elif self.combine_mode == 'max':
                    summary_feature = features_subdf.max(axis=1)
                elif self.combine_mode == 'mean':
                    summary_feature = features_subdf.mean(axis=1)
        summary_feature.name = 'summary'
        return summary_feature

    def compute_pvals(self, n_permutations=100):
        # Todo: allow both pvals at each step and cumulative pvals
        # Todo: allow user to choose which to compute
        # maybe it would be better to do individual (not cumulative) pvals in the main loop?
        """
        indiv_p_vals = pd.Series(index=range(len(self.selected_features)))
        if self.combine_first:
            for i, feature in enumerate(self.selected_features):
                so_far = self.selected_features[:i + 1]
                combined = combine_features(so_far)


                for p in range(n_permutations):
                    perm_target = permute_series(self.target)
                    p_ic = compute_ic(combined, perm_target)
        """

        n_iter = len(self.summary_ics)
        p_ics_df = pd.DataFrame(index=range(n_permutations), columns=range(n_iter))
        for p in range(n_permutations):
            perm_target = permute_series(self.target)
            perm_revealer = copy.deepcopy(self)
            perm_revealer.target = perm_target
            perm_revealer.selected_features = [] if self.seeds is None else self.seeds
            perm_revealer.summary_ics = []
            perm_revealer.fig = None
            perm_revealer.match()
            p_sel, p_ics = perm_revealer.selected_features, perm_revealer.summary_ics
            max_vals = min(len(p_ics), n_iter)
            p_ics_df.iloc[p, :max_vals] = p_ics[:max_vals]
        p_ics_df = p_ics_df.T.fillna(method='ffill').T.iloc[:, :n_iter]
        sic = pd.Series(self.summary_ics)
        self.cum_pvals = (p_ics_df > sic).sum(axis=0) / n_permutations
        return self

    def plot_matches(self,
                     out_file=None,
                     name_samples=False,
                     title=None,
                     dpi=None,
                     cmap=None,
                     ylabel='features',
                     xlabel='samples'):
        # Todo: move "features" down centered on only the matched features
        # Todo: include the summary feature(s)?
        # print(features)
        t = self.target.sort_values(ascending=False)
        to_show = pd.concat([t, self.features_df.loc[t.index, self.selected_features]], axis=1)
        to_show = scale_features_between_zero_and_one(to_show).T  # redundant now?
        fig, ax = plt.subplots()
        n_fig_rows = len(self.selected_features) + 1
        fig.set_size_inches((6, n_fig_rows / 2))
        ax.matshow(to_show, aspect='auto', cmap=cmap)
        n, m = to_show.shape
        plt.tick_params(
            axis='x',
            which='both',
            bottom='off',
            top='off',
            labelbottom='off')
        if name_samples:
            ax.set_xticks(range(len(t)))
            ax.set_xticklabels(t.index, rotation=90)
            ax.xaxis.tick_bottom()
        else:
            plt.xticks([])
        plt.yticks(range(n), to_show.index, **{'fontsize': 20})
        plt.ylabel(ylabel, fontsize=20)
        plt.xlabel(xlabel, fontsize=20)
        if title is not None:
            plt.title(title, fontsize=20)
        ax = add_rtexts_to_plot(ax,
                                ['IC'] + ['{0:.2f}'.format(ic).lstrip('0') for ic in self.summary_ics],
                                x_offset=0.03)
        ax.grid(False)
        if self.cum_pvals is not None:
            add_rtexts_to_plot(ax,
                               ['p'] + ['{0:.2f}'.format(p).lstrip('0') for p in self.cum_pvals],
                               x_offset=0.15)
        if out_file is not None:
            plt.savefig(out_file, bbox_inches='tight', dpi=dpi)
        return fig, ax

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