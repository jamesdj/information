import readline # not directly used, but avoids an import error in rpy2
import numpy as np
from scipy.stats import pearsonr
from statsmodels.nonparametric.kernel_density import KDEMultivariate
import scipy.optimize
from sklearn.model_selection import KFold

"""
R's bandwidth selection seems to still be better and faster
 than any alternative we've found so far in Python. This code
 uses it via rpy2 if available, and falls back on a Python
 alternative
"""

def module_exists(module_name):
    try:
        __import__(module_name)
    except ImportError:
        return False
    else:
        return True

have_rpy = module_exists('rpy2')

if have_rpy:
    # For bcv():
    # print("rpy2 found!")
    import rpy2.robjects as ro
    from rpy2.robjects.numpy2ri import numpy2ri

    ro.conversion.py2ri = numpy2ri
    from rpy2.robjects.packages import importr

    mass = importr("MASS")


def rbcv(x):
    """
    :param x: array-like, (n_samples,)
    :return: float, bandwidth
    """
    bandwidth = np.array(mass.bcv(x))[0]
    #print("rbcv")
    return bandwidth


def kde_xval(bw, args):
    sample = args['x']
    n_folds = args['n_folds']
    var_type = args['var_type']
    losses = []
    for train, test in KFold(n_splits=n_folds).split(sample):
        kde = KDEMultivariate(sample[train], var_type=var_type, bw=[bw])
        pdf = kde.pdf(sample[test])
        logpdf = np.log(pdf)
        logpdfsum = logpdf.sum()
        losses.append(-1 * logpdfsum)
    return np.mean(losses)


def xval_select_bw(sample, n_folds=10, var_type='c', bounds=[1E-5, 5]):

    result = scipy.optimize.minimize_scalar(kde_xval, bounds=bounds,
                             args={'x': sample, 'n_folds':n_folds, 'var_type': var_type},
                             method='bounded', options={'maxiter': 500, 'xatol': 1e-05})
    return result.x


def compute_bandwidth(x, var_type='c', rless=False):
    if not have_rpy or rless:
        bw = xval_select_bw(x, var_type=var_type)
        # delta = 'cv_ml'
    else:
        bw = rbcv(x) / 4  # scaling factor to make equivalent because different conventions are used
    return bw


def compute_unspecified_bandwidths(variables, bandwidths, var_types=None, rless=False):
    n_vars = len(variables)
    if var_types is None:
        var_types = ''.join(['c' for _ in range(n_vars)])
    if bandwidths is None:
        bandwidths = [None] * len(variables)
    delta = []
    for i, bw in enumerate(bandwidths):
        if bw is None:
            bw = compute_bandwidth(variables[i], var_type=var_types[i], rless=rless)
        delta.append(bw)
    delta = np.array(delta).reshape((n_vars,))
    return delta


def keep_nonnan_overlap(variables):
    n = len(variables[0])
    non_nans = [np.logical_not(np.isnan(v)) for v in variables]
    overlap = [True] * n
    for non_nan in non_nans:
        overlap &= non_nan
    variables = [v[overlap] for v in variables]
    return variables


def add_jitter(variables, jitter_scale=1E-10):
    # assumes variables are all of same length
    jitters = [jitter_scale * np.random.uniform(size=len(v)) for v in variables]
    variables = [variables[i] + jitters[i] for i in range(len(variables))]
    return variables


def compute_mutual_information(x, y, z=None, n_grid=25, var_types=None,
                               bandwidth_scaling=None, bandwidths=None, rless=False):
    """
    :param x: array-like, (n_samples,)
    :param y: array-like, (n_samples,)
    :param z: array-like, (n_samples,), optional, variable on which to condition
    :param n_grid: int, number of grid points at which to evaluate kernel density
    :param var_types: three-character string of 'c' (continuous), 'u' (unordered discrete) or 'o' (ordered discrete)
    :param bandwidth_scaling: float
    :return: float, information coefficient
    """
    n = len(x)
    variables = [np.array(x, dtype=float), np.array(y, dtype=float)]
    if z is not None:
        variables.append(np.array(z, dtype=float))
    for v in variables[1:]:
        if len(v) != n:
            raise ValueError("Input arrays have different lengths")
    n_vars = len(variables)
    if var_types is None:
        var_types = ''.join(['c' for _ in range(n_vars)])
        # Todo: guess variable types
    if len(var_types) != n_vars:
        raise ValueError("Number of specified variable types does not match number of variables")
    #print([len(v) for v in variables])
    variables = keep_nonnan_overlap(variables)
    #print([len(v) for v in variables])
    n_overlap = len(variables[0])
    if n_overlap < 2:
        return 0
    variables = add_jitter(variables)
    grids = [np.linspace(v.min(), v.max(), n_grid) for v in variables]
    mesh_grids = np.meshgrid(*grids)
    grid_shape = tuple([n_grid] * n_vars)
    grid = np.vstack([mesh_grid.flatten() for mesh_grid in mesh_grids])
    delta = compute_unspecified_bandwidths(variables, bandwidths, var_types)
    if bandwidth_scaling is not None:
        delta *= bandwidth_scaling
    kde = KDEMultivariate(variables, bw=delta, var_type=var_types)
    p_joint = kde.pdf(grid).reshape(grid_shape) + np.finfo(float).eps  # THIS IS THE HOT SPOT. Get faster method
    ds = [grid[1] - grid[0] for grid in grids]
    ds_prod = np.prod(ds)
    p_joint /= (p_joint.sum() * ds_prod)
    h_joint = - np.sum(p_joint * np.log(p_joint)) * ds_prod
    dx = ds[0]
    dy = ds[1]
    if z is None:
        dx = ds[0]
        dy = ds[1]
        px = p_joint.sum(axis=1) * dy
        py = p_joint.sum(axis=0) * dx
        hx = -np.sum(px * np.log(px)) * dx
        hy = -np.sum(py * np.log(py)) * dy
        mi = hx + hy - h_joint
        return mi
    else:
        dz = ds[2]
        pxz = p_joint.sum(axis=1) * dy
        pyz = p_joint.sum(axis=0) * dx
        pz = p_joint.sum(axis=(0, 1)) * dx * dy
        hxz = -np.sum(pxz * np.log(pxz)) * dx * dz
        hyz = -np.sum(pyz * np.log(pyz)) * dy * dz
        hz = -np.sum(pz * np.log(pz)) * dz
        cmi = hxz + hyz - h_joint - hz
        return cmi


def compute_ic(x, y, z=None, n_grid=25, var_types=None, bandwidths=None, rless=False, raise_errors=False):
    """
    :param x: array-like, (n_samples,)
    :param y: array-like, (n_samples,)
    :param z: array-like, (n_samples,), optional, variable on which to condition
    :param n_grid: int, number of grid points at which to evaluate kernel density
    :param var_types: three-character string of 'c' (continuous), 'u' (unordered discrete) or 'o' (ordered discrete)
    """
    try:
        variables = [x, y]
        if z is not None:
            variables.append(z)
            x, y, z = keep_nonnan_overlap(variables)
        else:
            x, y = keep_nonnan_overlap(variables)
        bandwidth_scaling, ic_sign = ic_bandwidth_scaling_and_sign(x, y)
        mi = compute_mutual_information(x, y, z=z, n_grid=n_grid,
                                        var_types=var_types, bandwidths=bandwidths,
                                        bandwidth_scaling=bandwidth_scaling, rless=rless)
        ic = ic_sign * np.sqrt(1 - np.exp(- 2 * mi))
    except Exception as e:
        print(e)
        ic = 0
        if raise_errors:
            raise(e)
    return ic


def ic_bandwidth_scaling_and_sign(x, y):
    rho, p = pearsonr(x, y)
    rho2 = abs(rho)
    bandwidth_scaling = (1 + (-0.75) * rho2)
    ic_sign = np.sign(rho)
    return bandwidth_scaling, ic_sign