"""
Preprocessing functions for time series.

All functions in this module should take X matrices with samples x
features
"""
# Authors: Alexandre Abraham, Gael Varoquaux, Philippe Gervais
# License: simplified BSD

import distutils.version

import numpy as np

np_version = distutils.version.LooseVersion(np.version.short_version).version


def _standardize(signals, detrend=False, normalize=True):
    """ Center and norm a given signal (time is along first axis)

    Parameters
    ==========
    signals: numpy.ndarray
        Timeseries to standardize

    detrend: bool
        if detrending of timeseries is requested

    normalize: bool
        if True, shift timeseries to zero mean value and scale
        to unit energy (sum of squares).

    Returns
    =======
    std_signals: numpy.ndarray
        copy of signals, normalized.
    """
    if detrend:
        signals = _detrend(signals, inplace=False)
    else:
        signals = signals.copy()

    if normalize:
        # remove mean if not already detrended
        if not detrend:
            signals -= signals.mean(axis=0)

        std = np.sqrt((signals ** 2).sum(axis=0))
        std[std < np.finfo(np.float).eps] = 1.  # avoid numerical problems
        signals /= std
    return signals


def _detrend(signals, inplace=False, type="linear"):
    """Detrend columns of input array.

    Signals are supposed to be columns of `signals`.
    This function is significantly faster than scipy.signal.detrend.

    Parameters
    ==========
    signals: numpy.ndarray
        This parameter must be two-dimensional.
        Signals to detrend. A signal is a column.

    inplace: bool
        Tells if the computation must be made inplace or not (default
        False).

    type: str
        Detrending type ("linear" or "constant").
        See also scipy.signal.detrend.

    Returns
    =======
    detrended_signals: numpy.ndarray
        Detrended signals. The shape is that of 'signals'.
    """
    if not inplace:
        signals = signals.copy()

    signals -= np.mean(signals, axis=0)
    if type == "linear":
        regressor = np.arange(signals.shape[0]).astype(np.float)
        regressor -= regressor.mean()
        regressor /= np.sqrt((regressor ** 2).sum())
        signals -= np.dot(regressor, signals) * regressor[:, np.newaxis]
    return signals


def clean(signals, detrend=True, standardize=True):
    """Improve SNR on masked fMRI signals.

       This function can do several things on the input signals, in
       the following order:
       - detrend
       - standardize
       - remove confounds
       - low- and high-pass filter

       Low-pass filtering improves specificity.

       High-pass filtering should be kept small, to keep some
       sensitivity.

       Filtering is only meaningful on evenly-sampled signals.

       Parameters
       ==========
       signals: numpy.ndarray
           Timeseries. Must have shape (instant number, features number).
           This array is not modified.

       detrend: bool
           If detrending should be applied on timeseries (before
           confound removal)

       standardize: bool
           If True, returned signals are set to unit variance.

       Returns
       =======
       cleaned_signals: numpy.ndarray
           Input signals, cleaned. Same shape as `signals`.

       Notes
       =====
       Confounds removal is based on a projection on the orthogonal
       of the signal space. See `Friston, K. J., A. P. Holmes,
       K. J. Worsley, J.-P. Poline, C. D. Frith, et R. S. J. Frackowiak.
       "Statistical Parametric Maps in Functional Imaging: A General
       Linear Approach". Human Brain Mapping 2, no 4 (1994): 189-210.
       <http://dx.doi.org/10.1002/hbm.460020402>`_
    """

    # Standardize / detrend
    normalize = False
    signals = _standardize(signals, normalize=normalize, detrend=detrend)

    if standardize:
        signals = _standardize(signals, normalize=True, detrend=False)
        signals *= np.sqrt(signals.shape[0])  # for unit variance

    return signals
