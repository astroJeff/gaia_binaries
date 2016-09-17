import numpy as np
from scipy.stats import gaussian_kde


plx_kde = None


def set_plx_kde(t, bandwidth=None):
    """ Set the plx_kde

    Parameters
    ----------
    t : ndarray float
        Catalog of parallax measures (units: mas)
    bandwidth : float
        Bandwidth for gaussian_kde (optional, 0.01 recommended)
    """



    global plx_kde
    if plx_kde is None:
        # We are only going to allow parallaxes above some minimum value
        if bandwidth is None:
            plx_kde = gaussian_kde(t['plx'][t['plx']>0.0])
        else:
            plx_kde = gaussian_kde(t['plx'][t['plx']>0.0], bw_method=bandwidth)


def get_plx_prior(plx):
    """ Obtain the parallax priors from the KDE

    Parameters
    ----------
    plx : float
        Parallax to get the prior for (units: mas)

    Returns
    -------
    prior_plx: float
        Prior probability for astrometric parallax
    """

    global plx_kde
    if plx_kde is None:
        print "You must set the parallax KDE first"
        return

    return plx_kde.evaluate((plx))
