import numpy as np
from scipy.stats import gaussian_kde
from astroML.density_estimation import bayesian_blocks

plx_kde = None
plx_hist_blocks = None
plx_bins_blocks = None


def set_plx_kde(t, bandwidth=None, method='kde'):
    """ Set the plx_kde

    Parameters
    ----------
    t : ndarray float
        Catalog of parallax measures (units: mas)
    bandwidth : float
        Bandwidth for gaussian_kde (optional, 0.01 recommended)
    method : string
        Method for density determination (options: kde, blocks)
    """


    if method is 'kde':
        global plx_kde
        if plx_kde is None:
            # We are only going to allow parallaxes above some minimum value
            if bandwidth is None:
                plx_kde = gaussian_kde(t['plx'][t['plx']>0.0])
            else:
                plx_kde = gaussian_kde(t['plx'][t['plx']>0.0], bw_method=bandwidth)
    elif method is 'blocks':
        global plx_bins_blocks
        global plx_hist_blocks

        # Set up Bayesian Blocks
        print "Calculating Bayesian Blocks..."
        nbins = np.min([len(t), 40000])
        bins = bayesian_blocks(t['plx'][t['plx']>0.0][0:nbins])
        hist, bins = np.histogram(t['plx'][t['plx']>0.0][0:nbins], bins=bins, normed=True)

        # Pad with zeros
        plx_bins_blocks = np.append(-1.0e100, bins)
        hist_pad = np.append(0.0, hist)
        plx_hist_blocks = np.append(hist_pad, 0.0)
        print "Bayesian Blocks set."

    else:
        print "You must include a valid method"
        print "Options: kde or blocks"
        return


def get_plx_prior(plx, method='kde'):
    """ Obtain the parallax priors from the KDE

    Parameters
    ----------
    plx : float
        Parallax to get the prior for (units: mas)
    method : string
        Method for density determination (options: kde, blocks)

    Returns
    -------
    prior_plx: float
        Prior probability for astrometric parallax
    """

    if method is 'kde':
        global plx_kde
        if plx_kde is None:
            print "You must set the parallax KDE first"
            return

        return plx_kde.evaluate((plx))

    if method is 'blocks':
        global plx_bins_blocks
        global plx_hist_blocks

        if plx_bins_blocks is None or plx_hist_blocks is None:
            print "You must set the Bayesian Blocks first"
            return

        x_2d, y_2d = np.meshgrid(plx, plx_bins_blocks)
        diff_array = np.array(x_2d-y_2d)
        shape = diff_array.shape
        flattened = diff_array.flatten()
        flattened[flattened<0.0] = 1.0e200
        diff_array = flattened.reshape(shape)
        idx = np.argmin(diff_array, axis=0)

        return plx_hist_blocks[idx]

        # For single values below:
        # idx = np.argmin((plx-plx_bins_blocks)[(plx-plx_bins_blocks)>0.0])
        # return plx_hist_blocks[idx]
