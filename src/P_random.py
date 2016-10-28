import numpy as np
from scipy.stats import gaussian_kde, multivariate_normal
from sklearn.neighbors import KernelDensity
from numpy.random import normal
import const as c
import parallax
import matplotlib.pyplot as plt


mu_kde = None
pos_kde = None
C1_prior_norm = None

def deg_to_rad(theta):
    """ Convert from degrees to radians """
    return np.pi * theta / 180.0

def rad_to_deg(theta):
    """ Convert from radians to degrees """
    return 180.0 * theta / np.pi

def get_theta_proj_degree(ra, dec, ra_b, dec_b):
    """ Return angular distance between two points

    Parameters
    ----------
    ra : float64
        Right ascension of first coordinate (degrees)
    dec : float64
        Declination of first coordinate (degrees)
    ra_b : float64
        Right ascension of second coordinate (degrees)
    dec_b : float64
        Declination of second coordinate (degrees)

    Returns
    -------
    theta : float64
        Angular distance (degrees)
    """

    ra1 = deg_to_rad(ra)
    dec1 = deg_to_rad(dec)
    ra2 = deg_to_rad(ra_b)
    dec2 = deg_to_rad(dec_b)

    dist = np.sqrt((ra1-ra2)**2 * np.cos(dec1)*np.cos(dec2) + (dec1-dec2)**2)

    return rad_to_deg(dist)


def get_delta_mu(mu_ra, mu_dec, mu_ra_b, mu_dec_b):
    """ Return proper motion difference between two points

    Parameters
    ----------
    mu_ra : float64
        Proper motion in RA of first object (mas/yr)
    mu_dec : float64
        Proper motion in Dec of first object (mas/yr)
    mu_ra_b : float64
        Proper motion in RA of second object (mas/yr)
    mu_dec_b : float64
        Proper motion in Dec of second object (mas/yr)

    Returns
    -------
    delta_mu : float64
        proper motion difference (mas/yr)
    """

    delta_mu = np.sqrt((mu_ra-mu_ra_b)**2 + (mu_dec-mu_dec_b)**2)

    return delta_mu




def nstars_nearby(ra, dec, radius=1.0, catalog=None):
    """ This function searches the catalog for all stars
    within some input radius.

    Arguments
    ---------
    ra, dec : float
        Search coordinate (degrees)
    radius : float
        Search radius (degrees)
    catalog : structure
        Catalog to search through

    Returns
    -------
    nstars : float
        Number of stars within given radius around input
        coordinate
    """

    if catalog is None:
        print "You must supply an input catalog"
        return


    ra_rad1 = deg_to_rad(ra)
    dec_rad1 = deg_to_rad(dec)
    ra_rad2 = deg_to_rad(catalog['ra'])
    dec_rad2 = deg_to_rad(catalog['dec'])

    dist = rad_to_deg(np.sqrt((ra_rad1-ra_rad2)**2 * np.cos(dec_rad1)*np.cos(dec_rad2) + (dec_rad1-dec_rad2)**2))

    return len(np.where(dist<radius)[0])


def get_sigma_pos(ra, dec, catalog=None, rad=5.0, method='sklearn_kde', bandwidth=None):
    """ This function calculates the local stellar density

    Parameters
    ----------
    ra, dec : float
        Coordinates to find the local stellar density (degrees)
    catalog : structure
        Catalog to search through (degrees)
    rad : float
        Search radius; empirical only (degrees)
    method : string
        Density determination method (options: scipy_kde, sklearn_kde, empirical)
    bandwidth : float
        Bandwidth for the sklearn_kde (optional)

    Returns
    -------
    sigma_star : float
        Local density of stars per square degree
    """

    global pos_kde

    # Catalog check
    if catalog is None:
        print "You must provide a catalog"
        return
    if method is 'empirical':
        # Estimate number density of stars from number of systems within 5 degrees
        sigma_star = (nstars_nearby(ra, dec, radius=rad, catalog=catalog)-1) / (4.0*np.pi* rad**2)
    elif method is 'scipy_kde':
        # Use a Gaussian KDE
        if pos_kde is None:

            if c.kde_subset:
                # Select a subset of catalog for the KDE
                ran_ids = np.arange(len(catalog))
                np.random.shuffle(ran_ids)

                # KDE needs to be in terms of (ra*cos(dec), dec) so units are deg^-2
                pos_kde = gaussian_kde((catalog['ra'][ran_ids[0:100000]] * np.cos(catalog['dec'][ran_ids[0:100000]]*np.pi/180.0), catalog['dec'][ran_ids[0:100000]]))
            else:
                pos_kde = gaussian_kde((catalog['ra'] * np.cos(catalog['dec']*np.pi/180.0), catalog['dec']))

        sigma_star = pos_kde.evaluate((ra*np.cos(dec*np.pi/180.0), dec))
    elif method is 'sklearn_kde':

        if pos_kde is None:
            kwargs = {'kernel':'tophat'}
            if bandwidth is None:
                pos_kde = KernelDensity(**kwargs)
            else:
                pos_kde = KernelDensity(bandwidth=bandwidth, **kwargs)


            if c.kde_subset:
                # Select a subset of catalog for the KDE
                ran_ids = np.arange(len(catalog))
                np.random.shuffle(ran_ids)

                pos_kde.fit( np.array([catalog['ra'][ran_ids[0:100000]] * np.cos(catalog['dec'][ran_ids[0:100000]]*np.pi/180.0), catalog['dec'][ran_ids[0:100000]]]).T )
            else:
                pos_kde.fit( np.array([catalog['ra'] * np.cos(catalog['dec']*np.pi/180.0), catalog['dec']]).T )

        values = np.array([ra*np.cos(dec*np.pi/180.0), dec]).T
        sigma_star = np.exp(pos_kde.score_samples(values))


    else:
        print "You must provide a valid method"
        print "Options: 'scipy_kde', 'sklearn_kde', or 'empirical'"
        return

    return sigma_star


def get_random_alignment_P_pos(ra1, dec1, ra2, dec2, density=None, catalog=None):
    """ This function determines the probability of two
    positions to be formed from simple random draws.
    Still need to check math

    Arguments
    ---------
    ra1, dec1 : float
        The first star's coordinates (degrees)
    ra2, dec2 : float
        The second star's coordinates (degrees)

    Returns
    -------
    P(pos) : float
        The probability, due to random alignments of randomly
        forming two stars with the given positions
    """


    # Catalog check
    if catalog is None:
        print "You must provide a catalog"
        return

    # Projected distance
    theta = get_theta_proj_degree(ra1, dec1, ra2, dec2)

    # Local stellar density, if not provided
    if density is None:
        density = get_sigma_pos(ra1, dec1, catalog=catalog)

    # P(pos)
    P_pos = 2.0*np.pi*theta * density

    return P_pos




def get_sigma_mu(mu_ra, mu_dec, catalog=None, rad=5.0, method='sklearn_kde', bandwidth=None):
    """ This function calculates the local proper
    motion density of stars per (mas/yr)^2

    Parameters
    ----------
    mu_ra, mu_dec : float
        Proper Motion likelihood to calculate (mas/yr)
    catalog : structure
        Catalog to search through (mas/yr)
    rad : float
        Search radius for calibration, empirical only (mas/yr)
    method : string
        Method to perform 2D interpolation (options:sklearn_kde, scipy_kde, empirical)
    bandwidth : float
        Bandwidth for KDE (optional; 0.5 is currently preferred)

    Returns
    -------
    sigma_mu : float
        Local density of stars (n/(mas/yr)^2)
    """

    # Catalog check
    if catalog is None:
        print "You must include a catalog."
        return

    global mu_kde

    if method is 'scipy_kde':
        # Use a Gaussian KDE
        if mu_kde is None:

            if c.kde_subset:
                # Select a subset of catalog for the KDE
                ran_ids = np.arange(len(catalog))
                np.random.shuffle(ran_ids)

                if bandwidth is None:
                    mu_kde = gaussian_kde((catalog['mu_ra'][ran_ids[0:100000]], catalog['mu_dec'][ran_ids[0:100000]]))
                else:
                    mu_kde = gaussian_kde((catalog['mu_ra'][ran_ids[0:100000]], catalog['mu_dec'][ran_ids[0:100000]]), bw_method=bandwidth)
            else:
                # Use the whole catalog
                if bandwidth is None:
                    mu_kde = gaussian_kde((catalog['mu_ra'], catalog['mu_dec']))
                else:
                    mu_kde = gaussian_kde((catalog['mu_ra'], catalog['mu_dec']), bw_method=bandwidth)

        sigma_mu = mu_kde.evaluate((mu_ra, mu_dec))

    elif method is 'sklearn_kde':
        # Use the sklearn KDE algorithm
        if mu_kde is None:

            kwargs = {'kernel':'tophat'}
            if bandwidth is None:
                mu_kde = KernelDensity(**kwargs)
            else:
                mu_kde = KernelDensity(bandwidth=bandwidth, **kwargs)

            if c.kde_subset:
                # Select a subset of catalog for the KDE
                ran_ids = np.arange(len(catalog))
                np.random.shuffle(ran_ids)

                mu_kde.fit( np.array([catalog['mu_ra'][ran_ids[0:100000]], catalog['mu_dec'][ran_ids[0:100000]]]).T )
            else:
                mu_kde.fit( np.array([catalog['mu_ra'], catalog['mu_dec']]).T )

        values = np.array([mu_ra, mu_dec]).T
        sigma_mu = np.exp(mu_kde.score_samples(values))

    elif method is 'empirical':
        n_stars_near = nstars_nearby_mu(mu_ra, mu_dec, radius=rad, catalog=catalog)-1
        sigma_mu = n_stars_near / (4.0*np.pi * rad**2)
    else:
        print "You must input an appropriate method."
        print "Options: 'scipy_kde', 'sklearn_kde', or 'empirical'"
        print "sklearn_kde is the preferred method."
        return


    # Set a minimum sigma_mu so it will never equal zero  - 1.0e-6
    if isinstance(sigma_mu, np.ndarray):
        sigma_min = np.ones((2, len(sigma_mu)))
        sigma_min[0] = sigma_mu
        sigma_min[1] = 1.0e-6
        return np.max(sigma_min, axis=0)
    else:
        return np.max(1.0e-6, sigma_mu)



def nstars_nearby_mu(mu_ra, mu_dec, radius=1.0, catalog=None):
    """ This function searches the catalog for all stars
    within some input proper motion radius.

    Arguments
    ---------
    mu_ra, mu_dec : float
        Search coordinate (mas/yr)
    radius : float
        Search radius (mas/yr)
    catalog : structure
        Catalog to search through (mas/yr)

    Returns
    -------
    nstars : float
        Number of stars within given proper motion radius around input
        coordinate
    """

    if catalog is None:
        print "You must supply an input catalog"
        return

    dist = np.sqrt((mu_ra-catalog['mu_ra'])**2 + (mu_dec-catalog['mu_dec'])**2)

    return len(np.where(dist<radius)[0])


def get_random_alignment_P_mu(mu_ra1, mu_dec1, mu_ra2, mu_dec2, delta_mu_ra_err=0.0, delta_mu_dec_err=0.0,
                                nsamples=100, density=None, catalog=None, method='kde'):
    """ This function determines the probability of two proper
    motions to be formed from simple random draws

    Arguments
    ---------
    mu_ra1, mu_dec1 : float
        The first star's proper motion (mas/yr)
    mu_ra2, mu_dec2 : float
        The second star's proper motion (mas/yr)
    delta_mu_ra_err, delta_mu_dec_err : float
        Uncertainties on delta_mu_ra and delta_mu_dec (mas/yr)
    nsamples : float
        Number of random samples to Monte Carlo approximate delta_mu_err

    Returns
    -------
    P(mu) : float
        The probability, due to random alignments of randomly
        forming two stars with the given proper motion
    """

    # Catalog check
    if catalog is None:
        print "You must include a catalog."
        return

    # Proper motion density
    if density is None:
        if method is 'empirical':
            density = get_sigma_mu(mu_ra1, mu_dec1, catalog=catalog, method='empirical')
        else:
            density = get_sigma_mu(mu_ra1, mu_dec1, catalog=catalog, method='sklearn_kde')

    # No proper motion error included
    if delta_mu_ra_err == 0.0 or delta_mu_dec_err == 0.0:
        delta_mu = np.sqrt((mu_ra1-mu_ra2)**2 + (mu_dec1-mu_dec2)**2)
        P_mu = 2.0*np.pi*delta_mu * density

    # Monte Carlo delta mu integration
    else:
        delta_mu_ra = normal(loc=(mu_ra1-mu_ra2), scale=delta_mu_ra_err, size=nsamples)
        delta_mu_dec = normal(loc=(mu_dec1-mu_dec2), scale=delta_mu_dec_err, size=nsamples)
        delta_mu = np.sqrt(delta_mu_ra**2 + delta_mu_dec**2)
        P_mu = (1.0/nsamples) * np.sum(2.0*np.pi*delta_mu * density)

    return P_mu


def set_prior_normalization(catalog, num_sys=1000000):
    """ This function calculates the normalization constant for the
    prior on random alignments by Monte Carlo integrating the stellar
    density squared over the whole sky

    Parameters
    ----------
    catalog : ndarray
        The stellar catalog over which we are integrating
    num_sys : int
        Number of random points for integration

    Returns
    -------

    """

    global C1_prior_norm

    #if C1_prior_norm is not None: return

    # Monte Carlo select random positions
    ran_theta = np.arccos(1.0-2.0*np.random.uniform(size = num_sys))
    ran_phi = 2.0 * np.pi * np.random.uniform(size = num_sys)
    ran_dec = (ran_theta-np.pi/2.0) * 180.0/np.pi
    ran_ra = ran_phi * 180.0/np.pi

    # Monte Carlo select random proper motions between (-500,+500)
    scale = 500.0
    ran_mu_ra = scale * np.random.uniform(size = num_sys) - scale/2.0
    ran_mu_dec = scale * np.random.uniform(size = num_sys) - scale/2.0


    # Calculate sigma^2
    sigma_pos = get_sigma_pos(ran_ra, ran_dec, catalog=catalog)
    sigma_pos_2 = sigma_pos**2
    sigma_mu = get_sigma_mu(ran_mu_ra, ran_mu_dec, catalog=catalog)
    sigma_mu_2 = sigma_mu**2

#    C1_prior_norm = 1.0/(2.0 * np.mean(sigma_pos_2) * c.deg_in_sky)
    C1_prior_norm = 1.0/(2.0 * np.mean(sigma_pos_2) * c.deg_in_sky * np.mean(sigma_mu_2) * scale**2)


def get_prior_random_alignment(ra, dec, mu_ra, mu_dec, t, sigma_pos=None, sigma_mu=None):
    """ Calculate the prior on C1

    Parameters
    ----------
    ra, dec : float
        Stellar position
    mu_ra, mu_dec : float
        Stellar proper motions
    t : ndarray
        The stellar catalog over which we are integrating
    sigma_pos : float
        local position density (optional)
    sigma_mu : float
        local proper motion density (optional)

    Returns
    -------
    C1_prior

    """

    global C1_prior_norm

    if C1_prior_norm is not None: set_prior_normalization(t)

    if sigma_pos is None: sigma_pos = get_sigma_pos(ra, dec, catalog=t)
    if sigma_mu is None: sigma_mu = get_sigma_mu(mu_ra, mu_dec, catalog=t)

    C1_prior = C1_prior_norm * sigma_pos**2 * sigma_mu**2 * len(t)**2

    return C1_prior


def get_P_random_alignment(ra1, dec1, ra2, dec2, mu_ra1, mu_dec1, mu_ra2, mu_dec2,
                           delta_mu_ra_err=0.0, delta_mu_dec_err=0.0,
                           nsamples=100,
                           pos_density=None, pm_density=None, catalog=None):
    """ This function calculates the probability of a
    pair of stars being formed due to random alignments.

    Parameters
    ----------
    ra1, dec1 : float
        Coordinates of the first star
    ra2, dec2 : float
        Coordinates of the second star
    mu_ra1, mu_dec1 : float
        Proper motion of the first star (mas/yr)
    mu_ra2, mu_dec2 : float
        Proper motion of the second star (mas/yr)
    delta_mu_ra_err, delta_mu_dec_err : float
        Proper motion difference uncertainties (mas/yr)
    nsamples : int
        Number of samples for Delta mu Monte Carlo integral
    catalog : structure
        Catalog to search through

    Returns
    -------
    P(data) : float
        Probability that the pair was produced randomly
    """

    # Catalog check
    if catalog is None:
        print "Must provide a catalog"
        return

    # P(pos)
    if pos_density is None:
        P_pos = get_random_alignment_P_pos(ra1, dec1, ra2, dec2, catalog=catalog)
    else:
        P_pos = get_random_alignment_P_pos(ra1, dec1, ra2, dec2,
                                           density=pos_density, catalog=catalog)

    # P(mu)
    if pm_density is None:
        P_mu = get_random_alignment_P_mu(mu_ra1, mu_dec1, mu_ra2, mu_dec2,
                                         delta_mu_ra_err=delta_mu_ra_err, delta_mu_dec_err=delta_mu_dec_err,
                                         nsamples=nsamples, catalog=catalog)
    else:
        P_mu = get_random_alignment_P_mu(mu_ra1, mu_dec1, mu_ra2, mu_dec2,
                                         delta_mu_ra_err=delta_mu_ra_err, delta_mu_dec_err=delta_mu_dec_err,
                                         nsamples=nsamples, density=pm_density, catalog=catalog)

    # So long as probabilities are independent:
    # P(pos,mu) = P(pos) * P(mu)
    P_pos_mu = P_pos * P_mu

    return P_pos_mu, P_pos, P_mu



def get_P_random_convolve(id1, id2, t, n_samples, pos_density, pm_density, plx_prior='empirical'):
    """ This function calculates the probability of a
    pair of stars being formed due to random alignments.

    Parameters
    ----------
    id1, id2 : float
        Indices of catalog for stars in the pair
    t : ndarray
        Catalog
    n_samples : int
        Number of samples for Delta mu Monte Carlo integral
    pos_density : float
        local position density
    pm_density : float
        local proper motion density
    plx_prior : string
        parallax prior method. See parallax.py for options

    Returns
    -------
    P(data) : float
        Probability that the pair was produced randomly
    """

    # Catalog check
    if t is None:
        print "Must provide a catalog"
        return


    # P(pos)
    P_pos = get_random_alignment_P_pos(t['ra'][id1], t['dec'][id1], t['ra'][id2], t['dec'][id2], density=pos_density, catalog=t)


    # Create astrometry vectors
    star1_mean = np.array([t['mu_ra'][id1], t['mu_dec'][id1], t['plx'][id1]])
    star2_mean = np.array([t['mu_ra'][id2], t['mu_dec'][id2], t['plx'][id2]])

    # Create covariance matrices
    star1_cov = np.array([[t['mu_ra_err'][id1]**2, t['mu_ra_mu_dec_cov'][id1], t['mu_ra_plx_cov'][id1]], \
                   [t['mu_ra_mu_dec_cov'][id1], t['mu_dec_err'][id1]**2, t['mu_dec_plx_cov'][id1]], \
                   [t['mu_ra_plx_cov'][id1], t['mu_dec_plx_cov'][id1], t['plx_err'][id1]**2]])
    star2_cov = np.array([[t['mu_ra_err'][id2]**2, t['mu_ra_mu_dec_cov'][id2], t['mu_ra_plx_cov'][id2]], \
                   [t['mu_ra_mu_dec_cov'][id2], t['mu_dec_err'][id2]**2, t['mu_dec_plx_cov'][id2]], \
                   [t['mu_ra_plx_cov'][id2], t['mu_dec_plx_cov'][id2], t['plx_err'][id2]**2]])

    # Create multivariate_normal objects
    star1_astrometry = multivariate_normal(mean=star1_mean, cov=star1_cov)
    star2_astrometry = multivariate_normal(mean=star2_mean, cov=star2_cov)


    # Draw random samples
    star1_samples = star1_astrometry.rvs(size=n_samples)
    star2_samples = star2_astrometry.rvs(size=n_samples)


    # P(mu)
    P_mu = 1.0/float(n_samples) * np.sum(get_random_alignment_P_mu(star1_samples[:,0], star1_samples[:,1], star2_samples[:,0], star2_samples[:,1],
                                                            density=pm_density, catalog=t))


    # P(plx)
    plx_min = 0.01 * np.ones(n_samples)  # Minimum parallax is 0.01
    # P_plx_1 = 1.0/float(n_samples) * np.sum(parallax.get_plx_prior(star1_samples[:,2][star1_samples[:,2]>0.0]))
    # P_plx_2 = 1.0/float(n_samples) * np.sum(parallax.get_plx_prior(star2_samples[:,2][star2_samples[:,2]>0.0]))
    P_plx_1 = 1.0/float(n_samples) * np.sum(parallax.get_plx_prior(np.max((plx_min,star1_samples[:,2]), axis=0), prior=plx_prior))
    P_plx_2 = 1.0/float(n_samples) * np.sum(parallax.get_plx_prior(np.max((plx_min,star2_samples[:,2]), axis=0), prior=plx_prior))



    # So long as probabilities are independent:
    # P(pos,mu) = P(pos) * P(mu)
    random_likelihood = P_pos * P_mu * P_plx_1 * P_plx_2


    return random_likelihood
