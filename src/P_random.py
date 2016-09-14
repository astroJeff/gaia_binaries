import numpy as np
from scipy.stats import gaussian_kde
from numpy.random import normal
import const as c


mu_kde = None
pos_kde = None

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


def get_sigma_pos(ra, dec, catalog=None, rad=5.0, method='kde'):
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
        Density determination method (options: kde, empirical)

    Returns
    -------
    sigma_star : float
        Local density of stars per square degree
    """

    # Catalog check
    if catalog is None:
        print "You must provide a catalog"
        return
    if method is 'empirical':
        # Estimate number density of stars from number of systems within 5 degrees
        sigma_star = (nstars_nearby(ra, dec, radius=rad, catalog=catalog)-1) / (4.0*np.pi* rad**2)
    elif method is 'kde':
        # Use a Gaussian KDE
        global pos_kde
        if pos_kde is None:

            if c.kde_subset:
                # Select a subset of catalog for the KDE
                ra_ran = np.copy(catalog['ra'])
                dec_ran = np.copy(catalog['dec'])
                np.random.shuffle(ra_ran)
                np.random.shuffle(dec_ran)

                # KDE needs to be in terms of (ra*cos(dec), dec) so units are deg^-2
                pos_kde = gaussian_kde((ra_ran[0:100000] * np.cos(dec_ran[0:100000]*np.pi/180.0), dec_ran[0:100000]))
            else:
                pos_kde = gaussian_kde((catalog['ra'] * np.cos(catalog['dec']*np.pi/180.0), catalog['dec']))

        sigma_star = pos_kde.evaluate((ra, dec))
    else:
        print "You must provide a valid method"
        print "Options: 'kde', 'empirical'"
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




def get_sigma_mu(mu_ra, mu_dec, catalog=None, rad=5.0, method='kde'):
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
        Method to perform 2D interpolation (options:kde)

    Returns
    -------
    sigma_mu : float
        Local density of stars (n/(mas/yr)^2)
    """

    # Catalog check
    if catalog is None:
        print "You must include a catalog."
        return

    if method is 'kde':
        # Use a Gaussian KDE
        global mu_kde
        if mu_kde is None:

            if c.kde_subset:
                # Select a subset of catalog for the KDE
                mu_ra_ran = np.copy(catalog['mu_ra'])
                mu_dec_ran = np.copy(catalog['mu_dec'])
                np.random.shuffle(mu_ra_ran)
                np.random.shuffle(mu_dec_ran)

                mu_kde = gaussian_kde((mu_ra_ran[0:100000], mu_dec_ran[0:100000]))
            else:
                # Use the whole catalog
                mu_kde = gaussian_kde((catalog['mu_ra'], catalog['mu_dec']))

        sigma_mu = mu_kde.evaluate((mu_ra, mu_dec))
    elif method is 'empirical':
        n_stars_near = nstars_nearby_mu(mu_ra, mu_dec, radius=rad, catalog=catalog)-1
        sigma_mu = n_stars_near / (4.0*np.pi * rad**2)
    else:
        print "You must input an appropriate method."
        print "Options: 'kde' or 'empirical'"
        return

    return sigma_mu



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
            density = get_sigma_mu(mu_ra1, mu_dec1, catalog=catalog, method='kde')

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
