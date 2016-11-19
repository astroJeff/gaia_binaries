import numpy as np
import matplotlib.pyplot as plt
from numpy.random import uniform
from scipy.optimize import newton
from scipy import stats
from scipy.stats import gaussian_kde, truncnorm, multivariate_normal
from sklearn.neighbors import KernelDensity
import corner

# Project modules
import const as c
import P_random
import parallax

binary_set = None
binary_kde = None

# Random binary parameters
def get_M1(M_low=0.5, M_high=10.0, num_sys=1):
    """ Generate a set of random primary masses from an IMF

    Parameters
    ----------
    M_low : float
        Lower mass limit
    M_high : float
        Upper mass limit
    num_sys : int
        Number of systems to Generate

    Returns
    -------
    M1 : ndarray
        Random primary masses (ndarray)
    """

    C_m = 1.0 / (M_high**(c.alpha+1.0) - M_low**(c.alpha+1.0))
    tmp_y = uniform(size=num_sys)

    return (tmp_y/C_m + M_low**(c.alpha+1.0))**(1.0/(c.alpha+1.0))

def get_M2(M1, num_sys=1):
    """ Generate secondary masses from flat mass ratio """
    return M1*uniform(size=num_sys)

def get_a(a_low=1.0e1, a_high=4.41e7, num_sys=1, alpha=-1.6, prob='log_flat'):
    """ Generate a set of orbital separations from a power law

    Parameters
    ----------
    a_low : float
        Lower separation limit
    a_high : float
        Upper mass limit (Default: 10 pc)
    num_sys : int
        Number of systems to randomly Generate
    prob : string
        Probability distribution (options: 'log_flat', 'raghavan', 'power_law')

    Returns
    -------
    a : ndarray
        Random separations (ndarray)
    """

    if prob == 'log_flat':
        C_a = 1.0 / (np.log(a_high) - np.log(a_low))
        tmp_y = uniform(size=num_sys)
        return a_low*np.exp(tmp_y/C_a)

    elif prob == 'raghavan':
        mu_P_orb = 5.03
        sigma_P_orb = 2.28
        # Assume a system mass of 2 Msun
        P_orb_low = np.log10(a_to_P(1.0, 1.0, a_low))
        P_orb_high = np.log10(a_to_P(1.0, 1.0, a_high))
        # Set limits for truncnorm
        a, b = (P_orb_low - mu_P_orb) / sigma_P_orb, (P_orb_high - mu_P_orb) / sigma_P_orb
        P_orb = 10.0**(truncnorm.rvs(a, b, loc=mu_P_orb, scale=sigma_P_orb, size=num_sys))
        return P_to_a(1.0, 1.0, P_orb)

    elif prob == 'power_law':
        if alpha == -1:
            return get_a(a_low=a_low, a_high=a_high, num_sys=num_sys, prob='log_flat')

        # Scale so 50% of binaries have a > 100 AU
        a_low, a_high = 100.0 * c.AU_to_cm / c.Rsun_to_cm, 4.41e7
        C_a = 0.5 / (a_high**(alpha + 1.0) - a_low**(alpha + 1.0))

        a_low = ((4.41e7)**(alpha + 1.0) - 1.0 / C_a) ** (1.0 / (alpha + 1.0))
        tmp_y = uniform(size=num_sys)
        return (tmp_y/C_a + a_low**(alpha + 1.0))**(1.0/(alpha + 1.0))

    else:
        print "You must provide a valid probability distribution"
        print "Options are 'log_flat', 'raghavan', or 'power_law'"
        return



def get_e(num_sys=1, prob='thermal'):
    """ Return e from an input distribution

    Parameters
    ----------
    num_sys : int
        Number of systems

    prob : string
        Probability distribution (options: 'thermal', 'flat', 'circular', 'tokovinin')
    """

    if prob == 'thermal':
        return np.sqrt(uniform(size=num_sys))
    elif prob == 'flat':
        return uniform(size=num_sys)
    elif prob == 'circular':
        return np.zeros(num_sys)
    elif prob == 'tokovinin':  # From Tokovinin & Kiyaeva (2016), MNRAS 456
        return (-1.0 + np.sqrt(1.0 + 15.0*uniform(size=num_sys))) / 3.0
    else:
        print "You must provide a valid probability distribution"
        print "Options are 'thermal', 'flat', 'circular', or 'tokovinin'"
        return

# Random orbital orientation Parameters
def get_M(num_sys=1):
    """ Random mean anomalies """
    return 2.0 * np.pi * uniform(size = num_sys)

def get_inc(num_sys=1):
    """ Random inclination angles """
    return np.arccos(1.0-2.0*uniform(size = num_sys))

def get_omega(num_sys=1):
    """ Random arguments of periapse """
    return 2.0*np.pi*uniform(size = num_sys)

def get_Omega(num_sys=1):
    """ Random longitudes of the ascending node """
    return 2.0*np.pi*uniform(size = num_sys)

def create_binaries(num_sys=1, ecc_prob='thermal', a_prob='log_flat'):
    """ Wrapper to generate num_sys number of random binaries

    Parameters
    ----------
    num_sys : int
        Number of random binaries to generate
    ecc_prob : string
        Probability distribution to use for eccentricity (options in get_e function)
    a_prob : string
        Probability distrbution to use for orbital separation (options in get_a function)

    Returns
    -------
    M1 : ndarray
        Primary masses (Msun)
    M2 : ndarray
        Secondary masses (Msun)
    a : ndarray
        separations (Rsun)
    e : float
        Eccentricity
    Omega : float
        Longitude of the ascending node (radians)
    omega : float
        Argument of periapse (radians)
    inc : float
        Inclination angle (radians)
    """

    M1 = get_M1(num_sys=num_sys)
    M2 = get_M2(M1, num_sys=num_sys)
    a = get_a(num_sys=num_sys, prob=a_prob)
    e = get_e(num_sys=num_sys, prob=ecc_prob)
    M = get_M(num_sys=num_sys)
    omega = get_omega(num_sys=num_sys)
    Omega = get_Omega(num_sys=num_sys)
    inc = get_inc(num_sys=num_sys)

    return M1, M2, a, e, M, Omega, omega, inc


# Binary functions
def P_to_a(M1, M2, P):
    """ Orbital period (days) to separation (Rsun) """
    mu = c.GGG * (M1 + M2) * c.Msun_to_g
    n = 2.0*np.pi / P / c.day_to_sec
    return np.power(mu/(n*n), 1.0/3.0) / c.Rsun_to_cm

def a_to_P(M1, M2, a):
    """ Orbital separation (Rsun) to period (days) """
    mu = c.GGG * (M1 + M2) * c.Msun_to_g
    n = np.sqrt(mu/(a**3 * c.Rsun_to_cm**3))
    return 2.0*np.pi / n / c.day_to_sec

def get_f(M, e):
    """ Function to get the true anomaly

    Parameters
    ----------
    M : float
        Mean anomaly (radians)
    e : float
        Eccentricity

    Returns
    -------
    f : float
        true anomaly (radians)
    """

    # Get eccentric anomaly
    def func_E(x,M,e):
        return M - x + e*np.sin(x)

    E = newton(func_E, 0.5, args=(M,e))

    # Get true anomaly from eccentric anomaly
    f = np.arccos((np.cos(E)-e)/(1.0-e*np.cos(E)))
    if np.sin(E) < 0:
        f = 2.0*np.pi - f

    return f

def get_proj_sep(f, e, sep, Omega, omega, inc):
    """ Function to get the projected physical separation

    Parameters
    ----------
    f : float
        True anomaly (radians)
    e : float
        Eccentricity
    sep : float
        Physical separation : a*[1-e*cos(f)]
    Omega : float
        Longitude of the ascending node (radians)
    omega : float
        Argument of periapse (radians)
    inc : float
        Inclination angle (radians)

    Returns
    -------
    proj_sep : float
        Projected separation
    """

    sep_x = sep*(np.cos(Omega)*np.cos(omega+f) - np.sin(Omega)*np.sin(omega+f)*np.cos(inc))
    sep_y = sep*(np.sin(Omega)*np.cos(omega+f) + np.cos(Omega)*np.sin(omega+f)*np.cos(inc))
    proj_sep = np.sqrt(sep_x**2 + sep_y**2)

    return proj_sep


def get_delta_v_trans(f, e, a, P, Omega, omega, inc):
    """ Return the tangential peculiar velocity

    Parameters
    ----------
    f : float
        True anomaly (radians)
    e : float
        Eccentricity
    a : float
        Physical separation : a*[1-e*cos(f)] (cm)
    P : float
        Orbital period (sec)
    Omega : float
        Longitude of the ascending node (radians)
    omega : float
        Argument of periapse (radians)
    inc : float
        Inclination angle (radians)

    Returns
    -------
    delta_v_trans : float
        Tangential velocity (km/s)
    """

    # r_dot = a * e * np.sin(f) / np.sqrt(1.0 - e*e) * (2.0*np.pi/P)
    # r_f_dot = a / np.sqrt(1.0 - e*e) * (1.0 + e*np.cos(f)) * (2.0*np.pi/P)
    # delta_vel_1 = r_dot * (np.cos(Omega)*np.cos(omega+f) - np.sin(Omega)*np.sin(omega+f)*np.cos(inc))
    # delta_vel_2 = r_f_dot * (np.cos(Omega)*np.sin(omega+f) + np.sin(Omega)*np.cos(omega+f)*np.cos(inc))
    # delta_v_trans = np.sqrt(delta_vel_1**2 + delta_vel_2**2) / 1.0e5

    coeff = (2.0*np.pi/P) * a / np.sqrt(1.0 - e*e)
    v_x = -np.sin(omega+f)*np.cos(Omega) - e*np.sin(omega)*np.cos(Omega) - \
            np.cos(omega+f)*np.cos(inc)*np.sin(Omega) - e*np.cos(omega)*np.cos(inc)*np.sin(Omega)
    v_y = -np.sin(omega+f)*np.sin(Omega) - e*np.sin(omega)*np.sin(Omega) + \
            np.cos(omega+f)*np.cos(inc)*np.cos(Omega) + e*np.cos(omega)*np.cos(inc)*np.cos(Omega)
    delta_v_trans = coeff * np.sqrt(v_x**2 + v_y**2) / 1.0e5

    return delta_v_trans


def calc_theta_delta_v_trans(M1, M2, a, e, M, Omega, omega, inc):
    """ From the random orbits, calculate the projected separation, velocity

    Parameters
    ----------
    M1 : float
        Primary mass (Msun)
    M2 : float
        Secondary mass (Msun)
    a : float
        Orbital separation (Rsun)
    e : float
        Eccentricity
    M : float
        Mean anomalies (radians)
    Omega : float
        Longitude of the ascending node (radians)
    omega : float
        Argument of periapse (radians)
    inc : float
        Inclination angle (radians)

    Returns
    -------
    proj_sep : float
        Projected separations (ndarray, Rsun)
    delta_v_trans : float
        Tangential velocities (ndarray, km/s)
    """

    # Calculate f's
    num_sys = len(M1)
    f = np.zeros(num_sys)
    for i in np.arange(num_sys):
        f[i] = get_f(M[i], e[i])

    # Calculate separations - in Rsun
    sep = a * (1.0 - e*e) / (1.0 + e*np.cos(f))
    proj_sep = get_proj_sep(f, e, sep, Omega, omega, inc)

    # Orbital period in days
    P = a_to_P(M1, M2, a)
    # Calculate proper motions
    delta_v_trans = get_delta_v_trans(f, e, a*c.Rsun_to_cm, P*c.day_to_sec, Omega, omega, inc)

    return proj_sep, delta_v_trans



def get_P_binary(proj_sep, delta_v_trans, num_sys=100000, method='kde', kde_method='sklearn'):
    """ This function calculates the probability of a
    random star having the observed proper motion

    Parameters
    ----------
    proj_sep : float
        Projected separation between two stars
    delta_v_trans : float
        Transverse velocity difference between two stars
    method : string
        Method to perform 2D interpolation (options:kde)
    kde_method : string
        Which KDE algorithm to use (options: scipy, sklearn)

    Returns
    -------
    P(proj_sep, delta_v_trans) : float
        Probability that angular separation, pm separation
        is due to a genuine binary
    """

    # Catalog check
    global binary_set

    if binary_set is None:
        generate_binary_set(num_sys=num_sys)

    if method is 'kde':
        # Use a Gaussian KDE
        global binary_kde
        #if binary_kde is None: binary_kde = gaussian_kde((binary_set["proj_sep"], binary_set["delta_v_trans"]))
        # We work in log space for the set of binaries

        if kde_method is not 'sklearn' and kde_method is not 'scipy':
            print "Must use a valid kde algorithm: options are 'sklearn' and 'scipy'"
            print "NOTE: sklean's KDE is the Lotus to scipy's 3-cylinder Pinto"
            return


        if binary_kde is None:
            if kde_method is 'sklearn':
                kwargs = {'kernel':'tophat'}
                binary_kde = KernelDensity(bandwidth=0.1, **kwargs)
                binary_kde.fit( np.array([np.log10(binary_set['proj_sep']), np.log10(binary_set['delta_v_trans'])]).T )
            else:
                binary_kde = gaussian_kde((np.log10(binary_set["proj_sep"]), np.log10(binary_set["delta_v_trans"])))


        if isinstance(delta_v_trans, np.ndarray) and isinstance(proj_sep, np.ndarray):

            if kde_method is 'sklearn':
                values = np.array([np.log10(proj_sep), np.log10(delta_v_trans)]).T
                prob_binary = np.exp(binary_kde.score_samples(values))
            else:
                values = np.array([np.log10(proj_sep), np.log10(delta_v_trans)])
                prob_binary = binary_kde.evaluate(values)


        elif isinstance(delta_v_trans, np.ndarray):

            if kde_method is 'sklearn':
                values = np.array([np.log10(proj_sep)*np.ones(len(delta_v_trans)), np.log10(delta_v_trans)]).T
                prob_binary = np.exp(binary_kde.score_samples(values))
            else:
                values = np.array([np.log10(proj_sep)*np.ones(len(delta_v_trans)), np.log10(delta_v_trans)])
                prob_binary = binary_kde.evaluate(values)

        else:
            if kde_method is 'sklearn':
                prob_binary = np.exp(binary_kde.score_samples([np.log10(proj_sep), np.log10(delta_v_trans)]))
            else:
                prob_binary = binary_kde.evaluate([np.log10(proj_sep), np.log10(delta_v_trans)])

    else:
        print "You must input an appropriate method."
        print "Options: 'kde' only"
        return

    # Convert back from log10-space to linear-space
    # the log(10) terms convert from log10 to ln
    prob_binary = prob_binary / (proj_sep*np.log(10.)) / (delta_v_trans*np.log(10.))

    return prob_binary


def get_P_binary_convolve(id1, id2, t, n_samples, plx_prior='empirical'):

    # Angular separation
    theta = P_random.get_theta_proj_degree(t['ra'][id1], t['dec'][id1], t['ra'][id2], t['dec'][id2])


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


    delta_mu_sample = np.sqrt((star1_samples[:,0]-star2_samples[:,0])**2 + (star1_samples[:,1]-star2_samples[:,1])**2)


    # convert from mas to asec
    dist_sample = 1.0e3 / star1_samples[:,2]

    # Convert from proper motion difference (mas/yr) to transverse velocity difference (km/s)
    # delta_v_trans = (delta_mu_sample/1.0e3/3600.0*np.pi/180.0) * dist_sample * (c.pc_to_cm/1.0e5) / (c.yr_to_sec)
    delta_v_trans = delta_mu_sample * dist_sample * c.km_s_to_mas_yr
    delta_v_trans[delta_v_trans<0.0] = 1.0e10
    # Find the physical separation (Rsun) from the angular separation (degree)
    # proj_sep = (theta*np.pi/180.0) * dist_sample * (c.pc_to_cm / c.Rsun_to_cm)
    proj_sep = theta * dist_sample * c.Rsun_to_deg
    proj_sep[proj_sep<0.0] = 1.0e10  # Remove negative separations from negative parallaxes



    # TESTING
    # Only calculate probabilities for systems with clearly non-zero probabilities
    # idx = np.where( (np.mean(proj_sep) * c.Rsun_to_cm * (delta_v_trans/1.0e5)**2) / (c.GGG * 10.0 * c.Msun_to_g) < 1.0)[0]
    idx = np.where( (proj_sep * c.Rsun_to_cm * (delta_v_trans*1.0e5)**2) / (c.GGG * 10.0 * c.Msun_to_g) < 1.0)[0]
    # No good random samples
    if len(idx) == 0: return 0.0

    jacob_dV_dmu = np.zeros(n_samples)
    jacob_ds_dtheta = np.zeros(n_samples)
    prob_bin_partial = np.zeros(n_samples)
    prob_plx_prior = np.zeros(n_samples)
    prob_plx_2 = np.zeros(n_samples)

    jacob_dV_dmu[idx] = dist_sample[idx] * c.km_s_to_mas_yr
    jacob_ds_dtheta[idx] = dist_sample[idx] * c.Rsun_to_deg
    prob_bin_partial[idx] = get_P_binary(proj_sep[idx], delta_v_trans[idx])
    # TESTING


    # Jacobians for transforming from angular to physical units
    # Units: [(km/s) / (mas/yr)]
    ## jacob_dV_dmu = dist_sample * (c.pc_to_cm/1.0e5) * (1.0 / ((180.0/np.pi)*3600.0*1.0e3)) * (1.0 / c.yr_to_sec)
    # jacob_dV_dmu = dist_sample * c.km_s_to_mas_yr
    # Units: [(Rsun) / (deg.)]
    ## jacob_ds_dtheta = dist_sample * (np.pi/180.0) * (c.pc_to_cm / c.Rsun_to_cm)
    # jacob_ds_dtheta = dist_sample * c.Rsun_to_deg



    # Find binary probabilities
    # prob_bin_partial = get_P_binary(proj_sep, delta_v_trans)
    if np.all(prob_bin_partial == 0.0): return 0.0


    # Now, let's add probabilities for second star's parallax to match
    pos = np.copy(star2_samples[idx])               # Copy over the astrometry from the second star
    pos[:,2] = star1_samples[idx,2]              # Use the parallaxes from the first star
    prob_plx_2[idx] = star2_astrometry.pdf(pos)     # Calculate the multivariate PDF
#    prob_plx_2 = norm.pdf(plx_sample, loc=t['plx'][id2], scale=t['plx_err'][id2])

    # plt.hist(prob_plx_2, histtype='step', color='k', bins=50)
    # plt.show()

    # Parallax prior
    plx_min = 0.01 * np.ones(len(idx))  # Minimum parallax is 0.01
    # plx_min = 0.01 * np.ones(n_samples)  # Minimum parallax is 0.01
    prob_plx_prior[idx] = parallax.get_plx_prior(np.max((plx_min,star1_samples[idx,2]), axis=0), prior=plx_prior)
    # prob_plx_prior[idx] = parallax.get_plx_prior(np.max((plx_min,star1_samples[:,2]), axis=0), prior=plx_prior)

    # Monte Carlo integral
    prob_binary = np.mean(prob_bin_partial * prob_plx_2 * prob_plx_prior * jacob_dV_dmu * jacob_ds_dtheta)

    return prob_binary



def generate_binary_set(num_sys=100000, ecc_prob='thermal', a_prob='log_flat'):
    """ Create set of binaries to be saved to P_binary.binary_set

    Parameters
    ----------
    num_sys : int
        Number of random binaries to generate (default = 1000000)
    ecc_prob : string
        Probability distribution to use for eccentricity (options in get_e function)
    a_prob : string
        Probability distrbution to use for orbital separation (options in get_a function)


    Returns
    -------
    None
    """

    global binary_set

    # Create random binaries
    M1, M2, a, e, M, Omega, omega, inc = create_binaries(num_sys, ecc_prob=ecc_prob, a_prob=a_prob)
    # Get random projected separations, velocities
    proj_sep, delta_v_trans = calc_theta_delta_v_trans(M1, M2, a, e, M, Omega, omega, inc)

    binary_set = np.zeros(num_sys, dtype=[('proj_sep', 'f8'),('delta_v_trans','f8')])

    binary_set['proj_sep'] = proj_sep
    binary_set['delta_v_trans'] = delta_v_trans

    return


def get_prior_binary(ra, dec, mu_ra, mu_dec, t, sigma_pos=None, sigma_mu=None):
    """ This function calculates the binary prior

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
    C2_prior : float
        Binary prior

    """

    if sigma_pos is None: sigma_pos = P_random.get_sigma_pos(ra, dec, catalog=t)
    if sigma_mu is None: sigma_mu = P_random.get_sigma_mu(mu_ra, mu_dec, catalog=t)

    C2_prior = sigma_pos * sigma_mu * len(t) * c.f_bin

    return C2_prior


def create_plot_binary(dist=100.0, num_sys=100, bins=25):
    """ Create a set of random binaries and plot the distribution
    of resulting theta vs pm

    Parameters
    ----------
    dist : float
        Distance to the population for angular (rather than physical) units (pc)
    num_sys : float
        Number of random binaries to generate
    bins : int
        Number of bins for contours

    Returns
    -------
    None
    """

    global binary_set

    if binary_set is None or len(binary_set) != num_sys:
        generate_binary_set(num_sys=num_sys, dist=dist)


    fig, ax1 = plt.subplots(1,1, figsize=(6,4))

    # Plot limits
    xmin, xmax = 0.0, 5000.0
    ymin, ymax = 0.0, 3.0
    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(ymin, ymax)

    # Plot labels
    ax1.set_xlabel("Projected separation (AU)")
    ax1.set_ylabel("Proper motion difference (km/s)")

    # Plot distribution
    contourf_kwargs = {'bins':bins}
    corner.hist2d(binary_set['proj_sep']*c.Rsun_to_cm/c.AU_to_cm, binary_set['pm'], nbins=bins,
                    range=([xmin,xmax],[ymin,ymax]), **contourf_kwargs)

    # Add angular separation at dist axis
    ax2 = ax1.twiny()
    xticks = np.linspace(xmin,xmax,6)
    angles = (xticks * c.AU_to_cm)/(dist * c.pc_to_cm) * (180.0 * 3600.0 / np.pi)
    ax2.set_xticks(angles)
    ax2.set_xlabel('Angular separation at distance of ' + str(dist) + ' pc (arcsec)')

    # Add proper motion at dist axis
    ax3 = ax1.twinx()
    yticks = np.linspace(ymin, ymax, 7)
    def pm_at_dist(pm, dist=100.0):
        return (pm * 1.0e5)/(dist * c.pc_to_cm) * (1.0e3 * 180.0 * 3600.0 / np.pi) * c.day_to_sec*365.25

    ax3.set_ylim(0.0, pm_at_dist(ax1.get_ylim()[1], dist=dist))
    ax3.set_ylabel('Proper motion at distance of ' + str(dist) + ' pc (mas/yr)')

    plt.tight_layout()
    plt.show()
