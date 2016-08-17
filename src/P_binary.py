import numpy as np
import matplotlib.pyplot as plt
from numpy.random import uniform
from scipy.optimize import newton
from scipy import stats
from scipy.stats import gaussian_kde
import corner

# Project modules
import const as c


binary_set = None
binary_set_dist = None
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

def get_a(a_low=1.0e4, a_high=1.0e6, num_sys=1):
    """ Generate a set of orbital separations from a power law

    Parameters
    ----------
    a_low : float
        Lower separation limit
    a_high : float
        Upper mass limit
    num_sys : in
        Number of systems to randomly Generate

    Returns
    -------
    a : ndarray
        Random separations (ndarray)
    """
    C_a = 1.0 / (np.log(a_high) - np.log(a_low))
    tmp_y = uniform(size=num_sys)

    return a_low*np.exp(tmp_y/C_a)

def get_e(num_sys=1):
    """ Return e from thermal distribution """
    return np.sqrt(uniform(size=num_sys))

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

def create_binaries(num_sys=1):
    """ Wrapper to generate num_sys number of random binaries

    Parameters
    ----------
    num_sys : int
        Number of random binaries to generate

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
    a = get_a(num_sys=num_sys)
    e = get_e(num_sys=num_sys)
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
    sep_x = sep*(np.cos(Omega)*np.sin(omega+f) + np.sin(Omega)*np.cos(omega+f)*np.cos(inc))
    sep_y = sep*(np.cos(Omega)*np.cos(omega+f) - np.sin(Omega)*np.sin(omega+f)*np.cos(inc))
    proj_sep = np.sqrt(sep_x**2 + sep_y**2)

    return proj_sep


def get_pm(f, e, a, P, Omega, omega, inc):
    """ Return the tangential peculiar velocity

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
    pm : float
        Tangential velocity (km/s)
    """

    r_dot = a * e * np.sin(f) / np.sqrt(1.0 - e*e) * (2.0*np.pi/P)
    r_f_dot = a / np.sqrt(1.0 - e*e) * (1.0 + e*np.cos(f)) * (2.0*np.pi/P)
    pm_1 = r_dot * (np.cos(Omega)*np.cos(omega+f) - np.sin(Omega)*np.sin(omega+f)*np.cos(inc))
    pm_2 = r_f_dot * (np.cos(Omega)*np.sin(omega+f) + np.sin(Omega)*np.cos(omega+f)*np.cos(inc))
    pm = np.sqrt(pm_1**2 + pm_2**2) / 1.0e5

    return pm


def calc_theta_pm(M1, M2, a, e, M, Omega, omega, inc):
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
    pm : float
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
    pm = get_pm(f, e, a*c.Rsun_to_cm, P*c.day_to_sec, Omega, omega, inc)

    return proj_sep, pm



def get_P_binary(theta, delta_mu, dist=100.0, num_sys=100000, method='kde'):
    """ This function calculates the probability of a
    random star having the observed proper motion

    Parameters
    ----------
    theta : float
        Angular distance between two stars
    delta_mu : float
        Proper Motion difference between two stars
    dist : float
        Distance to the stellar population (pc)
    method : string
        Method to perform 2D interpolation (options:kde)

    Returns
    -------
    P(theta, delta_mu) : float
        Probability that angular separation, pm separation
        is due to a genuine binary
    """

    # Catalog check
    global binary_set
    global binary_set_dist

    if binary_set is None or binary_set_dist != dist:
        generate_binary_set(num_sys=num_sys, dist=dist)
    if binary_set is not None and len(binary_set) != num_sys:
        generate_binary_set(num_sys=num_sys, dist=dist)

    # if sim_binaries is None and binary_set is None:
    #     print "No included set of simulated binaries."
    #     print "Generating " + str(num_sys) + " binaries now..."
    #
    #     binary_set = np.zeros(num_sys, dtype=[('theta', 'f8'),('delta_mu','f8')])
    #     # Helper function
    #     def pm_at_dist(pm, dist=100.0):
    #         return (pm * 1.0e5)/(dist * c.pc_to_cm) * (1.0e3 * 180.0 * 3600.0 / np.pi) * c.day_to_sec*365.25
    #
    #     # Create random binaries
    #     M1, M2, a, e, M, Omega, omega, inc = create_binaries(num_sys)
    #     # Get random projected separations, velocities
    #     proj_sep, pm = calc_theta_pm(M1, M2, a, e, M, Omega, omega, inc)
    #
    #     binary_set['theta'] = (proj_sep * c.Rsun_to_cm)/(dist * c.pc_to_cm) * (180.0 * 3600.0 / np.pi)
    #     binary_set['delta_mu'] = pm_at_dist(pm, dist=dist)
    #
    #     # Save binaries for later
    #     sim_binaries = binary_set
    #
    #     print "... Finished generating binaries"


    if method is 'kde':
        # Use a Gaussian KDE
        global binary_kde
        if binary_kde is None: binary_kde = gaussian_kde((binary_set["theta"], binary_set["delta_mu"]))

        if isinstance(delta_mu, np.ndarray):
            values = np.vstack([theta*np.ones(len(delta_mu)), delta_mu])
            P_binary = binary_kde.evaluate(values)
        else:
            P_binary = binary_kde.evaluate([theta, delta_mu])

    else:
        print "You must input an appropriate method."
        print "Options: 'kde' only"
        return

    return P_binary



def generate_binary_set(num_sys=100000, dist=100.0):
    """ Create set of binaries to be saved to P_binary.binary_set

    Parameters
    ----------
    num_sys : int
        Number of random binaries to generate (default = 1000000)
    dist : float
        Distance to binaries (default = 100 pc)

    Returns
    -------
    None
    """

    global binary_set
    global binary_set_dist

    # Create random binaries
    M1, M2, a, e, M, Omega, omega, inc = create_binaries(num_sys)
    # Get random projected separations, velocities
    proj_sep, pm = calc_theta_pm(M1, M2, a, e, M, Omega, omega, inc)

    binary_set = np.zeros(num_sys, dtype=[('proj_sep','f8'),('pm','f8'),('theta', 'f8'),('delta_mu','f8')])
    # Helper function
    def pm_at_dist(pm, dist=100.0):
        return (pm * 1.0e5)/(dist * c.pc_to_cm) * (1.0e3 * 180.0 * 3600.0 / np.pi) * c.day_to_sec*365.25

    binary_set['proj_sep'] = proj_sep
    binary_set['pm'] = pm
    binary_set['theta'] = (proj_sep * c.Rsun_to_cm)/(dist * c.pc_to_cm) * (180.0 * 3600.0 / np.pi)
    binary_set['delta_mu'] = pm_at_dist(pm, dist=dist)

    binary_set_dist = dist

    return


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
    global binary_set_dist

    if binary_set is None or binary_set_dist != dist:
        generate_binary_set(num_sys=num_sys, dist=dist)
    if binary_set is not None and len(binary_set) != num_sys:
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