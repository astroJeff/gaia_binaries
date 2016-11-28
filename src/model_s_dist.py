import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.stats import gaussian_kde
from scipy.optimize import minimize
import emcee
import pickle

import sys
sys.path.append('../src')
import const as c


method = 'emcee'
threads = 20


# Top level KDEs
dist_log_flat_100_kde = None
dist_log_flat_100_500_kde = None
dist_power_law_100_kde = None
dist_power_law_100_500_kde = None



def main():

    global dist_log_flat_100_kde 
    global dist_log_flat_100_500_kde 
    global dist_power_law_100_kde
    global dist_power_law_100_500_kde


    # Load catalogs
    print "Loading catalogs..."
    data_dir = '../data/TGAS/'
    TGAS_power_law, TGAS_log_flat, TGAS_power_law_good, TGAS_log_flat_good = load_catalogs(data_dir)
    print "Finished loading catalogs"

    # Create distance KDEs
    print "Generating distance KDEs..."
    dist_log_flat_good, dist_power_law_good, s_log_flat_good, s_power_law_good = calc_distances(TGAS_log_flat, TGAS_power_law)
    dist_log_flat_100_kde, dist_log_flat_100_500_kde = generate_dist_KDEs(dist_log_flat_good, s_log_flat_good)
    dist_power_law_100_kde, dist_power_law_100_500_kde = generate_dist_KDEs(dist_power_law_good, s_power_law_good)
    print "Finished generating distance KDEs"

    # Select subsets of systems within 2 regions
    print "Getting theta regions..."
    theta_log_flat_region_1, theta_log_flat_region_2 = get_theta_regions(dist_log_flat_good, TGAS_log_flat_good)
    theta_power_law_region_1, theta_power_law_region_2 = get_theta_regions(dist_power_law_good, TGAS_power_law_good)
    print "Finished getting theta regions"

    
    if method == 'emcee':

        # Run emcee
        print "Running emcee..."
        dist_KDE_prior = 'log_flat'
        args = dist_KDE_prior, theta_power_law_region_1, theta_power_law_region_2

        sampler = run_emcee(ln_posterior, args, nwalkers=20, threads=threads)
        print "Finished running emcee"

        pickle.dump( sampler, open( "model_s_dist_log_flat.p", "wb" ))


    else:

        # Starting values
        print "Minimizing..."
        alpha_1 = -1.0
        alpha_2 = -1.6
        s_crit = 2.0e3
        dist_KDE_prior = 'log_flat'

        x0 = np.array([alpha_1, alpha_2, s_crit])
        args = dist_KDE_prior, theta_power_law_region_1, theta_power_law_region_2

        bounds = ([-3.0, -1.0e-5], [-3.0, -1.0e-5], [1.0, 1.0e5])
        print minimize(get_neg_log_likelihood, x0, args=args, method='L-BFGS-B', bounds=bounds)
        print "Finished minimizing"


    return



def load_catalogs(data_dir):
    """ Function to return catalogs """

    if data_dir is None:
        print "You must provide the directory filepath"
        return

    dtype = [('P_posterior','f8'), ('theta','f8'),
             ('source_id_1','<i8'), ('TYC_id_1','S11'), ('hip_id_1','<i8'),
             ('ra_1','f8'), ('dec_1','f8'),
             ('mu_ra_1','f8'), ('mu_dec_1','f8'), ('mu_ra_err_1','f8'), ('mu_dec_err_1','f8'),
             ('plx_1','f8'), ('plx_err_1','f8'),
             ('gaia_g_flux_1','<f8'), ('gaia_g_flux_err_1','<f8'), ('gaia_g_mag_1','<f8'),
             ('TMASS_id_1','<i8'), ('TMASS_angle_dist_1','<f8'),
             #            ('TMASS_n_neighbours_1','<i8'), ('TMASS_n_mates_1','<i8'), ('TMASS_ph_qual_1','S11'),
             ('TMASS_ra_1','<f8'), ('TMASS_dec_1','<f8'),
             ('TMASS_j_mag_1','<f8'), ('TMASS_j_mag_err_1','<f8'),
             ('TMASS_h_mag_1','<f8'), ('TMASS_h_mag_err_1','<f8'),
             ('TMASS_ks_mag_1','<f8'), ('TMASS_ks_mag_err_1','<f8'),
             ('TYC_Vt_1','<f8'), ('TYC_Vt_err_1','<f8'),
             ('TYC_Bt_1','<f8'), ('TYC_Bt_err_1','<f8'),
             ('gaia_delta_Q_1','<f8'), ('gaia_noise_1','<f8'),
             #
             ('source_id_2','<i8'), ('TYC_id_2','S11'), ('hip_id_2','<i8'),
             ('ra_2','f8'), ('dec_2','f8'),
             ('mu_ra_2','f8'), ('mu_dec_2','f8'), ('mu_ra_err_2','f8'), ('mu_dec_err_2','f8'),
             ('plx_2','f8'), ('plx_err_2','f8'),
             ('gaia_g_flux_2','<f8'), ('gaia_g_flux_err_2','<f8'), ('gaia_g_mag_2','<f8'),
             ('TMASS_id_2','<i8'), ('TMASS_angle_dist_2','<f8'),
             #         ('TMASS_n_neighbours_2','<i8'), ('TMASS_n_mates_2','<i8'), ('TMASS_ph_qual_2','S11'),
             ('TMASS_ra_2','<f8'), ('TMASS_dec_2','<f8'),
             ('TMASS_j_mag_2','<f8'), ('TMASS_j_mag_err_2','<f8'),
             ('TMASS_h_mag_2','<f8'), ('TMASS_h_mag_err_2','<f8'),
             ('TMASS_ks_mag_2','<f8'), ('TMASS_ks_mag_err_2','<f8'),
             ('TYC_Vt_2','<f8'), ('TYC_Vt_err_2','<f8'),
             ('TYC_Bt_2','<f8'), ('TYC_Bt_err_2','<f8'),
             ('gaia_delta_Q_2','<f8'), ('gaia_noise_2','<f8')
            ]

    # folder = '../data/TGAS/'

    TGAS_power_law = np.genfromtxt(data_dir+'gaia_wide_binaries_TGAS_plx_exponential_a_power_law_cleaned.txt', dtype=dtype, names=True)
    TGAS_log_flat = np.genfromtxt(data_dir+'gaia_wide_binaries_TGAS_plx_exponential_cleaned.txt', dtype=dtype, names=True)

    TGAS_power_law_good = TGAS_power_law[TGAS_power_law['P_posterior'] > 0.99]
    TGAS_log_flat_good = TGAS_log_flat[TGAS_log_flat['P_posterior'] > 0.99]

    return TGAS_power_law, TGAS_log_flat, TGAS_power_law_good, TGAS_log_flat_good



def calc_distances(TGAS_log_flat, TGAS_power_law):
    """ Calculate the distance arrays for both models """

    #### LOG FLAT ####
    # Calculate distance from average parallaxes, weighted by uncertainties
    dist_log_flat = np.zeros(len(TGAS_log_flat))
    for i in np.arange(len(TGAS_log_flat)):
        vals = [TGAS_log_flat['plx_1'][i],TGAS_log_flat['plx_2'][i]]
        weights = [1.0/TGAS_log_flat['plx_err_1'][i]**2,1.0/TGAS_log_flat['plx_err_2'][i]**2]
        dist_log_flat[i] = 1.0e3/np.average(vals, weights=weights)

    # Calculate the physical separation in AU
    s_log_flat = (TGAS_log_flat['theta']*np.pi/180.0/3600.0) * dist_log_flat * (c.pc_to_cm/c.AU_to_cm)

    # Calculate the primary's proper motion
    mu_1_log_flat = np.sqrt(TGAS_log_flat['mu_ra_1']**2 + TGAS_log_flat['mu_dec_1']**2)
    delta_mu_log_flat = np.sqrt((TGAS_log_flat['mu_ra_1']-TGAS_log_flat['mu_ra_2'])**2 + (TGAS_log_flat['mu_dec_1']-TGAS_log_flat['mu_dec_2'])**2)

    # values for "good" pairs only
    dist_log_flat_good = dist_log_flat[TGAS_log_flat['P_posterior'] > 0.99]
    s_log_flat_good = s_log_flat[TGAS_log_flat['P_posterior'] > 0.99]



    #### POWER LAW ####
    # Calculate distance from average parallaxes, weighted by uncertainties
    dist_power_law = np.zeros(len(TGAS_power_law))
    for i in np.arange(len(TGAS_power_law)):
        vals = [TGAS_power_law['plx_1'][i],TGAS_power_law['plx_2'][i]]
        weights = [1.0/TGAS_power_law['plx_err_1'][i]**2,1.0/TGAS_power_law['plx_err_2'][i]**2]
        dist_power_law[i] = 1.0e3/np.average(vals, weights=weights)

    # Calculate the physical separation in AU
    s_power_law = (TGAS_power_law['theta']*np.pi/180.0/3600.0) * dist_power_law * (c.pc_to_cm/c.AU_to_cm)

    # Calculate the primary's proper motion
    mu_1_power_law = np.sqrt(TGAS_power_law['mu_ra_1']**2 + TGAS_power_law['mu_dec_1']**2)
    delta_mu_power_law = np.sqrt((TGAS_power_law['mu_ra_1']-TGAS_power_law['mu_ra_2'])**2 + (TGAS_power_law['mu_dec_1']-TGAS_power_law['mu_dec_2'])**2)

    # values for "good" pairs only
    dist_power_law_good = dist_power_law[TGAS_power_law['P_posterior'] > 0.99]
    s_power_law_good = s_power_law[TGAS_power_law['P_posterior'] > 0.99]

    return dist_log_flat_good, dist_power_law_good, s_log_flat_good, s_power_law_good


def generate_dist_KDEs(dist, s):
    idx = reduce(np.intersect1d,
                 [np.where(dist < 500.0)[0],
                  np.where(s < 5.0e4)[0],
                  np.where(s > 5.0e3)[0]
                 ])

    dist_100_kde = gaussian_kde(dist[idx][dist[idx]<100.0], bw_method=0.3)

    idx_100_500 = np.intersect1d(np.where(dist[idx]>100.0)[0], np.where(dist[idx]<500.0)[0])
    dist_100_500_kde = gaussian_kde(dist[idx][idx_100_500], bw_method=0.1)

    return dist_100_kde, dist_100_500_kde


def get_theta_regions(dist, TGAS_good):
    # Log flat sample

    idx_region_1 = reduce(np.intersect1d,
                          [np.where(dist<100.0)[0],
                           np.where(TGAS_good['theta'] > 10.0)[0],
                           np.where(TGAS_good['theta'] < 600.0)[0]
                          ])
    idx_region_2 = reduce(np.intersect1d,
                          [np.where(dist<500.0)[0],
                           np.where(dist>100.0)[0],
                           np.where(TGAS_good['theta'] > 10.0)[0],
                           np.where(TGAS_good['theta'] < 100.0)[0]
                          ])

    theta_region_1 = TGAS_good['theta'][idx_region_1]
    theta_region_2 = TGAS_good['theta'][idx_region_2]

    return theta_region_1, theta_region_2


def calc_integrand(dist, theta, dist_KDE_prior, region, alpha_1, alpha_2, s_crit):
    """
    This function is to calculate the integrand for the integral over distance

    Parameters
    ----------
    dist : float
        Distance (pc)
    theta : float
        Angular separation (arcsec)
    dist_kde : scipy gaussian_kde
        KDE based on the distance distribution observed
    alpha_1, alpha_2 : float
        Power law indices for s distribution
    s_crit : float
        Critical s defining the break in the power law

    Returns
    -------
    integrand : float
        Integrand of the equation

    """

    # Load dist KDEs
    global dist_log_flat_100_kde
    global dist_log_flat_100_500_kde
    global dist_power_law_100_kde
    global dist_power_law_100_500_kde

    if dist_KDE_prior != 'log_flat' and dist_KDE_prior != 'power_law': return 
    if region != 1 and region != 2: return

    if dist_KDE_prior == 'log_flat' :
        if region == 1:
            dist_kde = dist_log_flat_100_kde
        else:
            dist_kde = dist_log_flat_100_500_kde
    else:
        if region == 1:
            dist_kde = dist_power_law_100_kde
        else:
            dist_kde = dist_power_law_100_500_kde



    # Get the separation in Rsun
    s = (theta / 3600.0 * np.pi/180.0) * (dist * c.pc_to_cm / c.AU_to_cm)

    # Get probability from our broken power law model
    P_s = get_P_s(s, alpha_1, alpha_2, s_crit)

#     Projected separation is above 3.0e5 AU
#     P_s[np.where(s > 3.0e5)] = 0.0

    # Get the distance probability
    P_dist = dist_kde.evaluate((dist))

    # Calculate integrand
    integrand = P_s * dist * P_dist# * norm

    return integrand



def calc_integral(theta, dist_min, dist_max, dist_KDE_prior, region, alpha_1, alpha_2, s_crit):
    """
    Calculate the integral over distance of all binaries that
    could match the observed angular separation, theta.

    Parameters
    ----------
    theta : float
        Angular separation (asec)
    dist_min, dist_max : float
        Min, max distance of integration
    dist_kde : scipy gaussian_kde
        KDE to represent distance distribution of a sample
    alpha_1, alpha_2 : float
        Power law indices for s distribution
    s_crit : float
        Critical s defining the break in the power law



    Returns
    -------
    P_theta : float
        Value of integration
    """

    args = theta, dist_KDE_prior, region, alpha_1, alpha_2, s_crit

    val = quad(calc_integrand, dist_min, dist_max, args=args, epsrel=1.0e-4)

    return val[0]


def calc_theta_norm(theta_min, theta_max, dist_min, dist_max, dist_KDE_prior, region, alpha_1, alpha_2, s_crit):
    """ Calculate the normalization constant, Z, for the integral

    Parameters
    ----------
    theta_min, theta_max : float
        Minimum and maximum angles for calculating the integrals
    dist_min, dist_max : float
        Minimum and maximum distances of the sample
    dist_kde : scipy gaussian_kde
        KDE to represent distance distribution of a sample
    alpha_1, alpha_2 : float
        Power law indices for s distribution
    s_crit : float
        Critical s defining the break in the power law


    Returns
    -------
    Z : float
        Normalization constant for the integral
    """

    args = dist_min, dist_max, dist_KDE_prior, region, alpha_1, alpha_2, s_crit

    norm = quad(calc_integral, theta_min, theta_max, args=args)

    return norm[0]


def get_P_s(s, alpha_1, alpha_2, s_crit):
    """ Calculate the non-normalized distribution
    probability from a 2-component, broken power law

    Parameters
    ----------
    s : float
        Projected physical separation of the binary
    alpha_1, alpha_2 : float
        Power law indices for s distribution
    s_crit : float
        Critical s defining the break in the power law

    Returns
    -------
    P_s : float
        Probability of s
    """

    if isinstance(s, np.ndarray):
        P_s = s**alpha_1
        P_s[s>s_crit] = s_crit**(alpha_1-alpha_2) * s[s>s_crit]**alpha_2
    else:
        if s < s_crit:
            P_s = s**alpha_1
        else:
            P_s = s_crit**(alpha_1-alpha_2) * s**alpha_2

    return P_s


def get_neg_log_likelihood(p, dist_KDE_prior, \
                           theta_region_1, theta_region_2):
    """ Calculate the likelihood function for a set of parameters and data

    Parameters
    ----------
    alpha_1, alpha_2 : float
        Power law indices for s distribution
    s_crit : float
        Critical s defining the break in the power law
    dist_kde_region_1, dist_kde_region_2 : scipy gaussian_kde
        KDEs to represent distance distribution of samples within Region 1 and 2
    theta_region_1, theta_region_2 : float
        Distribution of angular separations within Region 1 and 2

    Returns
    -------
    neg_ln_likelihood : float
        Negative of the log likelihood of the two data sets
    """

    alpha_1, alpha_2, s_crit = p


    # Region 1 
    region = 1
    theta_min_region_1, theta_max_region_1 = 10.0, 600.0
    dist_min_region_1, dist_max_region_1 = 0.0, 100.0
    Z_const_1 = calc_theta_norm(theta_min_region_1, theta_max_region_1, \
                                dist_min_region_1, dist_max_region_1, \
                                dist_KDE_prior, region, alpha_1, alpha_2, s_crit)

    num_region_1 = len(theta_region_1)
    likelihood_1 = np.zeros(num_region_1)

    for i in np.arange(num_region_1):
        likelihood_1[i] = calc_integral(theta_region_1[i], dist_min_region_1, \
                                        dist_max_region_1, dist_KDE_prior, region, \
                                        alpha_1, alpha_2, s_crit)




    # Region 2
    region = 2
    theta_min_region_2, theta_max_region_2 = 10.0, 100.0
    dist_min_region_2, dist_max_region_2 = 100.0, 500.0
    Z_const_2 = calc_theta_norm(theta_min_region_2, theta_max_region_2, \
                                dist_min_region_2, dist_max_region_2, \
                                dist_KDE_prior, region, alpha_1, alpha_2, s_crit)

    num_region_2 = len(theta_region_2)
    likelihood_2 = np.zeros(num_region_2)

    for i in np.arange(num_region_2):
        likelihood_2[i] = calc_integral(theta_region_2[i], dist_min_region_2, \
                                        dist_max_region_2, dist_KDE_prior, region, \
                                        alpha_1, alpha_2, s_crit)


    neg_ll = -1.0 * (np.sum(np.log(likelihood_1)) + np.sum(np.log(likelihood_2)) + \
               - num_region_1 * np.log(Z_const_1) - num_region_2 * np.log(Z_const_2))

    print p, neg_ll

    # Combine likelihoods
    return neg_ll


def ln_posterior(p, dist_KDE_prior, theta_region_1, theta_region_2):
    """ Calculate the posterior probability for a set of parameters and data

    Parameters
    ----------
    alpha_1, alpha_2 : float
        Power law indices for s distribution
    s_crit : float
        Critical s defining the break in the power law
    dist_kde_region_1, dist_kde_region_2 : scipy gaussian_kde
        KDEs to represent distance distribution of samples within Region 1 and 2
    theta_region_1, theta_region_2 : float
        Distribution of angular separations within Region 1 and 2

    Returns
    -------
    ln_posterior : float
        Log posterior probability of the two data sets
    """

    alpha_1, alpha_2, s_crit = p

    # Priors only act as bounds
    if alpha_1 < -3.0 or alpha_1 > -1.0e-5:
        return -np.inf
    if alpha_2 < -3.0 or alpha_2 > -1.0e-5:
        return -np.inf
    if s_crit < 10.0 or s_crit > 1.0e5:
        return -np.inf

    # Call the neg_log_likelihood function and multiply by -1 to get the log likelihood
    ll = -1.0 * get_neg_log_likelihood(p, dist_KDE_prior, theta_region_1, theta_region_2)

    # Since we have no priors, we return the log likelihood here
    return ll


def run_emcee(ln_posterior, args, nburn=100, nsteps=100, nwalkers=16, threads=1, mpi=False):

    # Assign initial values
    p0 = np.zeros((nwalkers,3))
    p0[:,0] = np.random.normal(-1.0, 0.005, size=nwalkers) # alpha_1
    p0[:,1] = np.random.normal(-1.6, 0.005, size=nwalkers) # alpha_2
    p0[:,2] = 10**np.random.normal(3.0, 0.005, size=nwalkers) # pivot point

    # Set up sampler
    if mpi == True:
        pool = MPIPool()
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
        sampler = emcee.EnsembleSampler(nwalkers=nwalkers, dim=3, lnpostfn=ln_posterior, args=args, pool=pool)
    elif threads == 1:
        sampler = emcee.EnsembleSampler(nwalkers=nwalkers, dim=3, lnpostfn=ln_posterior, args=args)
    elif threads > 1 and isinstance( threads, ( int, long ) ):
        sampler = emcee.EnsembleSampler(nwalkers=nwalkers, dim=3, lnpostfn=ln_posterior, args=args, threads=threads)
    else:
        print "You must provide a reasonable integer number of threads, more than 1"
        return

    # Burn-in
    pos,prob,state = sampler.run_mcmc(p0, N=nburn)

    # Full run
    sampler.reset()
    pos,prob,state = sampler.run_mcmc(pos, N=nsteps)

    return sampler





if __name__ == '__main__':
    main()
