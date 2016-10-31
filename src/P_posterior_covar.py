import numpy as np
from numpy.random import normal, multivariate_normal
from scipy.stats import norm, truncnorm, multivariate_normal
from functools import reduce
import P_binary
import P_random
import parallax
import const as c
from astropy.table import Table
import pickle
import time
import matplotlib.pyplot as plt


# Line corresponding to bulk of simulated binaries
def min_line(x):
    line_slope = -0.5
    line_intercept = 2.5
    return 10.**(line_slope*np.log10(x) + line_intercept)


def match_binaries(t, sys_start=0, subsample=None, size_integrate_binary=10000, size_integrate_random=10000, plx_prior='empirical'):
    """ Function to match binaries within a catalog

    Arguments
    ---------
    t : ndarray
        Catalog for self-compare for matching
    sys_start : int (optional)
        Skip the first sys_start number of systems before beginning matching
    subsample : int (optional)
        If provided, program ends after only searching for matches to subset of this size
    size_integrate_binary : int (optional)
        If provided, the number of random draws for integration over P_binary
    size_integrate_random : int (optional)
        If provided, the number of random draws for integration over P_random

    Returns
    -------
    prob_out : ndarray
        Set of matched pairs, their IDs, and their probabilities

    """


    # Generate simulated binaries
    print "Generating binaries..."
    # NOTE: Computation time scales roughly with num_sys here:
    P_binary.generate_binary_set(num_sys=10000)
    print "Done generating binaries"


    # Generate random alignment KDEs using first entry as a test position
    # print "Calculating local densities..."
    P_random.mu_kde = None
    P_random.pos_kde = None
    # pos_density = P_random.get_sigma_pos(t['ra'], t['dec'], catalog=t, method='sklearn_kde')
    # pm_density = P_random.get_sigma_mu(t['mu_ra'], t['mu_dec'], catalog=t, method='sklearn_kde')
    pos_density = P_random.get_sigma_pos(t['ra'][0:1], t['dec'][0:1], catalog=t, method='sklearn_kde')
    pm_density = P_random.get_sigma_mu(t['mu_ra'][0:1], t['mu_dec'][0:1], catalog=t, method='sklearn_kde')
    # print "Done calculating densities."


    # Generate parallax KDE for parallax prior
    if plx_prior is 'empirical':
        parallax.set_plx_kde(t, bandwidth=0.01)


    # Set normalization constant for C1 prior
    print "Calculating normalization for random alignment prior..."
    if P_random.C1_prior_norm is None: P_random.set_prior_normalization(t)
    print "Done setting prior."


    # Start time
    start = time.time()

    # Now, let's calculate the probabilities
    length = len(t)
    print "We are testing", length, "stars..."

    dtype = [('i_1','i4'),('i_2','i4'),('ID_1','i4'),('ID_2','i4'),('P_random','f8'),('P_binary','f8'),('P_posterior','f8'), \
            ('theta','f8'), ('mu_ra_1','f8'), ('mu_dec_1','f8'), ('mu_ra_2','f8'), ('mu_dec_2','f8'), ('plx_1','f8'), ('plx_2','f8')]
    prob_out = np.array([], dtype=dtype)


    for i in np.arange(length):

        #if i%1000 == 0: print i, time.time()-start

        # So we can start at any point in the catalog
        if i < sys_start: continue
        if subsample is not None and i == subsample+sys_start:
            break


        # Get ids of all stars within 1 degree and parallaxes in agreement within 3-sigma
        i_star2 = np.arange(length - i - 1) + i + 1
        theta = P_random.get_theta_proj_degree(t['ra'][i], t['dec'][i], t['ra'][i_star2], t['dec'][i_star2])
        delta_plx = np.abs(t['plx'][i]-t['plx'][i_star2])
        delta_plx_err = np.sqrt(t['plx_err'][i]**2 + t['plx_err'][i_star2]**2)
#        ids_good = np.intersect1d(i_star2[np.where(theta < 1.0)[0]], i_star2[np.where(delta_plx < 3.0*delta_plx_err)[0]])
        ids_good = reduce(np.intersect1d,
                          (i_star2[np.where(theta < 1.0)[0]],
                           i_star2[np.where(delta_plx < 5.0*delta_plx_err)[0]],
                           i_star2[np.where(theta != 0.0)[0]]))

        # Move on if no matches within 1 degree
        if len(ids_good) == 0: continue

        # Select random delta mu's for Monte Carlo integration over observational uncertainties
        theta_good = P_random.get_theta_proj_degree(t['ra'][i], t['dec'][i], t['ra'][ids_good], t['dec'][ids_good])
        delta_mu_ra_err = np.sqrt(t['mu_ra_err'][i]**2 + t['mu_ra_err'][ids_good]**2)
        delta_mu_dec_err = np.sqrt(t['mu_dec_err'][i]**2 + t['mu_dec_err'][ids_good]**2)
        delta_mu_err = np.sqrt(delta_mu_ra_err**2 + delta_mu_dec_err**2)
        delta_mu_ra = t['mu_ra'][i] - t['mu_ra'][ids_good]
        delta_mu_dec = t['mu_dec'][i] - t['mu_dec'][ids_good]
        delta_mu = np.sqrt(delta_mu_ra**2 + delta_mu_dec**2)
        mu_diff_3sigma = delta_mu - 3.0*delta_mu_err



        # Identify potential matches as ones with non-zero P(binary)
#        mu_diff_vector = np.amax(np.vstack([mu_diff_3sigma, 0.1*np.ones(len(ids_good))]), axis=0)
        mu_diff_vector = mu_diff_3sigma

        # dist in pc
        min_dist = 1.0e3 / np.amax(np.vstack([np.ones(len(ids_good)) * (t['plx'][i]+3.0*t['plx_err'][i]), t['plx'][ids_good]+3.0*t['plx_err'][ids_good]]), axis=0)
        min_dist[min_dist<0.0] = 1.0e10
        # projected separation in pc
        proj_sep_vector = (theta_good*np.pi/180.0) * min_dist * (c.pc_to_cm / c.Rsun_to_cm)
        # Transverse velocity vector in km/s
        delta_v_trans_vector = (mu_diff_vector/1.0e3/3600.0*np.pi/180.0) * min_dist * (c.pc_to_cm/1.0e5) / (c.yr_to_sec)
        # So we don't have negative delta_v_trans_vectors
#        delta_v_trans_vector = np.amax(np.vstack([delta_v_trans_vector, 0.1*np.ones(len(ids_good))]), axis=0)
        delta_v_trans_vector = np.amax(np.vstack([delta_v_trans_vector, min_line(proj_sep_vector)]), axis=0)

        ids_good_binary = np.where(P_binary.get_P_binary(proj_sep_vector, delta_v_trans_vector) > 0.0)[0]

        # If no matches, move on
        if len(ids_good_binary) == 0: continue

        # Ids of all matches
        ids_good_binary_all = ids_good[ids_good_binary]



        # More precise integration for potential matches
        for k in np.arange(len(ids_good_binary_all)):

            # IDs for the secondary
            j = ids_good_binary_all[k]


            # Calculate the posterior probability
            prob_posterior, prob_random, prob_binary = calc_P_posterior(i, j, t,
                                                                        plx_prior=plx_prior,
                                                                        size_integrate_binary=size_integrate_binary,
                                                                        size_integrate_random=size_integrate_random)



            # Select potential matches
            # if prob_posterior > 0.5:
            if prob_posterior > 0.0:
                prob_temp = np.zeros(1, dtype=dtype)
                theta = P_random.get_theta_proj_degree(t['ra'][i], t['dec'][i], t['ra'][j], t['dec'][j])
                prob_temp[0] = i, j, t['ID'][i], t['ID'][j], prob_random, prob_binary, prob_posterior, \
                                theta, t['mu_ra'][i], t['mu_dec'][i], t['mu_ra'][j], t['mu_dec'][j], t['plx'][i], \
                                t['plx'][j]
                prob_out = np.append(prob_out, prob_temp)

                print i, j, t['ID'][i], t['ID'][j], theta*3600.0, t['mu_ra'][i], t['mu_dec'][i], t['mu_ra'][j], t['mu_dec'][j], \
                        t['plx'][i], t['plx_err'][i], t['plx'][j], t['plx_err'][j], prob_random, prob_binary, prob_posterior


    print "Elapsed time:", time.time() - start, "seconds"

    return prob_out


def calc_P_posterior(id1, id2, t, plx_prior='empirical', size_integrate_binary=1000, size_integrate_random=1000):


    pos_density = P_random.get_sigma_pos(t['ra'][id1]*np.ones(1), t['dec'][id1]*np.ones(1), catalog=t, method='sklearn_kde')
    pm_density = P_random.get_sigma_mu(t['mu_ra'][id1]*np.ones(1), t['mu_dec'][id1]*np.ones(1), catalog=t, method='sklearn_kde')


    # A few checks
    if pm_density == 0.0: return 1.0, 0.0, 1.0


    ####################### Binary Likelihood #########################
    P_binary_likelihood = P_binary.get_P_binary_convolve(id1, id2, t, size_integrate_binary, plx_prior=plx_prior)


    ####################### Random Alignment Likelihood #########################
    P_random_likelihood = P_random.get_P_random_convolve(id1, id2, t, size_integrate_random, pos_density, pm_density, plx_prior=plx_prior)


    ####################### Calculate Priors #########################
    C1_prior = P_random.get_prior_random_alignment(t['ra'][id1], t['dec'][id1], t['mu_ra'][id1], t['mu_ra'][id1], \
                                                   t, sigma_pos=pos_density, sigma_mu=pm_density)
    C2_prior = P_binary.get_prior_binary(t['ra'][id1], t['dec'][id1], t['mu_ra'][id1], t['mu_ra'][id1], \
                                                   t, sigma_pos=pos_density, sigma_mu=pm_density)

    ####################### Posterior Probability #########################
    # Save those pairs with posterior probabilities above 50%
    # return c.f_bin * prob_binary / (prob_random + c.f_bin * prob_binary), prob_random, prob_binary

    # print pos_density, pm_density, P_random.C1_prior_norm, C1_prior, C2_prior, prob_binary, prob_random

    P_normalization = C1_prior * P_random_likelihood + C2_prior * P_binary_likelihood

    if P_normalization == 0.0: return 0.0, 0.0, 0.0

    prob_posterior = C2_prior * P_binary_likelihood / P_normalization

    return prob_posterior, P_random_likelihood, P_binary_likelihood
