import numpy as np
from numpy.random import normal, multivariate_normal
import P_binary
import P_random
from astropy.table import Table
import pickle
import time



size_integrate = 10          # Number of samples for delta mu integration for initial search
size_integrate_full = 1000  # Number of samples for delta mu integration for possible matches


def match_binaries(t):
    """ Function to match binaries within a catalog

    Arguments
    ---------
    t : ndarray
        Catalog for self-compare for matching

    Returns
    -------
    prob_out : ndarray
        Set of matched pairs, their IDs, and their probabilities

    """

    # Start time
    start = time.time()


    # Generate simulated binaries
    print "Generating binaries..."
    P_binary.generate_binary_set(num_sys=100000)


    # Generate random alignment KDEs using first entry as a test position
    P_random.mu_kde = None
    P_random.pos_kde = None
    pos_density = P_random.get_sigma_pos(t['ra'][0], t['dec'][0], catalog=t, method='kde')
    pm_density = P_random.get_sigma_mu(t['mu_ra'][0], t['mu_dec'][0], catalog=t, method='kde')


    # Now, let's calculate the probabilities
    length = len(t)
    print "We are testing", length, "stars..."

    dtype = [('i_1','i4'),('i_2','i4'),('ID_1','i4'),('ID_2','i4'),('P_random','f8'),('P_binary','f8'),('P_posterior','f8')]
    prob_out = np.array([], dtype=dtype)


    for i in np.arange(length):

        if i%1000 == 0: print i, time.time()-start


        # Get ids of all stars within 1 degree
        i_star2 = np.arange(length - i - 1) + i + 1
        theta = P_random.get_theta_proj_degree(t['ra'][i], t['dec'][i], t['ra'][i_star2], t['dec'][i_star2])
        ids_good = i_star2[np.where(theta < 1.0)[0]]


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
        mu_diff_vector = np.amax(np.vstack([mu_diff_3sigma, 0.1*np.ones(len(ids_good))]), axis=0)
        ids_good_binary = np.where(P_binary.get_P_binary(theta_good * 3600.0, mu_diff_vector))[0]
        if len(ids_good_binary) == 0: continue
        ids_good_binary_all = ids_good[ids_good_binary]



        # More precise integration for potential matches
        for k in np.arange(len(ids_good_binary_all)):

            # IDs for the secondary
            j = ids_good_binary_all[k]

            # Star arrays
            star1 = t['ra'][i], t['dec'][i], t['mu_ra'][i], t['mu_dec'][i], t['mu_ra_err'][i], t['mu_dec_err'][i]
            star2 = t['ra'][j], t['dec'][j], t['mu_ra'][j], t['mu_dec'][j], t['mu_ra_err'][j], t['mu_dec_err'][j]


            prob_posterior = calc_P_posterior(star1, star2, pos_density, mu_density, i, j, catalog=t)


            print i, j, t['ID'][i], t['ID'][j], prob_random, prob_binary, prob_posterior

            if prob_posterior > 0.5:
                prob_temp = np.zeros(1, dtype=dtype)
                prob_temp[0] = i, j, t['ID'][i], t['ID'][j], prob_random, prob_binary, prob_posterior
                prob_out = np.append(prob_out, prob_temp)


    print "Elapsed time:", time.time() - start, "seconds"

    return prob_out


def calc_P_posterior(star1, star2, pos_density, mu_density, id1, id2, catalog=t):


    ####################### Binary Likelihood #########################
    # Angular separation
    theta = P_random.get_theta_proj_degree(t['ra'][id1], t['dec'][id1], t['ra'][id2], t['dec'][id2])


    # Proper motion uncertainties
    delta_mu_ra_err = np.sqrt(t['mu_ra_err'][id1]**2 + t['mu_ra_err'][id2]**2)
    delta_mu_dec_err = np.sqrt(t['mu_dec_err'][id1]**2 + t['mu_dec_err'][id2]**2)


    # Recalculate binary probabilities
    delta_mu_ra_sample = normal(loc=(t['mu_ra'][id1] - t['mu_ra'][id2]), \
                                             scale=delta_mu_ra_err, \
                                             size=size_integrate_full)
    delta_mu_dec_sample = normal(loc=(t['mu_dec'][id1] - t['mu_dec'][id2]), \
                                              scale=delta_mu_dec_err, \
                                              size=size_integrate_full)
    delta_mu_sample = np.sqrt(delta_mu_ra_sample**2 + delta_mu_dec_sample**2)

    # Generate random parallaxes from uncertainties in primary star
    plx_sample = normal(loc=star1['plx'], scale=star1['plx_err'], \
                                              size=size_integrate_full)

    # Distance in pc is just parallax in asec
    dist_sample = plx_sample * 1.0e3 # convert from mas to asec

    # Convert from proper motion difference (mas/yr) to transverse velocity difference (km/s)
    delta_v_trans = (delta_mu_sample/1.0e3/3600.0*np.pi/180.0) * dist_sample * (c.pc_to_cm/1.0e5) / (c.yr_to_sec)

    # Find the physical separation (Rsun) from the angular separation (degree)
    proj_sep = (theta*np.pi/180.0) * dist_sample * (c.pc_to_cm / c.Rsun_to_cm)

    # Find binary probabilities
    prob_tmp = P_binary.get_P_binary(proj_sep, delta_v_trans)

    # Now, let's add parallax probabilities
    


    prob_binary = 1.0/size_integrate_full * np.sum(prob_tmp)



    ####################### Random Alignment Likelihood #########################
    # Random Alignment densities
    pos_density = P_random.get_sigma_pos(t['ra'][id1], t['dec'][id1], catalog=t, method='kde')
    pm_density = P_random.get_sigma_mu(t['mu_ra'][id1], t['mu_dec'][id1], catalog=t, method='kde')


    # Calculate random alignment probabilities
    prob_random, prob_pos, prob_mu = P_random.get_P_random_alignment(star1[0], star1[1], star2[0], star2[1],
                                      star1[2], star1[3], star2[2], star2[3],
                                      delta_mu_ra_err=delta_mu_ra_err, delta_mu_dec_err=delta_mu_dec_err,
                                      pos_density=pos_density, pm_density=pm_density,
                                      catalog=t)



    ####################### Posterior Probability #########################
    # Save those pairs with posterior probabilities above 50%
    return c.f_bin * prob_binary / (prob_random + c.f_bin * prob_binary)
