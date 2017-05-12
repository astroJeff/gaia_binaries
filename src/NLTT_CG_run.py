import os
import P_binary
import P_random
import P_posterior
import const as c
import time
import glob
import numpy as np
from numpy.random import uniform, normal
from scipy.optimize import newton
from scipy import stats
from astropy.table import Table
from astropy import units as u
from astropy.coordinates import SkyCoord
from scipy.interpolate import interp1d, interp2d
from scipy.stats import gaussian_kde
import pickle







# Read in all pairs
# Read in sample from Tycho-2 table
filename = ('../data/rNLTT/catalog_tabs.dat')
t_full = Table.read(filename, format='ascii', guess=True)

# Change proper motion units from asec/yr to mas/yr
t_full['mu_ra'] = 1.0e3*t_full['mu_ra']
t_full['mu_dec'] = 1.0e3*t_full['mu_dec']
t_full['mu_ra_err'] = 1.0e3*t_full['mu_ra_err']
t_full['mu_dec_err'] = 1.0e3*t_full['mu_dec_err']

# Select only stars with proper motion uncertainties greater than 1 mas/yr - remove junk
ids_good = np.union1d( np.where(t_full['mu_ra_err'] > 1.0)[0], \
                      np.where(t_full['mu_dec_err'] > 1.0)[0] )
t = t_full[ids_good]






# Read in pairs identified by Chaname & Gould

filename_CG = ('../data/rNLTT/Chaname_Gould_pairs.dat')
CG_pairs = Table.read(filename_CG, format='cds')




n_mu_sample = 1000

dtype = [('NLTT_1','i4'),('NLTT_2','i4'),('P_random','f8'),('P_binary','f8'),\
         ('P_theta','f8'),('P_mu','f8'),('P_posterior','f8'),('CCode','i4')]
prob_out_CG = np.array([], dtype=dtype)

for pair in CG_pairs:

    if len(prob_out_CG)%100 == 0: print(len(prob_out_CG))

#    print pair['NLTT-A'], pair['NLTT-B'], pair['CCode']

    # Check if there is a match for star 1
    if len(np.where(pair['NLTT-A'] == t_full['NLTT'])[0]) == 0:
        continue

    # Check if there is a match for star 2
    if len(np.where(pair['NLTT-B'] == t_full['NLTT'])[0]) == 0:
        continue

    # Star 1
    star1 = t_full['ra'][np.where(pair['NLTT-A'] == t_full['NLTT'])][0], \
        t_full['dec'][np.where(pair['NLTT-A'] == t_full['NLTT'])][0], \
        t_full['mu_ra'][np.where(pair['NLTT-A'] == t_full['NLTT'])][0], \
        t_full['mu_dec'][np.where(pair['NLTT-A'] == t_full['NLTT'])][0], \
        t_full['mu_ra_err'][np.where(pair['NLTT-A'] == t_full['NLTT'])][0], \
        t_full['mu_dec_err'][np.where(pair['NLTT-A'] == t_full['NLTT'])][0]

    # Star 2
    star2 = t_full['ra'][np.where(pair['NLTT-B'] == t_full['NLTT'])][0], \
        t_full['dec'][np.where(pair['NLTT-B'] == t_full['NLTT'])][0], \
        t_full['mu_ra'][np.where(pair['NLTT-B'] == t_full['NLTT'])][0], \
        t_full['mu_dec'][np.where(pair['NLTT-B'] == t_full['NLTT'])][0], \
        t_full['mu_ra_err'][np.where(pair['NLTT-B'] == t_full['NLTT'])][0], \
        t_full['mu_dec_err'][np.where(pair['NLTT-B'] == t_full['NLTT'])][0]


    delta_pm_ra_err = np.sqrt(star1[4]**2 + star2[4]**2)
    delta_pm_dec_err = np.sqrt(star1[5]**2 + star2[5]**2)


    # If uncertainties are zero
    if delta_pm_ra_err == 0.0 or delta_pm_dec_err == 0.0:
        prob_temp = np.zeros(1, dtype=dtype)
        prob_temp['NLTT_1'][0] = pair['NLTT-A']
        prob_temp['NLTT_2'][0] = pair['NLTT-B']
        prob_temp['P_random'][0] = -1
        prob_temp['P_binary'][0] = -1
        prob_temp['P_theta'][0] = -1
        prob_temp['P_mu'][0] = -1
        prob_temp['P_posterior'][0] = -1
        prob_temp['CCode'][0] = pair['CCode']
        prob_out_CG = np.append(prob_out_CG, prob_temp[0])

        continue




    # Get probability of random alignment
    prob_random, P_pos, P_mu = P_random.get_P_random_alignment(star1[0], star1[1], star2[0], star2[1],
                                          star1[2], star1[3], star2[2], star2[3],
                                          delta_mu_ra_err=delta_pm_ra_err, delta_mu_dec_err=delta_pm_dec_err,
                                          catalog=t)

    # Get probability due to true binary
    theta = P_random.get_theta_proj_degree(star1[0], star1[1], star2[0], star2[1]) * 3600.0
    delta_mu = np.sqrt((star1[2]-star2[2])**2 + (star1[3]-star2[3])**2)
    delta_mu_ra_err = np.sqrt(star1[4]**2 + star2[4]**2)
    delta_mu_dec_err = np.sqrt(star1[5]**2 + star2[5]**2)
    if theta > 3600.0:
        prob_binary = 0.0
    else:
        delta_mu_ra_sample = normal(loc=(star1[2]-star2[2]), scale=delta_mu_ra_err, size=n_mu_sample)
        delta_mu_dec_sample = normal(loc=(star1[3]-star2[3]), scale=delta_mu_dec_err, size=n_mu_sample)
        delta_mu_sample = np.sqrt(delta_mu_ra_sample**2 + delta_mu_dec_sample**2)
        prob_binary = 1.0/n_mu_sample * np.sum(P_binary.get_P_binary(theta, delta_mu_sample))

    prob_temp = np.zeros(1, dtype=dtype)
    prob_temp['NLTT_1'][0] = pair['NLTT-A']
    prob_temp['NLTT_2'][0] = pair['NLTT-B']
    prob_temp['P_random'][0] = prob_random
    prob_temp['P_binary'][0] = prob_binary
    prob_temp['P_theta'][0] = P_pos
    prob_temp['P_mu'][0] = P_mu
    prob_temp['P_posterior'][0] = P_posterior.f_bin * prob_binary / (prob_random + P_posterior.f_bin * prob_binary)
    prob_temp['CCode'][0] = pair['CCode']

    prob_out_CG = np.append(prob_out_CG, prob_temp[0])



pickle.dump(prob_out_CG, open("../data/rNLTT/prob_CG_100pc.data", "wb"))
