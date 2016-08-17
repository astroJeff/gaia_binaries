import sys
import numpy as np
from numpy.random import normal
import P_binary
import P_random
from astropy.table import Table


print "Opening Tycho-2 catalog..."
# Read in sample from Tycho-2 table
filename = ('../data/tycho-2/tyc2.dat')
readme = ('../data/tycho-2/ReadMe')
tycho_full = Table.read(filename, format='cds', guess=False, readme=readme)
print "...finished reading data."


# Create the clean tycho-2 catalog
dtype = [('ID','i8'),('ra','f8'),('dec','f8'),('mu_ra','f8'),('mu_dec','f8'), \
         ('mu_ra_err','f8'),('mu_dec_err','f8'),('Bmag','f8'),('Vmag','f8')]

ids = np.intersect1d(np.where(tycho_full['q_pmRA'] >= 0.1), np.where(tycho_full['q_pmDE'] >= 0.1))

t = np.zeros(len(ids), dtype=dtype)
t['ID'] = tycho_full['TYC1'][ids]*100000 + tycho_full['TYC2'][ids]
t['ra'] = tycho_full['RAmdeg'][ids]
t['dec'] = tycho_full['DEmdeg'][ids]
t['mu_ra'] = tycho_full['pmRA'][ids]
t['mu_dec'] = tycho_full['pmDE'][ids]
t['mu_ra_err'] = tycho_full['e_pmRA'][ids]
t['mu_dec_err'] = tycho_full['e_pmDE'][ids]
t['Bmag'] = tycho_full['BTmag'][ids]
t['Vmag'] = tycho_full['VTmag'][ids]



# Generate simulated binaries
print "Generating binaries..."
P_binary.generate_binary_set()



# Now, let's calculate the probabilities
length = len(t)
length = 2
print "We are testing", length, "stars..."

dtype = [('i_1','i4'),('i_2','i4'),('ID_1','i4'),('ID_2','i4'),('P_random','f8'),('P_binary','f8'),('P_posterior','f8')]
prob_out = np.zeros(length, dtype=dtype)

for i in np.arange(length):

    if i%100 == 0: print i

    star1 = t['ra'][i], t['dec'][i], t['mu_ra'][i], t['mu_dec'][i], t['mu_ra_err'][i], t['mu_dec_err'][i]


    # Random Alignment densities
    pos_density = P_random.get_sigma_pos(star1[0], star1[1], catalog=t, method='kde')
    pm_density = P_random.get_sigma_mu(star1[2], star1[3], catalog=t, method='kde')


    prob = np.zeros(len(t)-i-1)
    for j in np.arange(len(t)-i-1)+i+1:

        if t['ID'][i] == t['ID'][j]:
            prob[j-i-1] = 1000.0
            continue

        star2 = t['ra'][j], t['dec'][j], t['mu_ra'][j], t['mu_dec'][j], t['mu_ra_err'][j], t['mu_dec_err'][j]

        delta_pm_ra_err = np.sqrt(star1[4]**2 + star2[4]**2)
        delta_pm_dec_err = np.sqrt(star1[5]**2 + star2[5]**2)


        prob[j-i-1], P_pos, P_mu = P_random.get_P_random_alignment(star1[0], star1[1], star2[0], star2[1],
                                          star1[2], star1[3], star2[2], star2[3],
                                          delta_mu_ra_err=delta_pm_ra_err, delta_mu_dec_err=delta_pm_dec_err,
                                          pos_density=pos_density, pm_density=pm_density,
                                          catalog=t)

    # Get best matching pair
    j = np.argmin(prob)+i+1
    star2 = t['ra'][j], t['dec'][j], t['mu_ra'][j], t['mu_dec'][j], t['mu_ra_err'][j], t['mu_dec_err'][j]
    theta = P_random.get_theta_proj_degree(star1[0], star1[1], star2[0], star2[1]) * 3600.0
    delta_mu = np.sqrt((star1[2]-star2[2])**2 + (star1[3]-star2[3])**2)
    delta_mu_ra_err = np.sqrt(star1[4]**2 + star2[4]**2)
    delta_mu_dec_err = np.sqrt(star1[5]**2 + star2[5]**2)

    # Only include binary probability if star is within 1 degree
    if theta > 3600.0:
        prob_binary = 0.0
    else:
        delta_mu_ra_sample = normal(loc=(star1[2]-star2[2]), scale=delta_mu_ra_err, size=100)
        delta_mu_dec_sample = normal(loc=(star1[3]-star2[3]), scale=delta_mu_dec_err, size=100)
        delta_mu_sample = np.sqrt(delta_mu_ra_sample**2 + delta_mu_dec_sample**2)
        prob_binary = 1.0/100 * np.sum(P_binary.get_P_binary(theta, delta_mu_sample))

    prob_out['i_1'][i] = i
    prob_out['i_2'][i] = j
    prob_out['ID_1'][i] = t['ID'][i]
    prob_out['ID_2'][i] = t['ID'][j]
    prob_out['P_random'][i] = prob[j-i-1]
    prob_out['P_binary'][i] = prob_binary
    prob_out['P_posterior'][i] = prob_binary / (prob[j-i-1] + prob_binary + 1.0e-99)

print "... finished"


print prob_out
