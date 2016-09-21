import os
import sys
sys.path.append('../src')
import P_binary
import P_random
import P_posterior
import const as c
import time
import numpy as np
from astropy.table import Table
import pickle



# Read in sample from Tycho-2 table
filename = ('../data/hipparchos/hip2.dat')
readme = ('../data/hipparchos/ReadMe')
hip_orig = Table.read(filename, format='cds', guess=False, readme=readme)



# Save data to a usable array
dtype = [('ID','i8'), ('ra','f8'), ('dec','f8'), ('mu_ra','f8'), ('mu_dec','f8'), \
         ('mu_ra_err','f8'), ('mu_dec_err','f8'), ('plx','f8'), ('plx_err','f8')]
hip = np.zeros(len(hip_orig), dtype=dtype)

hip['ID'] = hip_orig['HIP']
hip['ra'] = hip_orig['RArad'] * c.rad_to_deg
hip['dec'] = hip_orig['DErad'] * c.rad_to_deg
hip['mu_ra'] = hip_orig['pmRA']
hip['mu_dec'] = hip_orig['pmDE']
hip['mu_ra_err'] = hip_orig['e_pmRA']
hip['mu_dec_err'] = hip_orig['e_pmDE']
hip['plx'] = hip_orig['Plx']
hip['plx_err'] = hip_orig['e_Plx']



# Run matching
p_out = P_posterior.match_binaries(hip)



pickle.dump(p_out, open("../data/hipparchos.p","wb"))





