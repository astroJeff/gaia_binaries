import numpy as np
import P_posterior
from astropy.table import Table
import pickle


# Read in the revised NLTT table
filename = ('../data/rNLTT/catalog_tabs.dat')
NLTT_full = Table.read(filename, format='ascii', guess=True)

# Change proper motion units from asec/yr to mas/yr
NLTT_full['mu_ra'] = 1.0e3*NLTT_full['mu_ra']
NLTT_full['mu_dec'] = 1.0e3*NLTT_full['mu_dec']
NLTT_full['mu_ra_err'] = 1.0e3*NLTT_full['mu_ra_err']
NLTT_full['mu_dec_err'] = 1.0e3*NLTT_full['mu_dec_err']

# Select only systems with proper motion errors above 0.1 mas/yr
ids = np.intersect1d(np.where(NLTT_full['mu_ra_err'] >= 0.1), np.where(NLTT_full['mu_dec_err'] >= 0.1))

# Create the clean NLTT catalog
dtype = [('ID','i8'),('ra','f8'),('dec','f8'),('mu_ra','f8'),('mu_dec','f8'), \
         ('mu_ra_err','f8'),('mu_dec_err','f8'),('B','f8'),('V','f8')]

t = np.zeros(len(ids), dtype=dtype)
t['ID'] = NLTT_full['NLTT'][ids]
t['ra'] = NLTT_full['ra'][ids]
t['dec'] = NLTT_full['dec'][ids]
t['mu_ra'] = NLTT_full['mu_ra'][ids]
t['mu_dec'] = NLTT_full['mu_dec'][ids]
t['mu_ra_err'] = NLTT_full['mu_ra_err'][ids]
t['mu_dec_err'] = NLTT_full['mu_dec_err'][ids]
t['B'] = NLTT_full['B'][ids]
t['V'] = NLTT_full['V'][ids]


# Run search for wide binaries
p_out = P_posterior.match_binaries(t)


# Save resulting array
pickle.dump(p_out, open('../data/rNLTT/prob_out.data', "wb"))
