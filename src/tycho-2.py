import numpy as np
from astropy.table import Table
import pickle
import sys
sys.path.append('../src')
import P_posterior


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



# Run search for wide binaries
p_out = P_posterior.match_binaries(t)


# Save resulting array
pickle.dump(p_out, open('../data/tycho-2/prob_out_full.data', "wb"))




