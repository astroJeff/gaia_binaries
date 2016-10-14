import numpy as np
from astropy.table import Table
import pickle
import os



######## READ IN TGAS CATALOG ############

# Read in sample from TGAS table
dtype = [('ID','<i8'),('tyc','S11'),('hip','<i8'),('ra','<f8'),('dec','<f8'),('mu_ra','<f8'),('mu_dec','<f8'), \
     ('mu_ra_err','<f8'),('mu_dec_err','<f8'),('plx','<f8'),('plx_err','<f8'), \
     ('mu_ra_mu_dec_cov','<f8'),('mu_ra_plx_cov','<f8'),('mu_dec_plx_cov','<f8'),('d_Q','<f8'),('noise','<f8')]

tgas_full = np.array([], dtype=dtype)


for i in np.arange(16):
    if i < 10:
        filename = ('../data/TGAS/TgasSource_000-000-00' + str(i) + '.csv')
    else:
        filename = ('../data/TGAS/TgasSource_000-000-0' + str(i) + '.csv')

    print filename
    tgas_tmp = Table.read(filename, format='csv', guess=True)



    tgas = np.zeros(len(tgas_tmp), dtype=dtype)
                                                  # Gaia units
    tgas['ID'] = tgas_tmp['source_id']
    tgas['tyc'] = tgas_tmp['tycho2_id']
    tgas['hip'] = tgas_tmp['hip']
    tgas['ra'] = tgas_tmp['ra']                   # degrees
    tgas['dec'] = tgas_tmp['dec']                 # degrees
    tgas['mu_ra'] = tgas_tmp['pmra']              # mas/yr
    tgas['mu_ra_err'] = tgas_tmp['pmra_error']    # mas/yr
    tgas['mu_dec'] = tgas_tmp['pmdec']            # mas/yr
    tgas['mu_dec_err'] = tgas_tmp['pmdec_error']  # mas/yr
    tgas['plx'] = tgas_tmp['parallax']            # mas
    tgas['plx_err'] = tgas_tmp['parallax_error']  # mas
    tgas['mu_ra_mu_dec_cov'] = tgas_tmp['pmra_pmdec_corr']*tgas_tmp['pmra_error']*tgas_tmp['pmdec_error']
    tgas['mu_ra_plx_cov'] = tgas_tmp['parallax_pmra_corr']*tgas_tmp['parallax_error']*tgas_tmp['pmra_error']
    tgas['mu_dec_plx_cov'] = tgas_tmp['parallax_pmdec_corr']*tgas_tmp['parallax_error']*tgas_tmp['pmdec_error']
    tgas['d_Q'] = tgas_tmp['astrometric_delta_q']
    tgas['noise'] = tgas_tmp['astrometric_excess_noise_sig']

    tgas_full = np.append(tgas_full, tgas)





######### READ IN MATCHED PAIRS ###########

folder = '../data/TGAS/'

TGAS_tmp = pickle.load(open(folder + 'TGAS_match_1.p', 'rb'))

TGAS = np.array([], dtype=TGAS_tmp.dtype)


for filename in os.listdir(folder):
    if filename.startswith("TGAS_") and filename.endswith(".p"):
        TGAS_tmp = pickle.load(open(folder+filename, 'rb'))
        TGAS = np.append(TGAS, TGAS_tmp)






######### SELECT ONLY PAIRS WITH POSTERIOR PROBABILITY ABOVE 1% ############

TGAS_good = TGAS[TGAS['P_posterior'] > 0.01]






######### CREATE NEW CATALOG ##############

dtype = [('source_id_1','f8'), ('ra_1','f8'), ('dec_1','f8'), \
         ('source_id_2','f8'), ('ra_2','f8'), ('dec_2','f8'), \
         ('mu_ra_1','f8'), ('mu_dec_1','f8'), ('mu_ra_err_1','f8'), ('mu_dec_err_1','f8'), \
         ('mu_ra_2','f8'), ('mu_dec_2','f8'), ('mu_ra_err_2','f8'), ('mu_dec_err_2','f8'),
         ('plx_1','f8'), ('plx_1_err','f8'), ('plx_2','f8'), ('plx_err_2','f8'), \
         ('P_posterior','f8'), ('theta','f8')]

pairs = np.zeros(len(TGAS_good), dtype=dtype)


for i in np.arange(len(TGAS_good)):

    pairs['source_id_1'][i] = TGAS_good['ID_1'][i]
    pairs['source_id_2'][i] = TGAS_good['ID_2'][i]
    pairs['ra_1'][i] = tgas_full['ra'][TGAS_good['i_1'][i]]
    pairs['dec_1'][i] = tgas_full['dec'][TGAS_good['i_1'][i]]
    pairs['ra_2'][i] = tgas_full['ra'][TGAS_good['i_2'][i]]
    pairs['dec_2'][i] = tgas_full['dec'][TGAS_good['i_2'][i]]
    pairs['mu_ra_1'][i] = tgas_full['mu_ra'][TGAS_good['i_1'][i]]
    pairs['mu_dec_1'][i] = tgas_full['mu_dec'][TGAS_good['i_1'][i]]
    pairs['mu_ra_err_1'][i] = tgas_full['mu_ra_err'][TGAS_good['i_1'][i]]
    pairs['mu_dec_err_1'][i] = tgas_full['mu_dec_err'][TGAS_good['i_1'][i]]
    pairs['mu_ra_2'][i] = tgas_full['mu_ra'][TGAS_good['i_2'][i]]
    pairs['mu_dec_2'][i] = tgas_full['mu_dec'][TGAS_good['i_2'][i]]
    pairs['mu_ra_err_2'][i] = tgas_full['mu_ra_err'][TGAS_good['i_2'][i]]
    pairs['mu_dec_err_2'][i] = tgas_full['mu_dec_err'][TGAS_good['i_2'][i]]
    pairs['plx_1'][i] = tgas_full['plx'][TGAS_good['i_1'][i]]
    pairs['plx_err_1'][i] = tgas_full['plx_err'][TGAS_good['i_1'][i]]
    pairs['plx_2'][i] = tgas_full['plx'][TGAS_good['i_2'][i]]
    pairs['plx_err_2'][i] = tgas_full['plx_err'][TGAS_good['i_2'][i]]
    pairs['P_posterior'][i] = TGAS_good['P_posterior'][i]
    pairs['theta'][i] = TGAS_good['theta'][i]




header = 'source_ID_1 source_ID_2 ra_1 dec_1 ra_2 dec_2 mu_ra_1 mu_dec_1 mu_ra_err_1 mu_dec_err_1' + \
         'mu_ra_2 mu_dec_2 mu_ra_err_2 mu_dec_err_2 plx_1 plx_err_1 plx_2 plx_err_2' + \
         'P_posterior theta'
np.savetxt('../data/TGAS/gaia_wide_binaries.txt', pairs, delimiter=' ', header=header)
