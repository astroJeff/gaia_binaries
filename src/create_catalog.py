import numpy as np
from astropy.table import Table
import pickle
import os
import sys
import const as c


def get_moving_group_ids(t):

    idx = np.array([])
    # Pleiades
    idx = np.append(idx, moving_group(56.5, 24.1, 8.6, 20.10, -45.39, 7.4, plx_width=2.0, t=t))
    # Coma Ber
    idx = np.append(idx, moving_group(186.0, 26.0, 7.5, -11.75, -8.69, 11.53, plx_width=3.0, t=t))
    # Hyades
    idx = np.append(idx, moving_group(66.75, 15.87, 18.54, 110.0, -30.0, 21.3, mu_width=30, plx_width=5.0, t=t))
    # Praesepe
    idx = np.append(idx, moving_group(130.0, 19.7, 4.5, -35.81, -12.85, 5.49, plx_width=2.0, t=t))
    # Alpha Per
    idx = np.append(idx, moving_group(52.5, 49.0, 8.0, 22.73, -26.51, 5.8, mu_width=5.0, plx_width=2.0, t=t))
    # IC 2391
    idx = np.append(idx, moving_group(130.0, -53.1, 8.0, -24.69, 22.96, 6.90, plx_width=2.0, t=t))
    # IC 2602
    idx = np.append(idx, moving_group(160.2, -64.4, 2.9, -17.02, 11.15, 6.73, mu_width=5.0, plx_width=2.0, t=t))
    # Blanco I
    idx = np.append(idx, moving_group(1.1, -30.1, 2.5, 20.11, 2.43, 4.83, mu_width=5.0, plx_width=2.0, t=t))
    # NGC 2451
    idx = np.append(idx, moving_group(115.3, -38.5, 2.0, -21.41, 15.61, 5.45, mu_width=5.0, plx_width=2.0, t=t))
    # NGC 6475
    idx = np.append(idx, moving_group(268.4, -34.8, 3.5, 2.06, -4.98, 3.70, plx_width=2.0, t=t))
    # NGC 7092
    idx = np.append(idx, moving_group(322.9, 48.4, 3.5, -8.02, -20.36, 3.30, mu_width=5.0, plx_width=2.0, t=t))
    # NGC 2516
    idx = np.append(idx, moving_group(119.4, -60.7, 3.5, -4.17, 11.91, 2.92, mu_width=7.0, plx_width=2.0, t=t))

    return idx


def moving_group(ra, dec, radius, mu_ra, mu_dec, plx, mu_width=10.0, plx_width=4.0, t=None):
    if t is None: return
    pos_idx = ids_position(ra, dec, radius, t)
    mu_idx = ids_proper_motion(pos_idx, mu_ra, mu_dec, mu_width, t)
    plx_idx = ids_parallax(mu_idx, plx, plx_width, t)

    return plx_idx


def ids_position(ra, dec, radius, t):
    idx = np.where( (t['ra']-ra)**2*np.cos(dec*np.pi/180.)**2 + (t['dec']-dec)**2 - radius**2 < 0.0)
    if ra < radius/np.cos(dec*np.pi/180.0):
        idx = np.append(idx, np.where( (t['ra']-ra-360.0)**2*np.cos(dec*np.pi/180.)**2 + (t['dec']-dec)**2 - radius**2 < 0.0))
    return idx

def ids_proper_motion(idx, mu_ra, mu_dec, width, t):
    mu_idx = np.intersect1d(idx, np.where(np.abs((t['mu_ra']-mu_ra)**2 + (t['mu_dec']-mu_dec)**2 - width**2 < 0.0) ))
    return mu_idx

def ids_parallax(idx, plx, plx_width, t):
    idx_plx = np.intersect1d(idx, np.where(np.abs(t['plx']-plx) < 2.5))
    return idx_plx








if len(sys.argv) < 2:
     print "You must provide command line arguments"
     exit(-1)



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






######## READ IN TYCHO-HIPPARCOS MATCHING IDS ########
dtype = [('TYC','S11'), ('HIP','i8')]
tyc_hip = np.genfromtxt('../data/TGAS/tyc_hip.dat', dtype=dtype)


######## ADD TYC IDS FOR STARS WITH HIP IDS ##########
for i in np.arange(len(tgas_full)):
    if tgas_full['hip'][i] != 0:
        hip_id_diff = np.min(np.abs(tyc_hip['HIP'] - tgas_full['hip'][i]))
        if hip_id_diff == 0:
            idx = np.argmin(np.abs(tyc_hip['HIP'] - tgas_full['hip'][i]))
            tgas_full['tyc'][i] = tyc_hip['TYC'][idx]
        else:
            print "HIP id not found:", tgas_full['hip'][i], tyc_hip['TYC'][idx]





######## GENERATE IDS FOR STARS IN OPEN CLUSTERS ########
open_cluster_ids = get_moving_group_ids(t=tgas_full)




######## READ IN TGAS CATALOG WITH CROSS MATCHES ########
TGAS_combined = np.load('../data/TGAS/TGAS_combined.npy')





######### READ IN MATCHED PAIRS ###########

model = sys.argv[1]
folder = '../data/TGAS/' + sys.argv[1] + '/'


TGAS_tmp = pickle.load(open(folder + model+'_1.p', 'rb'))

TGAS = np.array([], dtype=TGAS_tmp.dtype)


for filename in os.listdir(folder):
    if filename.startswith("TGAS_") and filename.endswith(".p"):
        TGAS_tmp = pickle.load(open(folder+filename, 'rb'))
        TGAS = np.append(TGAS, TGAS_tmp)


print len(TGAS)



######### SELECT ONLY PAIRS WITH POSTERIOR PROBABILITY ABOVE 1% ############

TGAS_good = TGAS[TGAS['P_posterior'] > 0.01]


# Remove object if in listed moving group
idx = []
for i in np.arange(len(TGAS_good)):
    if TGAS_good['i_1'][i] in open_cluster_ids or TGAS_good['i_2'][i] in open_cluster_ids: continue
    idx.append(i)
TGAS_good = TGAS_good[idx]



######### CREATE NEW CATALOG ##############

dtype = [('P_posterior','f8'), ('theta','f8'),
         ('distance','<f8'), ('proj_sep','<f8'),
         ('source_id_1','<i8'), ('TYC_id_1','S11'), ('hip_id_1','<i8'),
         ('ra_1','f8'), ('dec_1','f8'),
         ('mu_ra_1','f8'), ('mu_dec_1','f8'), ('mu_ra_err_1','f8'), ('mu_dec_err_1','f8'),
         ('plx_1','f8'), ('plx_err_1','f8'),
         ('gaia_g_flux_1','<f8'), ('gaia_g_flux_err_1','<f8'), ('gaia_g_mag_1','<f8'),
         ('TMASS_id_1','<i8'), ('TMASS_angle_dist_1','<f8'),
         ('TMASS_n_neighbours_1','i8'), ('TMASS_n_mates_1','i8'), ('TMASS_ph_qual_1','S11'),
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
         ('TMASS_n_neighbours_2','i8'), ('TMASS_n_mates_2','i8'), ('TMASS_ph_qual_2','S11'),
         ('TMASS_ra_2','<f8'), ('TMASS_dec_2','<f8'),
         ('TMASS_j_mag_2','<f8'), ('TMASS_j_mag_err_2','<f8'),
         ('TMASS_h_mag_2','<f8'), ('TMASS_h_mag_err_2','<f8'),
         ('TMASS_ks_mag_2','<f8'), ('TMASS_ks_mag_err_2','<f8'),
         ('TYC_Vt_2','<f8'), ('TYC_Vt_err_2','<f8'),
         ('TYC_Bt_2','<f8'), ('TYC_Bt_err_2','<f8'),
         ('gaia_delta_Q_2','<f8'), ('gaia_noise_2','<f8')
        ]


length = len(TGAS_good)
pairs = np.zeros(length, dtype=dtype)

for i in np.arange(length):

    # Cross match with combined TGAS catalog
    idx1 = np.where(TGAS_combined['source_id'] == tgas_full['ID'][TGAS_good['i_1'][i]])
    idx2 = np.where(TGAS_combined['source_id'] == tgas_full['ID'][TGAS_good['i_2'][i]])


    # if len(idx1) > 1 or len(idx2) > 1: print i, len(idx1), len(idx2)

    pairs['P_posterior'][i] = TGAS_good['P_posterior'][i]
    pairs['theta'][i] = TGAS_good['theta'][i] * 3600.0


    # Weighted distance and projected separation
    vals = [tgas_full['plx'][TGAS_good['i_1'][i]],tgas_full['plx'][TGAS_good['i_2'][i]]]
    weights = [1.0/tgas_full['plx_err'][TGAS_good['i_1'][i]],1.0/tgas_full['plx_err'][TGAS_good['i_2'][i]]]
    pairs['distance'][i] = 1.0e3/np.average(vals, weights=weights)
    pairs['proj_sep'][i] = (pairs['theta'][i]*np.pi/180.0/3600.0) * pairs['distance'][i] * (c.pc_to_cm/c.AU_to_cm)



    pairs['source_id_1'][i] = tgas_full['ID'][TGAS_good['i_1'][i]]
    pairs['TYC_id_1'][i] = tgas_full['tyc'][TGAS_good['i_1'][i]]
    pairs['hip_id_1'][i] = tgas_full['hip'][TGAS_good['i_1'][i]]
    pairs['ra_1'][i] = tgas_full['ra'][TGAS_good['i_1'][i]]
    pairs['dec_1'][i] = tgas_full['dec'][TGAS_good['i_1'][i]]
    pairs['mu_ra_1'][i] = tgas_full['mu_ra'][TGAS_good['i_1'][i]]
    pairs['mu_dec_1'][i] = tgas_full['mu_dec'][TGAS_good['i_1'][i]]
    pairs['mu_ra_err_1'][i] = tgas_full['mu_ra_err'][TGAS_good['i_1'][i]]
    pairs['mu_dec_err_1'][i] = tgas_full['mu_dec_err'][TGAS_good['i_1'][i]]
    pairs['plx_1'][i] = tgas_full['plx'][TGAS_good['i_1'][i]]
    pairs['plx_err_1'][i] = tgas_full['plx_err'][TGAS_good['i_1'][i]]
    pairs['gaia_g_flux_1'][i] = TGAS_combined['gaia_g_flux'][idx1]
    pairs['gaia_g_flux_err_1'][i] = TGAS_combined['gaia_g_flux_error'][idx1]
    pairs['gaia_g_mag_1'][i] = TGAS_combined['gaia_g_mag'][idx1]
    pairs['TMASS_id_1'][i] = TGAS_combined['TMASS_id'][idx1]
    pairs['TMASS_angle_dist_1'][i] = TGAS_combined['TMASS_angle_dist'][idx1]
    pairs['TMASS_n_neighbours_1'][i] = TGAS_combined['TMASS_n_neighbours'][idx1]
    pairs['TMASS_n_mates_1'][i] = TGAS_combined['TMASS_n_mates'][idx1]
    if not TGAS_combined['TMASS_ph_qual'][idx1][0]:
        pairs['TMASS_ph_qual_1'][i] = '000'
    else:
        pairs['TMASS_ph_qual_1'][i] = TGAS_combined['TMASS_ph_qual'][idx1][0]
    pairs['TMASS_ra_1'][i] = TGAS_combined['TMASS_ra'][idx1]
    pairs['TMASS_dec_1'][i] = TGAS_combined['TMASS_dec'][idx1]
    pairs['TMASS_j_mag_1'][i] = TGAS_combined['TMASS_j_mag'][idx1]
    pairs['TMASS_j_mag_err_1'][i] = TGAS_combined['TMASS_j_mag_error'][idx1]
    pairs['TMASS_h_mag_1'][i] = TGAS_combined['TMASS_h_mag'][idx1]
    pairs['TMASS_h_mag_err_1'][i] = TGAS_combined['TMASS_h_mag_error'][idx1]
    pairs['TMASS_ks_mag_1'][i] = TGAS_combined['TMASS_ks_mag'][idx1]
    pairs['TMASS_ks_mag_err_1'][i] = TGAS_combined['TMASS_ks_mag_error'][idx1]
    pairs['TYC_Vt_1'][i] = TGAS_combined['TYC_Vt'][idx1]
    pairs['TYC_Vt_err_1'][i] = TGAS_combined['TYC_Vt_err'][idx1]
    pairs['TYC_Bt_1'][i] = TGAS_combined['TYC_Bt'][idx1]
    pairs['TYC_Bt_err_1'][i] = TGAS_combined['TYC_Bt_err'][idx1]
    pairs['gaia_delta_Q_1'][i] = tgas_full['d_Q'][TGAS_good['i_1'][i]]
    pairs['gaia_noise_1'][i] = tgas_full['noise'][TGAS_good['i_1'][i]]

    pairs['source_id_2'][i] = tgas_full['ID'][TGAS_good['i_2'][i]]
    pairs['TYC_id_2'][i] = tgas_full['tyc'][TGAS_good['i_2'][i]]
    pairs['hip_id_2'][i] = tgas_full['hip'][TGAS_good['i_2'][i]]
    pairs['ra_2'][i] = tgas_full['ra'][TGAS_good['i_2'][i]]
    pairs['dec_2'][i] = tgas_full['dec'][TGAS_good['i_2'][i]]
    pairs['mu_ra_2'][i] = tgas_full['mu_ra'][TGAS_good['i_2'][i]]
    pairs['mu_dec_2'][i] = tgas_full['mu_dec'][TGAS_good['i_2'][i]]
    pairs['mu_ra_err_2'][i] = tgas_full['mu_ra_err'][TGAS_good['i_2'][i]]
    pairs['mu_dec_err_2'][i] = tgas_full['mu_dec_err'][TGAS_good['i_2'][i]]
    pairs['plx_2'][i] = tgas_full['plx'][TGAS_good['i_2'][i]]
    pairs['plx_err_2'][i] = tgas_full['plx_err'][TGAS_good['i_2'][i]]
    pairs['gaia_g_flux_2'][i] = TGAS_combined['gaia_g_flux'][idx2]
    pairs['gaia_g_flux_err_2'][i] = TGAS_combined['gaia_g_flux_error'][idx2]
    pairs['gaia_g_mag_2'][i] = TGAS_combined['gaia_g_mag'][idx2]
    pairs['TMASS_id_2'][i] = TGAS_combined['TMASS_id'][idx2]
    pairs['TMASS_angle_dist_2'][i] = TGAS_combined['TMASS_angle_dist'][idx2]
    pairs['TMASS_n_neighbours_2'][i] = TGAS_combined['TMASS_n_neighbours'][idx2]
    pairs['TMASS_n_mates_2'][i] = TGAS_combined['TMASS_n_mates'][idx2]
    if not TGAS_combined['TMASS_ph_qual'][idx2][0]:
        pairs['TMASS_ph_qual_2'][i] = '000'
    else:
        pairs['TMASS_ph_qual_2'][i] = TGAS_combined['TMASS_ph_qual'][idx2][0]
    pairs['TMASS_ra_2'][i] = TGAS_combined['TMASS_ra'][idx2]
    pairs['TMASS_dec_2'][i] = TGAS_combined['TMASS_dec'][idx2]
    pairs['TMASS_j_mag_2'][i] = TGAS_combined['TMASS_j_mag'][idx2]
    pairs['TMASS_j_mag_err_2'][i] = TGAS_combined['TMASS_j_mag_error'][idx2]
    pairs['TMASS_h_mag_2'][i] = TGAS_combined['TMASS_h_mag'][idx2]
    pairs['TMASS_h_mag_err_2'][i] = TGAS_combined['TMASS_h_mag_error'][idx2]
    pairs['TMASS_ks_mag_2'][i] = TGAS_combined['TMASS_ks_mag'][idx2]
    pairs['TMASS_ks_mag_err_2'][i] = TGAS_combined['TMASS_ks_mag_error'][idx2]
    pairs['TYC_Vt_2'][i] = TGAS_combined['TYC_Vt'][idx2]
    pairs['TYC_Vt_err_2'][i] = TGAS_combined['TYC_Vt_err'][idx2]
    pairs['TYC_Bt_2'][i] = TGAS_combined['TYC_Bt'][idx2]
    pairs['TYC_Bt_err_2'][i] = TGAS_combined['TYC_Bt_err'][idx2]
    pairs['gaia_delta_Q_2'][i] = tgas_full['d_Q'][TGAS_good['i_2'][i]]
    pairs['gaia_noise_2'][i] = tgas_full['noise'][TGAS_good['i_2'][i]]




header = 'P_posterior theta distance proj_sep' + \
         'source_ID_1 TYC_ID_1 HIP_ID_1 ra_1 dec_1  mu_ra_1 mu_dec_1 mu_ra_err_1 mu_dec_err_1 plx_1 plx_err_1 ' + \
         'gaia_g_flux_1 gaia_g_flux_err_1 gaia_g_mag_1 ' + \
         '2MASS_ID_1 2MASS_angle_dist_1 2MASS_n_neighbours_1 2MASS_n_mates_1 2MASS_ph_qual_1 ' + \
         '2MASS_ra_1 2MASS_dec_1 ' + \
         '2MASS_j_mag_1 2MASS_j_mag_err_1 2MASS_h_mag_1 2MASS_h_mag_err_1 2MASS_ks_mag_1 2MASS_ks_mag_err_1 ' + \
         'Tycho_Vt_1 Tycho_Vt_err_1 Tycho_Bt_1 Tycho_Bt_err_1 ' + \
         'Gaia_delta_Q_1 Gaia_noise_1 ' + \
         'source_ID_2 TYC_ID_2 HIP_ID_2 ra_2 dec_2  mu_ra_2 mu_dec_2 mu_ra_err_2 mu_dec_err_2 plx_2 plx_err_2 ' + \
         'gaia_g_flux_2 gaia_g_flux_err_2 gaia_g_mag_2 ' + \
         '2MASS_ID_2 2MASS_angle_dist_2 2MASS_n_neighbours_2 2MASS_n_mates_2 2MASS_ph_qual_2 ' + \
         '2MASS_ra_2 2MASS_dec_2 ' + \
         '2MASS_j_mag_2 2MASS_j_mag_err_2 2MASS_h_mag_2 2MASS_h_mag_err_2 2MASS_ks_mag_2 2MASS_ks_mag_err_2 ' + \
         'Tycho_Vt_2 Tycho_Vt_err_2 Tycho_Bt_2 Tycho_Bt_err_2 ' + \
         'Gaia_delta_Q_2 Gaia_noise_2 ' + \
         '\nPositions are in degrees' + \
         '\nProper motions are in mas/yr' + \
         '\nParallaxes are in mas' + \
         '\nDistance is in pc' + \
         '\nProjected separation is in AU'


format = '%.3e %1.2f %1.3f %1.3f ' + \
         '%i %s %i %1.9f %1.9f %1.3f %1.3f %1.3f %1.3f %1.3f %1.3f %1.3e %1.3e %1.2f ' + \
         '%i %1.3f %i %i %s %1.9f %1.9f %1.2f %1.2f %1.2f %1.2f %1.2f %1.2f %1.2f %1.2f %1.2f %1.2f %1.2f %1.2f ' + \
         '%i %s %i %1.9f %1.9f %1.3f %1.3f %1.3f %1.3f %1.3f %1.3f %1.3e %1.3e %1.2f ' + \
         '%i %1.3f %i %i %s %1.9f %1.9f %1.2f %1.2f %1.2f %1.2f %1.2f %1.2f %1.2f %1.2f %1.2f %1.2f %1.2f %1.2f'

np.savetxt('../data/TGAS/'+'gaia_wide_binaries_'+model+'_cleaned.txt', pairs, delimiter=' ', header=header, fmt=format)
