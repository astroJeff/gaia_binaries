import numpy as np
from astropy.table import Table
import time




# Read in sample from TGAS table
dtype = [('source_id','<i8'),('tyc','S11'),('hip','<i8'),
         ('ra','<f8'),('dec','<f8'),
         ('mu_ra','<f8'),('mu_dec','<f8'), ('mu_ra_err','<f8'),('mu_dec_err','<f8'),
         ('plx','<f8'),('plx_err','<f8'),
         ('gaia_g_flux','<f8'), ('gaia_g_flux_error','<f8'), ('gaia_g_mag','<f8')
        ]

tgas_full = np.array([], dtype=dtype)

for i in np.arange(16):
    if i < 10:
        filename = ('../data/TGAS/TgasSource_000-000-00' + str(i) + '.csv')
    else:
        filename = ('../data/TGAS/TgasSource_000-000-0' + str(i) + '.csv')

    print(filename)
    tgas_tmp = Table.read(filename, format='csv', guess=True)



    tgas = np.zeros(len(tgas_tmp), dtype=dtype)

    tgas['source_id'] = tgas_tmp['source_id']
    tgas['tyc'] = tgas_tmp['tycho2_id']
    tgas['hip'] = tgas_tmp['hip']
    tgas['ra'] = tgas_tmp['ra']
    tgas['dec'] = tgas_tmp['dec']
    tgas['mu_ra'] = tgas_tmp['pmra']
    tgas['mu_ra_err'] = tgas_tmp['pmra_error']
    tgas['mu_dec'] = tgas_tmp['pmdec']
    tgas['mu_dec_err'] = tgas_tmp['pmdec_error']
    tgas['plx'] = tgas_tmp['parallax']
    tgas['plx_err'] = tgas_tmp['parallax_error']
    tgas['gaia_g_flux'] = tgas_tmp['phot_g_mean_flux']
    tgas['gaia_g_flux_error'] = tgas_tmp['phot_g_mean_flux_error']
    tgas['gaia_g_mag'] = tgas_tmp['phot_g_mean_mag']

    tgas_full = np.append(tgas_full, tgas)


# # URAT Catalog
# filename = ('../data/TGAS/TGAS_URAT.vot')
# URAT = Table.read(filename, format='votable')

# 2MASS Catalog
filename = ('../data/TGAS/TGAS_2MASS.vot')
TWO_MASS = Table.read(filename, format='votable')

# # ppmxl Catalog
# filename = ('../data/TGAS/TGAS_ppmxl.vot')
# ppmxl = Table.read(filename, format='votable')

# Tycho-2 Catalog
filename = ('../data/tycho-2/tyc2.dat')
readme = ('../data/tycho-2/ReadMe')
TYC = Table.read(filename, format='cds', guess=False, readme=readme)
TYC_ID = np.zeros(len(TYC), dtype='S11')
for i in np.arange(len(TYC)):
    TYC_ID[i] = str(TYC['TYC1'][i]) + '-' + str(TYC['TYC2'][i]) + '-' + str(TYC['TYC3'][i])


# Now, match the catalogs to create a combined master
# Read in sample from TGAS table
dtype = [('source_id','<i8'),('tyc','S11'),('hip','<i8'),
         ('ra','<f8'),('dec','<f8'),
         ('mu_ra','<f8'),('mu_dec','<f8'), ('mu_ra_err','<f8'),('mu_dec_err','<f8'),
         ('plx','<f8'),('plx_err','<f8'),
         ('gaia_g_flux','<f8'), ('gaia_g_flux_error','<f8'),
         ('gaia_g_mag','<f8'),
        #  ('URAT_id','<i8'), ('URAT_angle_dist','<f8'),
        #  ('URAT_ra','<f8'), ('URAT_dec','<f8'),
        #  ('URAT_f_mag','<f8'), ('URAT_f_mag_error','<f8'),
        #  ('URAT_b_mag','<f8'), ('URAT_b_mag_error','<f8'),
        #  ('URAT_v_mag','<f8'), ('URAT_v_mag_error','<f8'),
        #  ('URAT_g_mag','<f8'), ('URAT_g_mag_error','<f8'),
        #  ('URAT_r_mag','<f8'), ('URAT_r_mag_error','<f8'),
        #  ('URAT_i_mag','<f8'), ('URAT_i_mag_error','<f8'),
         ('TMASS_id','<i8'), ('TMASS_angle_dist','<f8'),
         ('TMASS_n_neighbours','i8'), ('TMASS_n_mates','i8'), ('TMASS_ph_qual','S11'),
         ('TMASS_ra','<f8'), ('TMASS_dec','<f8'),
         ('TMASS_j_mag','<f8'), ('TMASS_j_mag_error','<f8'),
         ('TMASS_h_mag','<f8'), ('TMASS_h_mag_error','<f8'),
         ('TMASS_ks_mag','<f8'), ('TMASS_ks_mag_error','<f8'),
         ('TYC_Vt','<f8'), ('TYC_Vt_err','<f8'),
         ('TYC_Bt','<f8'), ('TYC_Bt_err','<f8')
        #  ('ppmxl_id','<i8'), ('ppmxl_angle_dist','<f8'),
        #  ('ppmxl_ra','<f8'), ('ppmxl_dec','<f8'),
        #  ('ppmxl_b1_mag','<f8'), ('ppmxl_b2_mag','<f8'),
        #  ('ppmxl_r1_mag','<f8'), ('ppmxl_i_mag','<f8')
        ]


num = len(tgas_full)
tgas_full_combined = np.zeros(num, dtype=dtype)

start = time.time()


for i in np.arange(num):

    if i%1000 == 0: print(i, time.time()-start)


    # idx1 = np.where(tgas_full[i]['source_id'] == URAT['source_id'])
    idx2 = np.where(tgas_full[i]['source_id'] == TWO_MASS['source_id'])
    # idx3 = np.where(tgas_full[i]['source_id'] == ppmxl['source_id'])
    idx4 = np.where(tgas_full[i]['tyc'] == TYC_ID)


    tgas_full_combined[i]['source_id'] = tgas_full[i]['source_id']
    tgas_full_combined[i]['tyc'] = tgas_full[i]['tyc']
    tgas_full_combined[i]['hip'] = tgas_full[i]['hip']
    tgas_full_combined[i]['ra'] = tgas_full[i]['ra']
    tgas_full_combined[i]['dec'] = tgas_full[i]['dec']
    tgas_full_combined[i]['mu_ra'] = tgas_full[i]['mu_ra']
    tgas_full_combined[i]['mu_ra_err'] = tgas_full[i]['mu_ra_err']
    tgas_full_combined[i]['mu_dec'] = tgas_full[i]['mu_dec']
    tgas_full_combined[i]['mu_dec_err'] = tgas_full[i]['mu_dec_err']
    tgas_full_combined[i]['plx'] = tgas_full[i]['plx']
    tgas_full_combined[i]['plx_err'] = tgas_full[i]['plx_err']
    tgas_full_combined[i]['gaia_g_flux'] = tgas_full[i]['gaia_g_flux']
    tgas_full_combined[i]['gaia_g_flux_error'] = tgas_full[i]['gaia_g_flux_error']
    tgas_full_combined[i]['gaia_g_mag'] = (tgas_full[i]['gaia_g_mag'])


    #
    # if len(idx1[0] > 0):
    #     tgas_full_combined[i]['URAT_id'] = URAT[idx1]['urat1_oid']
    #     tgas_full_combined[i]['URAT_angle_dist'] = URAT[idx1]['angular_distance']
    #     tgas_full_combined[i]['URAT_ra'] = URAT[idx1]['ra']
    #     tgas_full_combined[i]['URAT_dec'] = URAT[idx1]['dec']
    #     tgas_full_combined[i]['URAT_f_mag'] = URAT[idx1]['f_mag']
    #     tgas_full_combined[i]['URAT_f_mag_error'] = URAT[idx1]['f_mag_error']
    #     tgas_full_combined[i]['URAT_b_mag'] = URAT[idx1]['b_mag']
    #     tgas_full_combined[i]['URAT_b_mag_error'] = URAT[idx1]['b_mag_error']
    #     tgas_full_combined[i]['URAT_v_mag'] = URAT[idx1]['v_mag']
    #     tgas_full_combined[i]['URAT_v_mag_error'] = URAT[idx1]['v_mag_error']
    #     tgas_full_combined[i]['URAT_g_mag'] = URAT[idx1]['g_mag']
    #     tgas_full_combined[i]['URAT_g_mag_error'] = URAT[idx1]['g_mag_error']
    #     tgas_full_combined[i]['URAT_r_mag'] = URAT[idx1]['r_mag']
    #     tgas_full_combined[i]['URAT_r_mag_error'] = URAT[idx1]['r_mag_error']
    #     tgas_full_combined[i]['URAT_i_mag'] = URAT[idx1]['i_mag']
    #     tgas_full_combined[i]['URAT_i_mag_error'] = URAT[idx1]['i_mag_error']


    if len(idx2[0] > 0):
        tgas_full_combined[i]['TMASS_id'] = TWO_MASS[idx2]['tmass_oid']
        tgas_full_combined[i]['TMASS_angle_dist'] = TWO_MASS[idx2]['angular_distance']
        tgas_full_combined[i]['TMASS_n_neighbours'] = TWO_MASS[idx2]['best_neighbour_multiplicity']
        tgas_full_combined[i]['TMASS_n_mates'] = TWO_MASS[idx2]['number_of_mates']
        tgas_full_combined[i]['TMASS_ph_qual'] = TWO_MASS[idx2]['ph_qual'][0]
        tgas_full_combined[i]['TMASS_ra'] = TWO_MASS[idx2]['ra']
        tgas_full_combined[i]['TMASS_dec'] = TWO_MASS[idx2]['dec']
        tgas_full_combined[i]['TMASS_j_mag'] = TWO_MASS[idx2]['j_m']
        tgas_full_combined[i]['TMASS_j_mag_error'] = TWO_MASS[idx2]['j_msigcom']
        tgas_full_combined[i]['TMASS_h_mag'] = TWO_MASS[idx2]['h_m']
        tgas_full_combined[i]['TMASS_h_mag_error'] = TWO_MASS[idx2]['h_msigcom']
        tgas_full_combined[i]['TMASS_ks_mag'] = TWO_MASS[idx2]['ks_m']
        tgas_full_combined[i]['TMASS_ks_mag_error'] = TWO_MASS[idx2]['ks_msigcom']

    # if len(idx3[0] > 0):
    #     tgas_full_combined[i]['ppmxl_id'] = ppmxl[idx3]['ppmxl_oid']
    #     tgas_full_combined[i]['ppmxl_angle_dist'] = ppmxl[idx3]['angular_distance']
    #     tgas_full_combined[i]['ppmxl_ra'] = ppmxl[idx3]['ra']
    #     tgas_full_combined[i]['ppmxl_dec'] = ppmxl[idx3]['dec']
    #     tgas_full_combined[i]['ppmxl_b1_mag'] = ppmxl[idx3]['b1mag']
    #     tgas_full_combined[i]['ppmxl_b2_mag'] = ppmxl[idx3]['b2mag']
    #     tgas_full_combined[i]['ppmxl_r1_mag'] = ppmxl[idx3]['r1mag']
    #     tgas_full_combined[i]['ppmxl_i_mag'] = ppmxl[idx3]['imag']

    if len(idx4[0] > 0):
        tgas_full_combined[i]['TYC_Vt'] = TYC['VTmag'][idx4]
        tgas_full_combined[i]['TYC_Vt_err'] = TYC['e_VTmag'][idx4]
        tgas_full_combined[i]['TYC_Bt'] = TYC['BTmag'][idx4]
        tgas_full_combined[i]['TYC_Bt_err'] = TYC['e_BTmag'][idx4]


# Save the catalog to a np array
np.save('../data/TGAS/TGAS_combined', tgas_full_combined)


print("Elapsed time:", time.time() - start, "seconds")
