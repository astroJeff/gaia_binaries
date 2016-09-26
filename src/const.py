import numpy as np

GGG = 6.674e-8                    # Gravitational constant in cgs
alpha = -2.35                     # IMF index
day_to_sec = 3600.0*24.0          # Sec in day
yr_to_sec = 365.25*3600.0*24.0    # Sec in year
Rsun_to_cm = 6.995e10             # Rsun to cm
Msun_to_g = 1.989e33              # Msun to g
AU_to_cm = 1.496e13               # AU to cm
pc_to_cm = 3.086e18               # pc to cm
deg_to_rad = 0.0174532925199 # Degrees to radians
rad_to_deg = 57.2957795131 # Radians to degrees
deg_in_sky = 41253.               # Square degrees in the sky

# For parallax
km_s_to_mas_yr = (pc_to_cm/1.0e5) * (1.0 / ((180.0/np.pi)*3600.0*1.0e3)) * (1.0 / yr_to_sec)
Rsun_to_deg = (np.pi/180.0) * (pc_to_cm / Rsun_to_cm)


f_bin = 0.5                       # binary fraction

kde_subset = True
