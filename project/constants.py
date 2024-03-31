import numpy as np

class Constants:
	"""
	Class for fixing all the constants. We can import the necessary constants
	from this class to have a consistent value of these constants through out
	the code.
	"""
	pi = np.pi
	G = 4.302*1.0e-6 # M_sun^-1 (km/s)^2 kpc
	H0 = 72.1 # (km/s) Mpc^-1
	err_H0 = 2. 
	h = H0/100. 
	rho_crit = 3.*(H0**2)/(8.*pi*G)*1e-6 # M_sum kpc^-3
	err_rho_crit = rho_crit * 2. * (err_H0/H0)
	kpc = 3.086*1e16 # 1kpc = 3.08*1e16 km
	yr = 3600.0*24.0*365.5 # 1yr = ___ sec
	Gyr = 1e9 # 1Gyr = 1e9 yr
	K = 0.3 #for lower cs K,a = 0.1,-1.5
			#for higher cs K,a = 1.0,-2.5
	a = -2.
	err_K = 0.2
	err_a = 0.5
	M_sun = 1.989*1e30 #kg
	ergs = 1e7 #joules
	vel_light = 2.998*1e8 #m/s
	fb = 0.16 # baryon fraction.
	m200_c200_scatter = 0.11 #dex
	GeV = 1.7827 * 1e-27 # 1GeV c^2 = this much kg
	GeV_by_cc = 1.7827 * (3.08**3) * 1e6/ 1.989 # Msun kpc^-3
