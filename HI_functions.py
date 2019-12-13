import numpy as np
import units_conversions as un
import Cosmo_functions as cf
from scipy.integrate import quad


def bias_z(z):
	b0z = 0.67
	b1z = 0.18
	b2z = 0.05
	b1z = b1z*z
	b2z = b2z*(z**2)
	return b0z + b1z + b2z

def bias_k(k):
	b0k = 1.00
	b1k = -1.1
	b2k = 0.40
	b1k = b1k*k
	b2k = b2k*(k**2)
	return b0k + b1k + b2k

def biasHI(x, paramSurvey):
	var = paramSurvey["biasHI_model"]
	
	if  var == "z":
		b = np.empty(len(x))
		for i,z_i in enumerate(x): b[i] = bias_z(z_i)
		return b
		
	elif var == "k":
		b = np.empty(len(x))
		for i,k_i in enumerate(x): b[i] = bias_k(k_i)
		return b
		
	else:
		print("Erro")
		print("Finish program")
		sys.exit(0)

def OmegaHI_h(z, params, model):
                                 
	if model == "constant":
		return 2.45*1.e-4 # Richard Battye et al., 2012, arxiv:1209.0343v1
		
	elif model == "crighton":
		OmegaHI0 = 4.*1.e-4
		g        = 0.6
		return OmegaHI0*((1. + z)**g)*params['h']
		
	else:
		rhoHI_z = 0 
		H02     = (100*params['h'])**2
		OmegaHI = 3*H02
		OmegaHI = 8*np.pi*un.unit("G_km3kg-1s-2")/OmegaHI
		omegaHI *= rhoHI
		return OmegaHI*params['h']


def T21_mean(z, params, paramSurvey):#mean_brightness_temperature
	OHI_model     = paramSurvey["unit_21cm"]
	OmegaHI_model = paramSurvey['OmegaHI_model']
	
	T0  = 44.#46.06
	T0  = T0*(OmegaHI_h(z, params, OmegaHI_model)/(2.45*1.e-4))
	E_z = cf.E_z(z,params)
	T   = T0*((1.+z)**2)/E_z
	
	if   OHI_model=="muk":
		return T
	elif OHI_model=="mk":
		return T*1.e-3
	elif OHI_model=="k":
		return T*1.e-6
	else:
		raise NameError

def angular_density_sources(z_min, z_max, params, finity_range): # ex.: z_min = bpar.z_min_BINGO, z_max = bpar.z_max_BINGO
	
	r2_H_z = lambda x: (cf.comoving_distance(x,params)**2)/cf.Hubble_z(x,params)
	
	if finity_range == True:
		N = quad(r2_H_z,z_min,z_max)[0]
	else:
		N = quad(r2_H_z,0.,np.inf)[0]
		
	N = (params["n0"]*un.unit("c_kms"))*N
	return N #Mpc-3
