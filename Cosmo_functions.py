#! /usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import numpy as np
import scipy.integrate as integrate
import units_conversions as un
#from scipy.special import spherical_jn


#=======================================================================
#=======================================================================
# critical and photonic density today ==================================
#=======================================================================
#=======================================================================


def rhocrit0(params):
	h    = params["h"]
	unit = params["density unit"]
	value = 1.878e-29*(h**2) #gcm-3   
	if unit=="K4":
		return value*un.conversion("gcm3_k4")
	elif unit=="gcm3":
		return value
	elif unit=="Msun_mpc3":
		unit = un.conversion("g_Msun")/(un.conversion("cm_mpc")**3)
		return value*unit
	else:
		raise ValueError
def Ogammah2(params):
	Tcmb = params["TCMB"]
	rhogamma = (np.pi**2)*(Tcmb**4)/15.
	rhogamma *= 1.279*10**(-35) #k^4 -> gcm^-3
	rhocrith2 = 1.878*10**(-29)
	return rhogamma/rhocrith2

def Orh2(params):
	neff = params["Neff"]	
	return Ogammah2(params)*(1. + 0.2271*neff)

#=======================================================================
#=======================================================================
# Equation of states (EoS) =============================================
#=======================================================================
#=======================================================================

def w_function(z,params):

	if   params['model_w'] == 'constant':
		params["w"]= params['w0']
		return params['w']
		
	elif params['model_w'] == 'cpl':
		wa = params['wa']
		wa = wa*z
		wa = wa/(1. + z)
		params["w"]= params['w0'] + wa
		return params['w']
	
	elif params['model_w'] == 'Interaction':
		print('No implemented')
		sys.exit(0)
		return None

	elif params['model_w'] == 'quintessence':
		print('No implemented')
		sys.exit(0)
		return None

	elif params['model_w'] == 'EDE':
		print('No implemented')
		sys.exit(0)
		return None
				
	else:
		raise NameError("There is such EoS model.")
		
#=======================================================================
#=======================================================================
# Hubble's functions H(z) ==============================================
#=======================================================================
#=======================================================================

def Hubble_a(a, params): 
	h2 = params['h2']
	w  = w_function((-1 + 1./a),params)
	Ok = params['Ok_h2']/h2/np.power(a,2)
	Om = params['Om_h2']/h2/np.power(a,3)
	Or = params['Or_h2']/h2/np.power(a,4)
	Od = params['Od_h2']/h2/np.power(a,-3.*(1+w)) 
		
	return 100*params['h']*np.sqrt(Om + Or + Ok + Od)

def Hubble_z(z, params): 
	h2 = params['h2']
	w  = w_function(z,params)
	Ok = params['Ok_h2']/h2*np.power(1+z,2)
	Om = params['Om_h2']/h2*np.power(1+z,3)
	Or = params['Or_h2']/h2*np.power(1+z,4)
	Od = params['Od_h2']/h2*np.power(1+z,3.*(1+w))
	return 100*params['h']*np.sqrt(Od + Ok + Om + Or)

def E_z(z, params): 
	h2 = params['h2']
	w  = w_function(z,params)
	Ok = params['Ok_h2']/h2*np.power(1+z,2)
	Om = params['Om_h2']/h2*np.power(1+z,3)
	Or = params['Or_h2']/h2*np.power(1+z,4)
	Od = params['Od_h2']/h2*np.power(1+z,3.*(1+w))
	return np.sqrt(Od + Ok + Om + Or)

def E_a(a,params):# For Dz
	h2  = params["h2"]
	Ok  = params["Ok_h2"]/h2
	Or  = params["Or_h2"]/h2	
	Ob  = params["Ob_h2"]/h2
	Oc  = params["Oc_h2"]/h2
	Ode = params["Od_h2"]/h2
	w   = params["w"]
	Ok  = Ok*a**(-2)
	Om  = (Ob+Oc)*a**(-3)
	Or  = Or*a**(-4)
	Ode = Ode*a**(-3*(1+w))
	return np.sqrt(Ok + Om + Or + Ode)
    
def dE_da(a,params):# For Dz
	h2  = params["h2"]
	Ok  = params["Ok_h2"]/h2
	Or  = params["Or_h2"]/h2	
	Ob  = params["Ob_h2"]/h2
	Oc  = params["Oc_h2"]/h2
	Ode = params["Od_h2"]/h2
	w  = params["w"]
	E   = E_a(a,params)
	k  = Ok*a**(-2)
	Om  = (Ob+Oc)*a**(-3)
	Or  = Or*a**(-4)
	Ode = Ode*a**(-3*(1+w))
	dEda = 2*Ok + 3*Om + 4*Or + 3*(1+w)*Ode
	dEda /= -2*a*E
	
	return dEda
#=======================================================================
#=======================================================================
# Distance functions  ==================================================
#=======================================================================
#=======================================================================

def luminosity_distance(z, params):
	
	H0       = 100*params['h']
	integral = lambda x: H0/Hubble_z(x,params)
	integral = integrate.quad(integral,0.,z)[0]
	
	if params['Ok_h2']==0:
		Dl = un.unit("c_kms")*(1.+z)
		Dl = Dl*integral/H0
		return Dl
		
	elif params['Ok_h2']<0:
		h2 = params['h']**2
		Dl = np.sqrt(-params['Ok_h2']/h2)*integral
		Dl = np.sin(Dl)
		Dl = Dl/np.sqrt(-params['Ok_h2'])/100.
		Dl *=un.unit("c_kms")*(1+z)
		return Dl		
	elif params['Ok_h2']>0:
		h2 = params['h']**2
		Dl = np.sqrt(params['Ok_h2']/h2)*integral
		Dl = np.sinh(Dl)
		Dl = Dl/np.sqrt(params['Ok_h2'])/100.
		Dl *=un.unit("c_kms")*(1+z)
		return Dl
			
	else: 
		print('Error')
		import sys
		sys.eixt(0)


def angular_distance(z, params):
	da = luminosity_distance(z,params)
	da = da/((1.+z)**2)
	return da

def comoving_distance(z, params):
	dc = lambda x: 1./Hubble_z(x,params)
	dc = integrate.quad(dc,0.,z)[0]
	dc = un.unit("c_kms")*dc
	return dc 

def angle_averaged_distance(z, params):
	H   = Hubble_z(z, params)
	da  = angular_distance(z, params)
	da  = da*(1.+z)
	da  = da**2
	dv  = da*z
	dv  = un.unit("c_kms")*dv/H
	dv  = pow(dv,1./3.)
	return dv

#=======================================================================
#=======================================================================
#= Growth Function LCDM ================================================
#=======================================================================
#=======================================================================

def Dz_z_LCDM(z, params):#Dodelson, Modern Cosmology pg 207-211, eq. 7.77, for to no interation models
	
	if params['Ok_h2']==0 or params['Om_k']==0.: # Growth_function. This function is to flat universe.
		a_z      = 1./(1. + z)
		integral = integrate.quad(lambda x: 1./np.power(x*Hubble_a(x,params)/(100*params['h']),3),0.,a_z)[0]
		Om       = params['Om_h2']/(params['h']**2)
		H_a      = Hubble_z(z,params)/(100*params['h'])
		
		D1       = 5.*Om*H_a/2.
		D1       = D1*integral
		
		norm     = integrate.quad(lambda x: 1./pow(x*Hubble_a(x,params)/(100*params['h']),3),0.,1.)[0]
		norm     = 5.*Om*norm/2.
		
		return D1/norm
		
	elif params['Ok_h2']>0: # Growth_function. This function is to open universe. k<0 -> Ok>0
		a_z = 1./(1. + z)
		Om  = params['Om_h2']/(params['h']**2)
		x   = 1. - Om
		x   = x*a_z/Om
		
		D1  = np.log(np.sqrt(1+x) - np.sqrt(x))
		D1  = 3.*D1*np.sqrt(1+x)
		D1  = D1/np.power(x,3./2.)
		D1  += 3./x
		D1  += 1
		D1  *= (5.*a_z)/(2.*x)
		
		x    = (1. - Om)/Om
		norm = np.log(np.sqrt(1+x) - np.sqrt(x))
		norm = 3.*D1*np.sqrt(1+x)
		norm = D1/np.power(x,3./2.)
		norm += 3./x
		norm += 1
		norm *= 5./(2.*x)
		
		return D1/norm
	else:
		print("Error")
		print("Finish program !")
		sys.exit(0)
	return None

def Dz_LCDM(z,params):
	z_type = un.verification_dtype_list(z)
	if len(z_type)>1: 
		D = np.empty(len(z))
		for i,z_i in enumerate(z): D[i] = Dz_z_LCDM(z_i,params)
		return D
	elif len(z_type)==1: return Dz_z_LCDM(z,params)
	else:
		print("Error")
		print("Finish program")
		sys.exit(0)
		


#=======================================================================
#=======================================================================
#= Analytic Growth Function ============================================
#=======================================================================
#=======================================================================

	
def dD_da(D, a, Ode,Ok,Or,Ob,Oc,w,h2):
	paramsDa = {"Ok_h2":Ok*h2,"Or_h2":Or*h2, "Ob_h2":Ob*h2,"Oc_h2":Oc*h2,"Od_h2":Ode*h2,"w":w,"h2":h2}
	
	E     = E_a(a,paramsDa)
	dEda  = dE_da(a,paramsDa)
	dD0da = D[1]
	dD1da = -((dEda/E) + (3/a))*D[1] + ((3*(Ob+Oc))/(2*(a**5)*(E**2)))*D[0]
	return [dD0da,dD1da]

def D_a(params, norm=True):
    from scipy.integrate import odeint
    
    a    = np.linspace(1e-2,1,1e4)
    h2   = params["h2"]
    Ok   = params["Ok_h2"]/h2
    Or   = params["Or_h2"]/h2
    Ob   = params["Ob_h2"]/h2
    Oc   = params["Oc_h2"]/h2
    Ode  = params["Od_h2"]/h2
    w    = w_function((1./a)-1,params)

    D_ini = [a[0], 1.] #(D(a[0]),dD/da(a[0]))
    D_out = odeint(dD_da, D_ini, a, args=(Ode,Ok,Or,Ob,Oc,w,h2))[:,0]
    
    if   norm==True:  return D_out/np.max(D_out)
    elif norm==False: return D_out
    else:             raise ValueError
        
def D_z(z,params,norm=True):
	from scipy.interpolate import interp1d
	Da       = np.flip(D_a(params,True))
	a        = np.linspace(1e-2,1,1e4)
	D_interp = interp1d(a,Da)
	a_output = 1./(1+z)
	return D_interp(a_output)
	
#=======================================================================
#=======================================================================
#= Growth rate =========================================================
#=======================================================================
#=======================================================================

def growth_rate(a_range,a_out,D):
    D = interp1d(a_range,D)
    lna = np.log(a_out)
    lnD = np.log(D(a_out))
    dlnDdlna = np.gradient(lnD,lna)
    return dlnDdlna
        
def f_peebles(a ,params):
    h2   = params["h2"]
    Ob   = params["Ob_h2"]/h2
    Oc   = params["Oc_h2"]/h2
    gamma = params["gamma"]
    E = E_a(a,params)
    Om = Ob+Oc
    Om/=((a**3)*(E**2))
    return Om**gamma
     
#=======================================================================
#=======================================================================
#= Window Function =====================================================
#=======================================================================
#=======================================================================

def window_function(z,z_bin, k, params, paramSurvey): #window = "battye","tophat","gaussian","zero","one"
	z_min  = z_bin[0]
	z_max  = z_bin[1]
	window = paramSurvey['window']

	if window == "battye":
		if z_min==z_max:
			return 0
		elif z>=z_min and z<=z_max:
			W = z_max - z_min
			return 1./W
		else: 
			return 0.
	
	elif window =="tophat": 
		if z_min==z_max:# exijo que exista intervalo, senão não faz sentido o Intesity Mapping(IM)
			return 0
		elif z>=z_min and z<=z_max:
			return 1.
		else:
			return 0.
	
	elif window == "gaussian":
		kR  = k*params['R']
		kR2 = kR**2
		WR  = np.exp(-kR2/2.)
		WR  = WR/((np.sqrt(2.*np.pi)*params['R'])**3)		
		return WR
			
	elif window == "zero":
		return 0.
	
	elif window == "one":
		return 1.
	
	else:
		print("Error")
		print("Finish Program")
		import sys
		sys.exit(0)
		
	return None

def fourier_window_function(k, params):
	symmetry = "spherical"
	window = "tophat"
	
	if symmetry == "spherical" and window == "tophat":
		kR  = k*params['R']
		kR3 = kR**3
		WR  = (kR)*np.cos(kR)
		WR  = np.sin(kR) - WR
		WR  = 3.*WR
		return WR/(kR3)
		
	elif symmetry == "spherical" and window == "gaussian":
		kR  = k*params['R']
		kR2 = kR**2
		WR  = np.exp(-kR2/2.)
		return WR
	
	else:
		print("Error")
		print("Finish Program")
		sys.exit(0)
	
	return None

#=======================================================================
#=======================================================================
#= Parameters ==========================================================
#=======================================================================
#=======================================================================

def AP_parameter(z,params): #Alcock-Paczynski parameter
	F_ap = (1+z)/un.unit("c_kms")
	F_ap *= angular_distance(z,params)*Hubble_z(z, params)
	return F_ap
	
def acoustic_parameter_A(z, params): #Eisenstein et al. 2005
	A = np.sqrt(params['Om_h2']*(1.e4))/un.unit("c_kms")
	A *= angle_averaged_distance(z, params)/z
	return A
