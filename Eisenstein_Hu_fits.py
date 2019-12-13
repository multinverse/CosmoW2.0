import numpy as np
import scipy.integrate as integrate
import units_conversions as un
from scipy.special import spherical_jn
import Cosmo_functions as cf

#==================================================================================
#==================================================================================
#= Cosmological variables =========================================================
#==================================================================================
#==================================================================================

def z_eq(params):
	zeq = params['Om_h2']/params['Or_h2']
	zeq = zeq-1.
	return zeq

def k_eq(params): 
	keq = 2.*params['Om_h2']*(100**2)
	keq = keq*z_eq(params)
	keq = np.sqrt(keq)
	return keq/un.unit("c_kms")

def k_silk(params):
	ksilk   = np.power(10.4*params['Om_h2'],-0.95)
	ksilk   = 1.+ ksilk
	ksilk   = np.power(params['Om_h2'],0.73)*ksilk
	ksilk   = np.power(params['Ob_h2'],0.52)*ksilk
	return 1.6*ksilk #Mpc-1

#==================================================================================
#==================================================================================
# Fits variables to build of the tranfers functions ===============================
#==================================================================================
#==================================================================================

def b1(params):
	b_1 = 0.607*np.power(params['Om_h2'],0.674)
	b_1 = 1. + b_1
	b_1 = b_1*0.313/np.power(params['Om_h2'],0.419)
	return b_1

def b2(params):
	return 0.238*np.power(params['Om_h2'],0.223)

def z_drag(params):
	b_1 = b1(params)
	b_2 = b2(params)
	zd = 1291*np.power(params['Om_h2'],0.251)
	zd = zd*(1. + b_1*np.power(params['Ob_h2'],b_2))
	zd = zd/(1. + 0.659*np.power(params['Om_h2'],0.828))
	return zd

def R_parameter(z,params):
	R = (3.*params['Ob_h2'])/(4.*params['Og_h2'])
	R = R/(1.+z)
	return R

def sound_speed(z,params): 
	R = R_parameter(z,params)
	return un.unit("c_kms")/np.sqrt(3.*(1.+R)) #km.s-1

def sound_horizon_drag(params): 
	
	zeq     = z_eq(params)
	zdrag   = z_drag(params)
	keq     = k_eq(params)
	R_drag  = R_parameter(zdrag,params)
	R_eq    = R_parameter(zeq,params)
		
	s       = np.log((np.sqrt(1.+R_drag) + np.sqrt(R_drag + R_eq))/(1. + np.sqrt(R_eq)))
	s       = np.sqrt(6./R_eq)*s
	s       = (2./(3.*keq))*s
	return s 


#==================================================================================
#==================================================================================
#==================================================================================
#==================================================================================
#==================================================================================

def s_parameter	(params):
	s = 44.5*np.log(9.83/params['Om_h2'])
	s = s/np.sqrt(1. + 10.*np.power(params['Ob_h2'],0.75))
	return s

def k_peak(params):
	s = s_parameter(params)
	k = 5*np.pi*(1. + 0.217*params['Om_h2'])
	return k/(2.*s)

def alpha_gamma(params):
	f     = params['Ob_h2']/params['Om_h2']
	f2    = f**2
	alpha = 0.38*np.log(22.3*params['Om_h2'])*f2
	alpha = alpha - 0.328*np.log(431.*params['Om_h2'])*f 
	return 1. + alpha

def gamma(params):
	return params['Om_h2']/params['h']
	
def gamma_eff(ks, params):
	alpha  = alpha_gamma(params)
	return gamma(params)*(alpha + ((1.- alpha)/(1. + (0.43*ks)**4)))

def q_parameter(k, gamma, params):
	theta  = params['TCMB']/2.7
	theta2 = theta**2
	return (k/params['h'])*(theta2/gamma) #k should are in h Mpc-1.

def C0(q):
	c = 731./(1.+62.5*q)
	return c+14.2

def L0(q): 
	return np.log(2.*np.e +1.8*q)

def T0(q):
	t=L0(q)
	t=t/(t+(C0(q)*(q**2)))
	return t

#==================================================================================
#==================================================================================
# Fits Eisenstein&Hu ===============================================================
#==================================================================================
#==================================================================================

def a_1(params):
	a       = 32.1*params['Om_h2']
	a       = 1.+ np.power(a,-0.532)
	a       = a*np.power(46.9*params['Om_h2'],0.670)
	return a

def a_2(params):
	a       = 45.0*params['Om_h2']
	a       = 1.+ np.power(a,-0.582)
	a       = a*np.power(12.0*params['Om_h2'],0.424)
	return a
	
def b_1(params):
	b       = 458.*params['Om_h2']
	b       = 1.+np.power(b,-0.708)
	b       = 0.944/b
	return b

def b_2(params):
	b       = 0.395*params['Om_h2']
	b       = np.power(b,-0.0266)
	return b

def alpha_b(params):
		
	zeq     = z_eq(params)
	zd      = z_drag(params)
	Rd      = R_parameter(zd,params)
	keq     = k_eq(params)
	s       = s_parameter(params)
	ks      = keq*s
	G       = G_function(zeq,zd)
	
	alpha = 2.07*ks*G
	alpha = alpha/np.power(1.+Rd,0.75) 
	return alpha

def alpha_c(params):
	f      = params['Ob_h2']/params['Om_h2']
	f3     = f**3
	a1      = np.power(a_1(params),-f)
	a2      = np.power(a_2(params),-f3)
	return a1*a2

def beta_b(params):
	f       = params['Ob_h2']/params['Om_h2']
	beta    = 1.+ (17.2*params['Om_h2'])**2
	beta    = np.sqrt(beta)
	beta    = (3. - 2.*f)*beta
	beta    = 0.5 + f + beta
	return beta

def beta_c(params):
	fc      = params['Oc_h2']/params['Om_h2']
	b1      = b_1(params)
	b2      = b_2(params)
	beta    = (np.power(fc,b2)-1.)
	beta    = b1*beta
	beta    = beta +1.
	return 1./beta

def beta_node(params):
	return 8.41*np.power(params['Om_h2'],0.435)

def G_function(zeq, zd):
	x = (1.+zeq)/(1.+zd)
	G = (np.sqrt(1.+x)+1.)/(np.sqrt(1.+ x)-1.)
	G = (2.+3.*x)*np.log(G)
	G = G - 6.*np.sqrt(1.+x)
	G = x*G
	return G

def T0_til(q, k, alphac, betac):
	q2 = q**2
	C  = (1.+69.9*np.power(q,1.08))
	C  = 386./C
	C  = C+(14.2/alphac)
	
	T  = np.log(np.e + 1.8*betac*q) 
	T  = T/(T+C*q2)
	return T

def s_shifting(k,params):
	if k==0 or k==0.:return 0.
	betan      = beta_node(params)
	s          = s_parameter(params)
	ks         = k*s
	betan      = (betan/ks)**3
	s_til      = np.power(1.+ betan,1./3.)
	return s/s_til

#==================================================================================
#==================================================================================
#= Transfer Functions : Baryons and CDM ===========================================
#==================================================================================
#==================================================================================

def Tb_eisenstein(k,params):
	#if k==0 or k==0.: return 1.
	
	alphab   = alpha_b(params)
	betab    = beta_b(params)
	
	q        = q_parameter(k, gamma(params), params)
	ksilk    = k_silk(params)
		
	s        = s_parameter(params)
	s_til    = s_shifting(k, params)
	ks       = k*s
		
	T0       = T0_til(q,k,1.,1.)
	Tb       = T0/(1.+(ks/5.2)**2)
	Tb       = Tb + (alphab/(1.+np.power(betab/ks,3)))*np.exp(-np.power(k/ksilk,1.4))
	
	j0       = spherical_jn(0,k*s_til,0)
	Tb       = Tb*j0
	return Tb

def Tc_eisenstein(k, params):
	ks      = k*s_parameter(params)
		
	alphac  = alpha_c(params)
	betac   = beta_c(params)
	q       = q_parameter(k, gamma(params),params)
	
	T0      = T0_til(q,k,alphac,betac)
	T0_1    = T0_til(q,k,1.,betac)
	f       = 1. + np.power(ks/5.4,4)
	f       = 1./f
	
	Tcdm    = f*T0_1 + (1.-f)*T0
	return Tcdm

#==================================================================================
#==================================================================================
#= Transfer Functions : NoBAO, Nobaryons, BBKS, Eisenstein&HU and CAMB ============
#==================================================================================
#==================================================================================

def Tm_noBAO(k, params):
	ks    = k*s_parameter(params)
	Gamma = gamma_eff(ks, params)
	q_eff = q_parameter(k, Gamma, params)
	return T0(q_eff)

def Tm_nobaryons(k, params):
	q = q_parameter(k, gamma(params), params)
	return T0(q)

def Tm_BBKS(k, params):# fit Bardeen,Bond,Kaiser & Szalay(BBKS)
	keq = k_eq(params)
	x   = k/keq
	TF  = 1.+(0.284*x)+np.power(1.18*x,2)+np.power(0.399*x,3)+np.power(0.490*x,4)
	TF  = np.power(TF,-0.25)
	TF  = TF*np.log(1.+0.171*x)
	if k!=0:TF = TF/(0.171*x)
	else: TF=1.
	return TF

def Tm_eisenstein(k, params):
	                                                                                               
	fb = params["Ob_h2"]/params["Om_h2"]
	fc = params["Oc_h2"]/params["Om_h2"]
	Tc = Tc_eisenstein(k, params)
	Tb = Tb_eisenstein(k, params)
	return fb*Tb + fc*Tc 


def TF_vector(k_vector,params):
	model     = params["transfer_function"]
	k_vector  = un.verification_dtype_list(k_vector)
	len_model = np.shape(model)[0]
	len_k     = np.shape(k_vector)[0]
	TF_out    = np.zeros((len_model,len_k))
	
	if len_model > 0 and len_k>1:
		for i,name in enumerate(model):
			if   name=="EH":
				for j,kj in enumerate(k_vector):
					TF_out[i][j]=Tm_eisenstein(kj,params)
			elif name=="BBKS":
				for j,kj in enumerate(k_vector):
					TF_out[i][j]=Tm_BBKS(kj,params)
			elif name=="noBAO":
				for j,kj in enumerate(k_vector):
					TF_out[i][j]=Tm_noBAO(kj,params)
			elif name=="nobaryons":
				for j,kj in enumerate(k_vector):
					TF_out[i][j]=Tm_nobaryons(kj,params)
			elif name=="camb":
				import camb
				from scipy.interpolate import interp1d
				from camb import model, initialpower
				from camb.dark_energy import DarkEnergyFluid
				pars = camb.CAMBparams()
				pars.set_cosmology(H0=100*params['h'], ombh2=params['Ob_h2'], omch2=params['Oc_h2'], omk=params['Ok_h2']/params['h2'],TCMB=params['TCMB'])
				pars.InitPower.set_params(As=params['As'], ns=params['ns'], r=0)
				pars.DarkEnergy = DarkEnergyFluid(w=params['w0'], wa=params['wa'])
				pars.set_matter_power(redshifts=[0], kmax=params['kmax'])
				results       = camb.get_results(pars)
				trans         = results.get_matter_transfer_data()
				kh            = trans.transfer_data[0,:,0]
				kk            = kh*results.Params.h
				transfer      = trans.transfer_data[model.Transfer_tot-1,:,0]
				transfer      = interp1d(kk,transfer)(k_vector)
				TF_out[i][:]  = transfer
			else:
				raise NameError("There is no such name")
	elif len_model > 0 and len_k==1:
		for i,name in enumerate(model):
			if   name == "EH":
				TF_out[i][0]=Tm_eisenstein(k_vector[0],params)
			elif name == "BBKS":
				TF_out[i][0]=Tm_BBKS(k_vector[0],params)
			elif name == "noBAO":
				TF_out[i][0]=Tm_noBAO(k_vector[0],params)
			elif name == "nobaryons":
				TF_out[i][0]=Tm_nobaryons(k_vector[0],params)
			else:
				raise NameError("There is no such name")
	else:
		raise NameError("There is no such name")
	return TF_out

	
#==================================================================================
#==================================================================================
#= Primordial Power Spectrum and cursive Transfer Functions  of i-specie. =========
#==================================================================================
#= see  Euclid preparation: VII. Forecast validation for Euclid cosmological ======
#= probes (2019). arXiv: 1910.09273v1 equation (31) ===============================
#==================================================================================
#==================================================================================

#def cursiveTFi(k,z,TF,params):
#	Da = cf.D_z(z,params,True)	
def dimensionless_primordial_Pk(k,params):
	As = params["As"]
	k0 = params['kpivot']
	ns = params["ns"]
	return As*((k/k0)**(ns-1))
	
#=======================================================================
#=======================================================================
#= Amplitude with sigmaR ===============================================
#=======================================================================
#=======================================================================

def sigma2R(k,PK,params,integrator="trapz"):
	if integrator=="quad":
		from scipy.integrate import quad
		integrand = lambda x: (fourier_window_function(k,params)**2)*((x**2)*PK(x)/(2*np.pi**2))
		return quad(integrand,k[0],k[-1])[0]
	elif integrator=="trapz":
		kernel  = fourier_window_function(k,params)**2
		deltaPK = (k**2)*PK(k)/(2*np.pi**2)
		return np.trapz(deltaPK*kernel,k)
	elif integrator=="simpson":
		from scipy.integrate import simps
		kernel  = fourier_window_function(k,params)**2
		deltaPK = (k**2)*PK(k)/(2*np.pi**2)
		return simps(deltaPK*kernel,k)
	else:
		raise ValueError("There is no such integrator")

def sigma2N(k,TF,params,integrator="trapz"): #see Euclid preparation: VII. Forecast validation for Euclid cosmological probes (2019). arXiv: 1910.09273v1    equation (29)
	ns = params["ns"]
	if integrator=="quad":
		from scipy.integrate import quad
		integrand = lambda x: (fourier_window_function(k,params)**2)*((x**(2+ns))*(TF(x)**2)/(2*np.pi**2))
		return quad(integrand,k[0],k[-1])[0]
	elif integrator=="trapz":
		kernel  = fourier_window_function(k,params)**2
		deltaPK = (k**(2+ns))*(TF(x)**2)/(2*np.pi**2)
		return np.trapz(deltaPK*kernel,k)
	elif integrator=="simpson":
		from scipy.integrate import simps
		kernel  = fourier_window_function(k,params)**2
		deltaPK = (k**(2+ns))*(TF(x)**2)/(2*np.pi**2)
		return simps(deltaPK*kernel,k)
	else:
		raise ValueError("There is no such integrator")

def delta_H(params):
	
	n_til = params['ns'] - 1.
	h     = params['h']
	Om    = params['Om_h2']/h**2
	
	if params['Ok_h2']==0 or params['Ok_h2']==0.:
		delH = np.exp(-(n_til + 0.14*n_til**2))
		delH *= Om**(-(0.35 + 0.19*np.log(Om)+0.017*n_til))
		delH *= 1.95e-5
		return delH
	elif params['Ok_h2']>0:
		delH = np.exp(-(0.95*n_til + 0.169*n_til**2))
		delH *= Om**(-(0.785 + 0.05*np.log(Om)))
		delH *= 1.94e-5
		return delH
	else: 
		print("Error")
		print("Finish program !!!")
		sys.exit(0)

def Amp_Pk(k,TF,params):
	
	if   params['type_of_amplitude']=='sigmaR':
		sigma_2N = sigma2N(k,TF,params,"trapz")
		sigma_2R = params["sigma8"]**2
		return sigma_2R/sigma_2N
		
	elif params['type_of_amplitude']=='cobe':
		amp  = 2*np.pi**2
		amp *= delta_H(params)**2
		amp /= (100*params["h"]/un.unit("c_kms"))**(params["ns"]+3)
		return amp

	
	elif params['type_of_amplitude']=="camb":
		import camb
		from camb import model, initialpower
		from camb.dark_energy import DarkEnergyFluid
		z = np.array([0])
		pars = camb.CAMBparams()
		pars.set_cosmology(H0=100*params['h'], ombh2=params['Ob_h2'], omch2=params['Oc_h2'], omk=params['Ok_h2']/params['h2'],TCMB=params['TCMB'])
		pars.InitPower.set_params(As=params['As'], ns=params['ns'], r=0)
		pars.DarkEnergy = DarkEnergyFluid(w=params['w0'], wa=params['wa'])
		pars.set_matter_power(redshifts=z, kmax=params['kmax'])
		pars.InitPower.set_params(ns=params['ns'])
		pars.set_matter_power(redshifts=z, kmax=params['kmax'])
		results       = camb.get_results(pars)
		primoridal_PK = results.Params.scalar_power(k)
		return primoridal_PK*k**4/(k**3/(2*np.pi**2))	
	
	elif params['type_of_amplitude']=='value':
		try:
			return params["value_of_amplitude"]
		except:
			raise ValueError			
	
	else: 
		raise ValueError("There is no such amplitude")

#==================================================================================
#==================================================================================
#== Power Spectrum: noBAO, nobaryons, BBKS and Eisenstein&Hu ======================
#==================================================================================
#==================================================================================

def Pk(k,params,TF_input = None):
	if type(TF_input).__name__=="NoneType":
		TF = TF_vector(k,params)
	else:
		TF = TF_input
		
	Amp = Amp_Pk(k,TF,params)
	
	if params['type_of_amplitude']=="camb":
		if   params['normalization_transfer_function']=="1" and type(TF_input).__name__!="NoneType":
			TF = TF_vector(k,params)
			len_TF   = np.shape(params['transfer_function'])[0]
			
			try:
				loc_camb = np.where(params['transfer_function']=="camb")[0][0]
				TFcamb   = TF[loc_camb,0]
			except:
				par = params
				par['transfer_function']=["camb"]
				TFcamb = TF_vector(k,par)[0]
				TFcamb*=params['alpha_camb']
							
			for  i in range(len_TF):
				TF[i,:] *= (TFcamb/TF[i,0])

		elif params['normalization_transfer_function']=="camb":
			pass
		else:
			raise NameError("There is no such type_of_amplitude.")
		return Amp*(TF**2)
	else:
		return Amp*(k**params['ns'])*(TF**2)

def Pkz(z,params, TF_input = None):		
	
	k  = np.logspace(np.log10(params["kmin"]),np.log10(params["kmax"]),num=1e4)
	Dz = cf.D_z(z,params,True)
	PK = Pk(k,params,TF_input)
	z  = un.verification_dtype_list(z)
	len_z = np.shape(z)[0]
	len_PK = np.shape(PK)[0]

	if   len_z == 1:
		for i in range(len_PK):
			PK[i,:] *= Dz**2
		return PK
	
	elif len_z >  1:		
		for i in range(len_z):
			for j in range(len_PK):
				if i+j == 0:
					PKz = PK[0,:]*Dz[0]**2
				else:
					PKz = np.vstack((PKz,PK[j,:]*Dz[i]**2))
		return PKz

	else:
		raise ValueError


#==================================================================================
#==================================================================================
#== 21cm Angular Power Spectrum: Limber Approximation =============================
#==================================================================================
#==================================================================================

def CL21_Limber(l_vector, z_bin, kk, PK, Dz, params, paramSurvey):
	from scipy.interpolate import interp1d
	from scipy.integrate import quad
	import HI_functions as hi
	
	#len_PK = len(params['transfer_function'])
	zz = np.linspace(z_bin[0],z_bin[1],endpoint=True,num=1e3)
	PK = interp1d(kk,PK)
	
	if   paramSurvey['biasHI_model']=="k":
		bHI = hi.biasHI(kk,paramSurvey)
		bHI = interp1d(kk,bHI)
		raise NameError("not implemented")
	elif paramSurvey['biasHI_model']=="z":
		bHI = hi.biasHI(zz, paramSurvey)
		bHI = interp1d( zz, bHI)
	elif paramSurvey['biasHI_model']=="constant":
		bHI = lambda x: 1
	else:
		raise NameError("in biasHI_model parameter")
	
	cl_l = np.zeros((len(l_vector)))
	
	T21  = lambda x: hi.T21_mean(x,params,paramSurvey)
	Ez   = lambda x: cf.E_z(x,params)
	chiz = lambda x: cf.comoving_distance(x,params)
	Wz   = lambda x: cf.window_function(x,z_bin,kk,params,paramSurvey)
	
	for j,ll in enumerate(l_vector):	
		kl          = lambda x: np.sqrt(ll*(ll+1.))/chiz(x)
		PKx         = lambda x: PK(kl(x))
		integrating = lambda x: (100*params['h']/un.unit("c_kms"))*(PKx(x)*Ez(x))*((Wz(x)*Dz(x)*T21(x))/chiz(x))**2
		cl_l[j]     = quad( integrating, z_bin[0], z_bin[1] )[0]
	
	return cl_l
