import numpy as np

#c_light_m_s        = 2.99792458e8          #m/s
#c_light            = c_light_m_s/1e3       #km/s
#G_const_Mpc_Msun_s = 4.51737014558*1.e-48
#G_const            = 6.67408*1.e-11        #(m3)/(kg*s2)
#G_const_km         = 6.67408*1.e-2         #(km3)/(kg*s2)
#Mpc_cm             = 3.08568025*1.e+24     #Mpc in cm
#Mpc_km             = 3.08568025*1.e+19     #Mpc in km
#K4_gcm3            = 1.279*10**(-35)       # 1 k**4 = 1.279.1e-35 g cm-3


def conversion(call=None):    
    #"A_B": unit A --> unit B
    if call=="Msun_kg":
        return 1.989e30
    elif call=="Msun_g":
        return 1.989e33
    elif call=="kg_Msun":
        return 1./1.989e30
    elif call=="g_Msun":
        return 1./1.989e33
    elif call=="pc_km":
        return 3.0857e13
    elif call=="pc_m" or call=="kpc_km":
        return 3.0857e16
    elif call=="pc_cm":
        return 3.0857e18
    elif call=="mpc_km" or call=="kpc_m":
        return 3.0857e19
    elif call=="kpc_cm":
        return 3.0857e21
    elif call=="mpc_m":
        return 3.0857e22
    elif call=="mpc_cm":
        return 3.0857e24
    elif call=="km_pc":
        return 1./3.0857e13
    elif call=="m_pc" or call=="km_kpc":
        return 1./3.0857e16
    elif call=="cm_pc":
        return 1./3.0857e18
    elif call=="km_pc" or call=="km_mpc":
        return 1./3.0857e19
    elif call=="cm_kpc":
        return 1./3.0857e21
    elif call=="cm_mpc":
        return 1./3.0857e24
    elif call=="gcm3_k4":
        return 1./1.279e-35
    elif call=="k4_gcm3":
        return 1.279e-35
    else:
        return None

def unit(call=None):
	if   call=="c_ms":
		return 2.99792458e8
	elif call=="c_kms":
		return 2.99792458e8/1e3
	elif call=="G_pc1Msun-1km2s2":
		return 4.302e-3
	elif call=="G_kpc1Msun-1km2s2":
		return 4.302e-6
	elif call=="G_mpc1Msun-1km2s2":
		return 4.302e-9
	elif call=="G_m3kg-1s-2":
		return 6.67430e-11		
	elif call=="G_km3kg-1s-2":
		return 6.67430e-20
	elif call=="lambda_21cm_m":
		return 0.21106
	elif call=="lambda_21cm_cm":
		return 21.106
	elif call=="freq_21cm_MHz":
		return 1420.405751
	elif call=="freq_21cm_GHz":
		return 1.420405751
	else:
		return None
	

def convers_z_nu(z):
	nu_emit = unit("freq_21cm_MHz")
	return nu_emit/(1.+z)
	
def convers_nu_z(nu):
    nu_emit = unit("freq_21cm_MHz")
    return (nu_emit/nu)-1.

def convers_lamb_nu(lamb):
	return conversion("c_kms")*1.e3/lamb

def convers_nu_lamb(nu):
	return conversion("c_kms")*1.e3/nu

def verification_dtype_list(x):	
	if (type(x)==int) or (type(x)==float):
		return[x]
	elif type(x)==list:
		return x
	elif (x.dtype == np.dtype(float))or(x.dtype == np.dtype(int)):
		x = x.tolist()
		if (type(x)==float)or(type(x)==int):
			return [x]
		else:
			return x
	else: 
		return x	


