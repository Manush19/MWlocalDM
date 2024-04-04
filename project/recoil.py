import numpy as np
import scipy as sp
from scipy.signal import medfilt
from project.constants import MW_params as pm

norm = lambda vdf,v: vdf/np.trapz(vdf, v)

def vdf_fn(v_,v,vdf): 
    vdffn = sp.interpolate.interp1d(v,vdf,
                                    kind = 'linear',
                                    bounds_error = False,
                                    fill_value = 0)(v_)
    vdffn[vdffn < 1e-5] = 0
    return vdffn

def get_vdf_ert(vE, v, vdf, vesc, vert = pm.Vert):
    vdfE = []
    v[v == 0] = 1e-5
    vdf[v > vesc] = 1e-5 
    vdf = norm(vdf/v**2,v)
    for vE_ in vE:
        cmax = (vesc**2 - vE_**2 - vert**2)/(2.*vE_*vert)
        
        x_plus = np.linspace(-1, cmax, 1000)
        newv_plus = np.sqrt(vE_**2 + vert**2 + 2*vE_*vert*x_plus)
        f_xplus = vdf_fn(newv_plus, v, vdf)
        int_plus = np.trapz(f_xplus, x_plus)
        t_plus = np.heaviside(vesc + vert - vE_, 0)*int_plus


        x_minus = np.linspace(1, cmax, 1000)
        newv_minus = np.sqrt(vE_**2 + vert**2 + 2.*vE_*vert*x_minus)
        f_xminus = vdf_fn(newv_minus, v, vdf)
        int_minus = np.trapz(f_xminus, x_minus)
        t_minus = np.heaviside(vesc - vert - vE_, 0)*int_minus

        vdfE.append((vE_**2)*(t_plus - t_minus)*2*np.pi)

    vdfE = np.array(vdfE)
    return norm(medfilt(vdfE, kernel_size = 3), vE)


def get_eta(v, vdf, tol = 1e-5):
    return None