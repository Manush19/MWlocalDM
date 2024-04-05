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

def get_vdf_ert(vE, v, vdf, vesc, vcirc):
    vert = vcirc*(1.05+0.07)
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


def get_eta(vmin, vE, vdfE, vesc, vcirc):
    vert = vcirc*(1.05 + 0.07)
    eta = []
    for v_ in vmin:
        v_plus = np.min([v_, vesc + vert])
        v_minus = np.min([v_, vesc - vert])

        low1 = v_plus
        hig1 = vesc + vert
        if low1 <= hig1:
            idx1 = np.where((vE >= low1) & (vE <= hig1))
            idg1 = vdfE[idx1]
            v1 = vE[idx1]
            int1 = np.trapz(idg1/v1, v1)
        else:
            int1 = 0

        low2 = v_minus
        hig2 = vesc - vert
        if low2 >= hig2:
            idx2 = np.where((vE >= low2) & (vE <= hig2))
            idg2 = vdfE[idx2]
            v2 = vE[idx2]
            int2 = np.trapz(idg2/v2, v2)
        else:
            int2 = 0
        eta.append(int1 - int2)
    return np.array(eta)


Atomic_mass = {'Na': 23.0, 'I': 127.0, 'W': 184.0, 'Ca': 40.0, 'O': 16.0, 'Al': 27.0, 'Xe': 132.0, 'Ge': 74.0}

class Nuclear:
    def __init__(self, element, vE, vdfE, vesc, vcirc, rhosun, **kwargs):

        self.element = element
        self.A = Atomic_mass[element]
        self.mT = Atomic_mass[element]
        self.vE = vE
        self.vdfE = vdfE
        self.vesc = vesc
        self.vcirc = vcirc
        self.vert = self.vcirc*(1.05 + 0.07)
        self.rhosun = rhosun

        self.unit_cross_section = 1e27/(0.389) #GeV^-2
        self.unit_density = (0.197)**3 * 1e-39 #GeV^4
        self.unit_amu = 0.931494 #GeV
        self.unit_fermi = 1/0.1975 #fm^-1
        self.unit_η = 2.998 * 1e5 #unitless
        self.unit_prefactor = 24*60*60*1e46/(6.58*1.8) #1/[kg day KeV] 
        self.unit_vmin = 2.998*1e5 # km/s
        
        # Unit conversion
        self.ρ0 = self.ρ0 * self.unit_density# GeV^4
        self.mT = self.mT * self.unit_amu# GeV
        self.mN = 0.939 #GeV
        self.yr = 365*24 #days

    def ff_helm(self,E):
        s = 0.9
        a = 0.52
        c = (1.23*(self.A**(1./3.)) - 0.60)
        R = np.sqrt(c**2 + 7*(np.pi**2)*(a**2)/3. - 5*(s**2))
        
        q = np.sqrt(2*self.mT*E*1e-6)*self.unit_fermi #fm^-1
        x = q*R
        j1 = (np.sin(x)-x*np.cos(x))/x**2
        return 3*j1*np.exp(-(q*s)**2/2.)/(q*R)

    def eta_fn(self, vmin):
        return get_eta(vmin, self.vE, self.vdfE, self.vesc, self.vcirc)

    def prefactor(self, mdm, sigma_p, E):
        sigma_p = sigma_p*















