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
        self.ve = vE
        self.vdf_e = vdfE
        self.vesc = vesc
        self.vcirc= vcirc
        self.vert = self.vcirc*(1.05+0.07)
        self.ρ0 = rhosun
        
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

        self.ω = kwargs.get('ω') if 'ω' in kwargs.keys() else 1
        self.ω = 1e3*365 * self.ω
        self.Ethr = kwawrgs.get('Ethr') if 'Ethr' in kwargs.keys() else 0.1
        self.E = kwargs.get('E') if 'E' in kwargs.keys() else np.logspace(np.log10(self.Ethr), 2, 1000)
        
        self.vmin = np.logspace(-5,3,1000)

    def fHelm(self,E):
        # s = 1
        # R_ = 1.2*self.A**(1/3)
        # R = np.sqrt(R_**2 - 5*s**2)
        s = 0.9
        a = 0.52
        c = (1.23*(self.A**(1./3.)) - 0.60)
        R = np.sqrt(c**2 + 7*(np.pi**2)*(a**2)/3. - 5*(s**2))
        
        q = np.sqrt(2*self.mT*E*1e-6)*self.unit_fermi #fm^-1
        x = q*R
        # j1 = sp.special.j1(x)
        j1 = (np.sin(x)-x*np.cos(x))/x**2
        return 3*j1*np.exp(-(q*s)**2/2.)/(q*R)
    
    def velInt(self,vmin):
        η = []
        for v in vmin:
            v_plus = np.min([v,self.vesc + self.vert])
            v_minus = np.min([v,self.vesc - self.vert])
            low1 = v_plus
            hig1 = self.vesc + self.vert
            if low1 <= hig1:
                idx1 = np.where((self.ve >= low1) & (self.ve <= hig1))
                idg1 = self.vdf_e[idx1]
                v1 = self.ve[idx1]
                int1 = np.trapz(idg1/v1,v1)
            else:
                int1 = 0
            low2 = v_minus
            hig2 = self.vesc - self.vert
            if low2 >= hig2:
                idx2 = np.where((self.ve >= low2) & (self.ve <= hig2))
                idg2 = self.vdf_e[idx2]
                v2 = self.ve[idx2]
                int2 = np.trapz(idg2/v2, v2)
            else:
                int2 = 0
            η.append(int1 - int2)
        return np.array(η)
    
    def preFactor(self,mdm,σp,E):
        σp = σp*self.unit_cross_section
        μN = mdm*self.mN/(mdm+self.mN)
        return σp*self.ρ0*(self.A**2)*(self.fHelm(E)**2)/(mdm*2*(μN**2))
        
    def diffRate(self,mdm,σp,E): # Formarly Rate
        self.x,self.y = [],[]
        μT = mdm*self.mT/(mdm+self.mT)
        rate = []
        for E_ in E:
            vmin = np.sqrt(self.mT*E_*1e-6/(2*μT**2)) * self.unit_vmin
            η = self.velInt([vmin])
            self.x.append(vmin)
            self.y.append(η[0])
            rate.append(self.preFactor(mdm,σp,E_) * η[0])
        return np.array(rate) * self.unit_prefactor * self.unit_η
    
    def totN(self, mdm, σp, **kwargs):
        # ω is in kg days (default 1 t yr)
        # Ethr is the threshold energy (default = 0.1 KeV)
        Ethr = kwargs.get('Ethr') if 'Ethr' in kwargs.keys() else self.Ethr
        E = kwargs.get('E') if 'E' in kwargs.keys() else self.E
        E = E[E > Ethr]
        ω = kwargs.get('ω') if 'ω' in kwargs.keys() else self.ω
        ω = 1e3*365 * ω

        return ω*np.trapz(self.diffRate(mdm,σp,E),E)











