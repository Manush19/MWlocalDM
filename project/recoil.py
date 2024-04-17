import numpy as np
import scipy as sp
from scipy.signal import medfilt
from scipy.stats import poisson
from project.constants import MW_params as pm

norm = lambda f, x: f/np.trapz(f, x) if not np.all(f == 0) else f
p50 = lambda x: np.percentile(x, 50)

def vdf_fn(v_, v, vdf): 
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
        self.exposure = 1e3*365 * self.ω
        self.Ethr = kwargs.get('Ethr') if 'Ethr' in kwargs.keys() else 0.1
        self.Eroi = kwargs.get('Eroi') if 'Eroi' in kwargs.keys() else 5.0
        self.E = kwargs.get('E') if 'E' in kwargs.keys() else np.linspace(self.Ethr, self.Eroi, 300)
        
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
        
    def diffSg(self,mdm,σp, **kwargs): # Formarly Rate
        E = kwargs.get('E') if 'E' in kwargs.keys() else self.E

        if mdm == 0:
            return np.ones(np.shape(E))*1e-15
        
        μT = mdm*self.mT/(mdm+self.mT)
        rate = []
        for E_ in E:
            vmin = np.sqrt(self.mT*E_*1e-6/(2*μT**2)) * self.unit_vmin
            η = self.velInt([vmin])

            rate.append(self.preFactor(mdm, σp, E_) * η[0])
        return np.array(rate) * self.unit_prefactor * self.unit_η

    def diffBg(self, bl = 0., **kwargs):
        E = kwargs.get('E') if 'E' in kwargs.keys() else self.E
        bl = 1e-32 if bl <= 0 else bl
        return np.ones(np.shape(E)) * bl

    def diffTot(self, mdm, σp, bl = 0., **kwargs):
        return self.diffSg(mdm, σp, **kwargs) + self.diffBg(bl, **kwargs)

    def totNsg(self, mdm, σp, **kwargs):
        E = kwargs.get('E') if 'E' in kwargs.keys() else self.E
        ω = kwargs.get('ω') if 'ω' in kwargs.keys() else self.ω
        Ethr = kwargs.get('Ethr') if 'Ethr' in kwargs.keys() else self.Ethr
        E = E[E >= Ethr]
        exposure = 1e3*365*ω
        return exposure*np.trapz(self.diffSg(mdm, σp, E = E), E)

    def totNbg(self, bl, **kwargs):
        E = kwargs.get('E') if 'E' in kwargs.keys() else self.E
        ω = kwargs.get('ω') if 'ω' in kwargs.keys() else self.ω
        Ethr = kwargs.get('Ethr') if 'Ethr' in kwargs.keys() else self.Ethr
        E = E[E >= Ethr]
        exposure = 1e3*365*ω
        # return exposure*np.trapz(self.diffBg(bl, E = E), E)
        return exposure*bl*(E[-1] - E[0])
    
    def totNtot(self, mdm, σp, bl=0., **kwargs):
        return self.totNsg(mdm, σp, **kwargs) + self.totNbg(bl, **kwargs)

    def totNgrid(self, Mgrid, Sgrid, blgrid, σ0=1e-46, **kwargs):
        Mdm = Mgrid[0,:]
        Sdm = Sgrid[:,0]

        Nsg = np.zeros(Mgrid.shape)
        Nbg = np.zeros(Mgrid.shape)

        σ0_by_N0 = self.σpMdmNsg(Mdm, N=1, σ0=σ0, **kwargs)
        for i in range(Mgrid.shape[0]):
            Nsg[i,:] = Sdm[i] / σ0_by_N0
                
        for i in range(blgrid.shape[0]):
            for j in range(blgrid.shape[1]):
                Nbg[i,j] = self.totNbg(blgrid[i,j], **kwargs)
                
        return Nsg + Nbg

    def σpMdmNsg(self, Mdm, N=1, σ0=1e-46, **kwargs):
        isfloat = True if isinstance(Mdm, float) else False
        Mdm = np.array([Mdm]) if isfloat else Mdm
        N0 = np.zeros(Mdm.shape)
        for i, mdm in enumerate(Mdm):
            N0[i] = self.totNsg(mdm, σ0, **kwargs)
        N0[N0 <= 0] = 1e-32
        Sdm = σ0/N0
        return Sdm

    def binTot(self, mdm, σp, bl, bin_edges, accuracy=10):
        # No kwargs, Ethr, ω, Eroi etc, should be specified in the namespace
        Nbins = len(bin_edges) - 1
        bint = np.zeros(Nbins)
        Eint = np.zeros(Nbins)
        for i in range(Nbins):
            E = np.linspace(bin_edges[i], bin_edges[i + 1], accuracy)
            bint[i] = self.totNtot(mdm, σp, bl, E = E)
            Eint[i] = 0.5*(bin_edges[i] + bin_edges[i + 1])
        return {'Neachbin': bint,
                'E_center': Eint,
                'E_edges': bin_edges,
                'accuracy': accuracy,
                'mdm': mdm,
                'σp': σp,
                'bl': bl}


    def mocksample(self, mdm, σp, bl=0., Ntot='mean', seed=None, **kwargs):
        """
        Do initialize E, Ethr, omega in the namespace.
        """
        E_array = np.linspace(self.Ethr, self.Eroi, 500)
        
        pdfsg = self.diffSg(mdm, σp, E=E_array)
        pdfbg = self.diffBg(bl, E=E_array)
        pdfsg = np.zeros(E_array.shape) if mdm == 0 else pdfsg
        pdfbg = np.zeros(E_array.shape) if bl == 0 else pdfbg
        pdf = pdfsg + pdfbg
        pdf = norm(pdf, E_array)

        cdf = np.cumsum(pdf)
        cdf = cdf/cdf[-1]

        N = self.totNtot(mdm, σp, bl, E=E_array)
        if Ntot == 'mean':
            N = N
        elif Ntot == 'poisson':
            N = np.random.poisson(N, 1)

        if seed:
            np.random.seed(seed)

        Esample = []
        for u in np.random.uniform(0, 1, size=int(np.floor(N))):
            index = (np.abs(cdf - u).argmin())
            Esample.append(E_array[index])
            
        return {'Esample': np.array(Esample), 
                'E_array': E_array,
                'pdf': pdf,
                'cdf': cdf,
                'pdfsg': pdfsg,
                'pdfbg': pdfbg,
                'N_obs': int(np.floor(N))}




def binSamp(sample, bin_edges, accuracy):
    Nbins = len(bin_edges) - 1
    binned = np.histogram(sample, Nbins)
    E_array = []
    E_center = []
    for i in range(Nbins):
        E_array.extend(np.linspace(binned[1][i], binned[1][i+1], accuracy))
        E_center.append(0.5*(binned[1][i] + binned[1][i+1]))
    binnedE = np.array(E_array)
    Ntot = np.size(sample)
    return {'binnedsample': binned,
            'binnedE': binnedE,
            'Ntot': Ntot,
            'bin_edges': bin_edges,
            'Nbins': Nbins,
            'E_center': E_center,
            'accuracy': accuracy}

class ProfileLikelihood:
    def __init__(self, nr, sample_dict, **kwargs):
        self.nr = nr
        self.s = sample_dict
        self.ni = bsamp['binnedsample'][0]
        self.bins = bsamp['bins']
        

    def findllnorm(self, mdm = 5., σp = 1e-45, bl = 0.001):
        self.mdm_sdm(mdm, σp)
        self.llnorm = 1.0
        norm = np.abs(self.nllike(bl))
        print (norm)
        return norm

    def mdm_sdm(self, mdm, σp):
        self.mdm = mdm
        self.σp = σp

    def nllike(self, bl):
        E, λ = self.nr.binTot(self.bins, self.mdm, self.σp, bl)
        ll = np.sum(self.ni * np.log(λ)) - np.sum(λ)
        if np.isfinite(ll) == False:
            ll = np.nan_to_num(ll, nan=-1e32, posinf=-1e32, neginf=-1e-32)
        return -ll

    # If Npred is >> Ntot or Npred << Ntot likelihood can give a very small number.

        
        







