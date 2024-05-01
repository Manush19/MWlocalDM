import numpy as np
import scipy as sp
from scipy.signal import medfilt
from scipy.stats import poisson
from scipy.optimize import minimize, fsolve, bisect
from project.constants import MW_params as pm
from iminuit import Minuit
import project.quadrantHopping as quadH
import warnings
import pickle
import math
from tqdm.notebook import tqdm

MW_dict = pickle.load(open('../Output/MW_dict.pkl','rb'))
MWlike = pickle.load(open('../Output/MWlike_dict.pkl','rb'))
mwd = MW_dict['vdf_RCfit']
mwgals = MWlike['mwgals']
mwld = MWlike['vdf_RCfit']

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
        bint[bint <= 0] = 1e-32
        return {'Neachbin': bint,
                'E_center': Eint,
                'E_edges': bin_edges,
                'accuracy': accuracy,
                'mdm': mdm,
                'σp': σp,
                'bl': bl}

    def binTot_mdm_array(self, bin_edges, Mdm=np.linspace(1, 10, 1000), σ0=1e-46, accuracy = 10):
        bintots = []
        for mdm in Mdm:
            bintots.append(self.binTot(mdm, σ0, bl=0, bin_edges=bin_edges))
        self.bintots = bintots
        self.mdm_array = Mdm
        self.σ0 = σ0
    
    def mocksample(self, mdm, σp, bl, Nbins = 50, Ntot='mean', seed=None, **kwargs):
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

        if seed:
            np.random.seed(seed)
        N = self.totNtot(mdm, σp, bl, E=E_array)
        
        if Ntot == 'mean':
            N = N
        elif Ntot == 'poisson':
            N = np.random.poisson(N, 1)
        else:
            print ('Please give a valid key word Ntot')
            return None

        Esample = []
        for u in np.random.uniform(0, 1, size=int(np.floor(N))):
            index = (np.abs(cdf - u).argmin())
            Esample.append(E_array[index])

        hist = np.histogram(Esample, Nbins)
            
        return {'Esample': np.array(Esample), 
                'E_array': E_array,
                'binned_Esample': hist[0],
                'bin_edges': hist[1],
                'pdf': pdf,
                'cdf': cdf,
                'pdfsg': pdfsg,
                'pdfbg': pdfbg,
                'N_obs': int(np.floor(N)),
                'Nbins': Nbins}





class MLE:
    def __init__(self, mock, nr_init, globalmin='smart', 
                 likelihood=1, **kwargs):
        """
        If likelilhood == 2, please provide the gal and idx
        keywords arguments for the galaxy and chain no of the
        VDF and ρ_sun to be used.
        """
        self.mock = mock
        self.nr = nr_init
        self.nobs = self.mock['binned_Esample']
        self.ΔE = (self.nr.Eroi - self.nr.Ethr)/self.mock['Nbins']
        self.exp = self.nr.exposure

        self.gal = kwargs.get('gal') if 'gal' in kwargs.keys() else 'MW'

        self.likelihood = likelihood
        if self.likelihood == 1:
            self.nlL = self.nlL1
        elif self.likelihood == 2:
            self.mwD = self.get_mwD(self.gal)
            self.nlL = self.nlL2
        elif self.likelihood == 3:
            self.gals = ['MW'] + mwgals
            self.nlL = self.nlL3
            
        if 'chainlen' in kwargs.keys():
            self.chainlen = kwargs.get('chainlen')
        else:
            self.chainlen = len(mwd['vdfEs'])
        # print (f'Running with chainlen = {self.chainlen}')
        
        if globalmin:
            self.globalmin(Mdm=globalmin)
        else:
            self.ran_globalmin = False

    def globalmin(self, Mdm = 'smart'):
        self.ran_globalmin = True
        if isinstance(Mdm, str) and Mdm == 'smart':
            Mdm_try1 = np.linspace(1,10,20)
            self.fd_globalmin(Mdm_try1)
            indx = np.where(self.Tq <= 5.99)[0]
            mdm_low = Mdm_try1[indx[0]-1]
            mdm_hig = Mdm_try1[indx[-1]+1]           
            Mdm_try2 = np.linspace(mdm_low, mdm_hig, 100)
            self.fd_globalmin(Mdm_try2)

        elif isinstance(Mdm, str) and Mdm == 'precise':
            Mdm_try1 = np.linspace(1,10,20)
            self.fd_globalmin(Mdm_try1)
            indx = np.where(self.Tq == np.min(self.Tq))[0]
            mdm_low = Mdm_try1[indx[0]-1]
            mdm_hig = Mdm_try1[indx[-1]+1]
            Mdm_try2 = np.linspace(mdm_low, mdm_hig, 10)
            self.fd_globalmin(Mdm_try2)
            indx = np.where(self.Tq == np.min(self.Tq))[0]
            mdm_low = Mdm_try2[indx[0]-1]
            mdm_hig = Mdm_try2[indx[-1]+1]
            Mdm_try3 = np.linspace(mdm_low, mdm_hig, 10)
            self.fd_globalmin(Mdm_try3)

        elif isinstance(Mdm, np.ndarray):
            self.fd_globalmin(Mdm)
        else:
            self.fd_globalmin(np.linspace(1,10,10))

    def fd_globalmin(self, Mdm):
        self.Mdm = Mdm
        self.λsg0s = self.get_λsg0s()
        best = self.get_best()
        self.Sdm, self.Nll = best

        self.minindx = np.where(self.Nll == np.min(self.Nll))[0][0]
        self.mdm_min = self.Mdm[self.minindx]
        self.sdm_min = self.Sdm[self.minindx]
        self.nll_min = self.Nll[self.minindx]
        self.Tq = 2*(self.Nll - self.nll_min)

    def get_Tgrid(self):
        if not self.ran_globalmin:
            print ('self.globalmin() is running for the first time')
            self.globalmin()

        Mdm = np.linspace(2,5,30)
        Sdm = np.logspace(-46, np.log10(3e-45), 30)
        self.Mdm = Mdm
        self.λsg0s = self.get_λsg0s()
        Mgrid = np.zeros([self.Mdm.size]*2)
        Sgrid = np.zeros([self.Mdm.size]*2)
        Lgrid = np.zeros([self.Mdm.size]*2)
        for mi, mdm in enumerate(Mdm):
            for si, sdm in enumerate(Sdm):
                Mgrid[mi,si] = mdm
                Sgrid[mi,si] = sdm
                Lgrid[mi,si] = self.lsdm̂_func(np.log10(sdm), mdm)
        Tgrid = 2*(Lgrid - np.min(Lgrid))
        return Mgrid, Sgrid, Lgrid, Tgrid
        

    def get_limits(self, tq_limit=5.99):
        if not self.ran_globalmin:
            print ('self.globalmin() is running for the first time')
            self.globalmin()
            
        Mdm_ = []
        Sdm_upp = []
        Sdm_low = []
        for i,mdm in enumerate(self.Mdm):
            if self.Tq[i] > 5.99:
                continue
            def func(lsdm):
                nll = self.lsdm̂_func(lsdm, mdm)
                tq = 2*(nll - self.nll_min)
                return tq - tq_limit
            solupp = bisect(func, np.log10(self.Sdm[i]), -44)
            sollow = bisect(func, -46, np.log10(self.Sdm[i]))
            
            Sdm_upp.append(math.pow(10, solupp))
            Sdm_low.append(math.pow(10, sollow))
            Mdm_.append(mdm)
        return [np.array(Mdm_), np.array(Sdm_upp), 
                np.array(Sdm_low)]

    def get_mwD(self, gal):
        if gal == 'MW':
            return mwd
        elif gal in mwgals:
            return mwld[gal]
        else:
            print (f'gal = {gal} is not MW or MWlike')

    def jpdf_max_idx(self, gal):
        return 100

    def get_λsg0s(self):
        if self.likelihood == 1:
            λsg0s = []
            for mdm in self.Mdm:
                binT = self.nr.binTot(mdm, 1e-46, 0.,
                                      self.mock['bin_edges'])
                λsg0s.append(binT['Neachbin'])
            return λsg0s
        elif self.likelihood == 2:
            λsg0s_chains = []
            for ci in range(self.chainlen):
                if self.chainlen == 1:
                    ci = self.jpdf_max_idx(self.gal)
                nr_ = Nuclear(self.nr.element, vE=self.mwD['vE'],
                              vdfE = self.mwD['vdfEs'][ci],
                              vesc = self.mwD['vescs'][ci],
                              vcirc = self.mwD['vcircs'][ci],
                              rhosun = self.mwD['rhosuns'][ci],
                              Ethr = self.nr.Ethr,
                              Eroi = self.nr.Eroi,
                              ω = self.nr.ω)
                λsg0s = []
                for mdm in self.Mdm:
                    binT = nr_.binTot(mdm, 1e-46, 0., 
                                      self.mock['bin_edges'])
                    λsg0s.append(binT['Neachbin'])
                λsg0s_chains.append(λsg0s)
            return λsg0s_chains
        elif self.likelihood == 3:
            λsg0s_gals = []
            for gal in self.gals:
                mwD = self.get_mwD(gal)
                λsg0s_chains = []
                for ci in range(self.chainlen):
                    if self.chainlen == 1:
                        ci = self.jpdf_max_idx(gal)
                    nr_ = Nuclear(self.nr.element, vE=mwD['vE'],
                                  vdfE = mwD['vdfEs'][ci],
                                  vesc = mwD['vescs'][ci],
                                  vcirc = mwD['vcircs'][ci],
                                  rhosun = mwD['rhosuns'][ci],
                                  Ethr = self.nr.Ethr,
                                  Eroi = self.nr.Eroi,
                                  ω = self.nr.ω)
                    λsg0s = []
                    for mdm in self.Mdm:
                        binT = nr_.binTot(mdm, 1e-46, 0., 
                                      self.mock['bin_edges'])
                        λsg0s.append(binT['Neachbin'])
                    λsg0s_chains.append(λsg0s)
                λsg0s_gals.append(λsg0s_chains)
            return λsg0s_gals
                    
    def λsg0(self, mdm, idx=None, gal=None):
        indx = np.where(self.Mdm == mdm)[0][0]
        if self.likelihood == 1:
            return self.λsg0s[indx]
        elif self.likelihood == 2:
            return self.λsg0s[idx][indx]
        elif self.likelihood == 3:
            gali = self.gals.index(gal)
            return self.λsg0s[gali][idx][indx]

    def λsg(self, mdm, sdm, idx=None, gal=None):
        return self.λsg0(mdm, idx, gal)*sdm/1e-46

    def nlL1(self, mdm, sdm, bl, idx=None, gal=None):
        λ = self.λsg(mdm, sdm, idx, gal) + self.exp*self.ΔE*bl
        lL = np.sum(self.nobs*np.log(λ)) - np.sum(λ)
        nlL = -lL
        nlL = 1e32 if np.isnan(nlL) else nlL
        return nlL

    def nlL2(self, mdm, sdm, bl, idx, gal):
        nlL = self.nlL1(mdm, sdm, bl, idx, gal)
        # nlL -= np.log(np.prod(self.get_mwD(gal)['par_pdfs'][idx]))
        nlL = 1e32 if np.isnan(nlL) else nlL
        return nlL

    def nlL3(self, mdm, sdm, bls, idxs, gals):
        nll_gals = []
        for gal, idx, bl in zip(gals, idxs, bls):
            nll_gals.append(self.nlL2(mdm, sdm, bl, idx, gal))
        nlL = np.array(nll_gals).sum()
        nlL = 1e32 if np.isnan(nlL) else nlL
        return nlL

    def get_best(self):
        Sdm, Nll = [],[]
        for mdm in self.Mdm:
            sdm, nll = self.min_sdm(mdm)
            Sdm.append(sdm)
            Nll.append(nll)
        return (np.array(Sdm), np.array(Nll))

    def min_sdm(self, mdm):
            minz = minimize(self.lsdm̂_func, x0=[-45], args=(mdm))
            return (math.pow(10, minz.x[0]), minz.fun)

    def lsdm̂_func(self, lsdm, mdm):
        if lsdm > -30:
            return 1e32
        sdm = math.pow(10, lsdm)
        if self.likelihood == 1:
            blm = fsolve(self.bl̂_func, [0.001], args=(mdm, sdm))[0]
            return self.nlL1(mdm, sdm, blm)
        elif self.likelihood == 2:
            nlls = []
            for idx in range(self.chainlen):
                blm = fsolve(self.bl̂_func, [0.001], 
                             args=(mdm,sdm,idx,self.gal))[0]
                nlls.append(self.nlL2(mdm, sdm, blm, idx, self.gal))
            return np.min(nlls)
        elif self.likelihood == 3:
            idxs, bls = [], []
            for gal in self.gals:
                blms, nlls = [], []
                for idx in range(self.chainlen):
                    blm = fsolve(self.bl̂_func, [0.001], 
                                 args=(mdm, sdm, idx, gal))[0]
                    blms.append(blm)
                    nlls.append(self.nlL2(mdm, sdm, blm, idx, gal))
                nlls = np.array(nlls)
                idx_min = np.where(nlls == nlls.min())[0][0]
                idxs.append(idx_min)
                bls.append(blms[idx_min])
            return self.nlL3(mdm, sdm, bls, idxs, self.gals)

    def bl̂_func(self, bl, mdm, sdm, idx=None, gal=None):
        λ = self.λsg(mdm,sdm,idx,gal) + self.exp*bl*self.ΔE
        return np.mean(self.nobs/λ) - 1.0

    def min_bl(self, mdm, sdm, idx=None, gal=None):
        fsol = fsolve(self.bl̂_func, x0=[0.001], args=(mdm,sdm,idx,gal))
        return fsol[0]






