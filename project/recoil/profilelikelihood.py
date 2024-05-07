import math 
import numpy as np 
from numpy.core.function_base import linspace as linspace
import pandas as pd
from scipy.optimize import fsolve, bisect, minimize 
import sys, os, pickle 
from tqdm.notebook import tqdm 
from multiprocessing import Pool, cpu_count

sys.path.append('../../')
from project.recoil.nuclear import Nuclear as Nr
from project.recoil.nuclear import norm, p50

MWlike = pickle.load(open('../../Output/MWlike_dict.pkl','rb'))
mwld = MWlike['vdf_RCfit']
mwd = mwld['MW']
mwgals = MWlike['gals']

norm = lambda f, x: f/np.trapz(f, x) if not np.all(f == 0) else f
p50 = lambda x: np.percentile(x, 50)
pQ = lambda x, Q: np.percentile(x, Q)





class MLE:
    def __init__(self, mock, nr_init, Mdm=np.linspace(1,10,20), run_globalmin=True):
        self.mock = mock
        self.nr = nr_init
        self.nobs = self.mock['binned_Esample']
        self.ΔE = (self.nr.Eroi - self.nr.Ethr)/self.mock['Nbins']
        self.exp = self.nr.exposure
        self.ran_globalmin = False
        self.λsg0s = np.zeros([1])
        self.Mdm = np.around(Mdm, 2)
        if run_globalmin:
            self.globalmin(self.Mdm)

    def globalmin(self, Mdm):
        self.ran_globalmin = True
        self.Mdm = np.around(Mdm, 2)
        self.λsg0s = self.get_λsg0s(self.Mdm)
        self.Sdm, self.Nll = self.get_best(self.Mdm)

        self.minindx = np.where(self.Nll == np.min(self.Nll))[0][0]
        self.mdm_min = self.Mdm[self.minindx]
        self.sdm_min = self.Sdm[self.minindx]
        self.nll_min = self.Nll[self.minindx]
        self.Tq = 2*(self.Nll - self.nll_min)

    def jpdf_max_idx(self, gal):
        mwD = self.get_mwD(gal)
        pdfs = mwD['par_pdfs']
        jpdfs = np.prod(pdfs, axis=1)
        index = np.where(jpdfs == np.max(jpdfs))
        return index[0][0]

    def get_mwD(self, gal):
        if gal == 'MW':
            return mwd 
        elif gal in mwgals:
            return mwld[gal]
        
    def get_best(self, Mdm):
        Sdm, Nll = [], []
        for mdm in Mdm:
            sdm, nll = self.min_sdm(mdm)
            Sdm.append(sdm)
            Nll.append(nll)
        return (np.array(Sdm), np.array(Nll))
        
    def min_sdm(self, mdm):
        minz = minimize(self.lsdm_func, x0=[-45.], args=(mdm))
        return (math.pow(10, minz.x[0]), minz.fun)
    
    def get_limits(self, Mdm, tq_limit=5.99):
        if not self.ran_globalmin:
            print (f'Running self.globalmin() for the first time')
            self.globalmin(Mdm)
        Mdm_lim = []
        Sdm_upp = []
        Sdm_low = []
        for i,mdm in enumerate(Mdm):
            if self.Tq[i] > tq_limit:
                continue
            def func(lsdm):
                nll = self.lsdm_func(lsdm, mdm)
                tq = 2*(nll - self.nll_min)
                return tq - tq_limit
            solupp = bisect(func, np.log10(self.Sdm[i]), -44.0)
            sollow = bisect(func, -46.0, np.log10(self.Sdm[i]))
            Sdm_upp.append(math.pow(10, solupp))
            Sdm_low.append(math.pow(10, sollow))
            Mdm_lim.append(mdm)
        return (np.array(Mdm_lim), np.array(Sdm_upp), np.array(Sdm_low))
    
    def get_Tgrid(self, Mdm=np.linspace(2,6,40), Sdm = np.logspace(-46., -44, 40)):
        if not self.ran_globalmin:
            print ('self.globalmin() is running for the first time.')
            self.globalmin(Mdm)
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
    
    def get_λsg0s(self, Mdm, datafile = '../../Output/rate_prediction.pkl'):
        if not os.path.exists(datafile):
            self.init_λsg0s(datafile)
        λsg0s = self.filter_and_write(Mdm, write=True, datafile=datafile)
        return λsg0s

    def init_λsg0s(self, datafile):
        λsg0s = {'statistical': {}, 
                 'massmodel': {gal: {} for gal in ['MW']+mwgals},
                 'sample': {}}
        pickle.dump(λsg0s, open(datafile,'wb'))
        




class StatisticalUncertanity(MLE):
    def __init__(self, mock, nr_init, Mdm=np.linspace(1,10,20), globalmin=True):
        super().__init__(mock, nr_init, Mdm, globalmin)

    def filter_and_write(self, Mdm, write=True, datafile='../../Output/rate_prediction.pkl'):
        Mdm = np.around(Mdm, 2)
        λsg0d = pickle.load(open(datafile, 'rb'))
        λsg0s = λsg0d['statistical'].copy()
        mdm_keys = np.array(list(λsg0s.keys()))
        for mdm in Mdm:
            if len(mdm_keys) > 0:
                if np.min(np.abs(mdm_keys - mdm)) < 0.01:
                    continue 
            λsg0s[mdm] = self.compute_λsg0s(mdm)
        if write:
            λsg0d['statistical'] = λsg0s
            pickle.dump(λsg0d, open(datafile, 'wb'))
        return λsg0s
    
    def compute_λsg0s(self, mdm):
        binT = self.nr.binTot(mdm, 1e-46, 0., self.mock['bin_edges'])
        return binT['Neachbin']
    
    def λsg0(self, mdm):
        mdm_keys = np.array(list(self.λsg0s.keys()))
        mdm_idx = np.argmin(np.abs(mdm_keys - mdm))
        return self.λsg0s[mdm_keys[mdm_idx]]
    
    def λsg(self, mdm, sdm):
        return self.λsg0(mdm)*sdm/1e-46
    
    def nlL(self, mdm, sdm, bl):
        λ = self.λsg(mdm, sdm) + self.exp*self.ΔE*bl 
        lL = np.sum(self.nobs*np.log(λ)) - np.sum(λ)
        nlL = -lL
        nlL = 1e32 if np.isnan(nlL) else nlL
        return nlL
    
    def lsdm_func(self, lsdm, mdm):
        if lsdm > -30:
            return 1e32
        sdm = math.pow(10, lsdm)
        blm = fsolve(self.bl_func, [0.001], args=(mdm, sdm))[0]
        return self.nlL(mdm, sdm, blm)
    
    def bl_func(self, bl, mdm, sdm):
        λ = self.λsg(mdm, sdm) + self.exp*bl*self.ΔE
        return np.mean(self.nobs/λ) - 1.0





class MassModelUncertanity(MLE):
    def __init__(self, mock, nr_init, Mdm=np.linspace(1,10,20), globalmin=True, chainlen='all', gal='MW', fixed_rhosun=True):
        self.gal = gal
        self.mwD = self.get_mwD(self.gal)
        self.fixed_rhosun = fixed_rhosun
        if isinstance(chainlen, str) and chainlen == 'all':
            self.chainlen = len(self.mwD['vdfEs'])
        elif isinstance(chainlen, int) and chainlen < 10000:
            self.chainlen = chainlen
        super().__init__(mock, nr_init, Mdm, globalmin)

    def filter_and_write(self, Mdm, write=True, datafile='../../Output/rate_prediction.pkl'):
        Mdm = np.around(Mdm, 2)
        λsg0d = pickle.load(open(datafile, 'rb'))
        λsg0s = λsg0d['massmodel'][self.gal].copy()
        chain_keys = list(λsg0s.keys())
        for ci in tqdm(range(self.chainlen)):
            if not ci in chain_keys:
                λsg0s[ci] = {}
                for mdm in Mdm:
                    λsg0s[ci][mdm] = self.compute_λsg0s(mdm, ci)
            else:
                mdm_keys = np.array(list(λsg0s[ci].keys()))
                for mdm in Mdm:
                    if np.min(np.abs(mdm_keys - mdm)) > 0.01:
                        λsg0s[ci][mdm] = self.compute_λsg0s(mdm, ci)
        if write:
            λsg0d['massmodel'][self.gal] = λsg0s
            pickle.dump(λsg0d, open(datafile, 'wb'))
        return λsg0s
    
    def compute_λsg0s(self, mdm, chain_idx):
        vE = self.mwD['vE']
        vdfE = self.mwD['vdfEs'][chain_idx]
        vesc = self.mwD['vescs'][chain_idx]
        vcirc = self.mwD['vcircs'][chain_idx]
        if self.fixed_rhosun:
            rhosun = p50(self.mwD['rhosuns'])
        else:
            rhosun = self.mwD['rhosuns'][chain_idx]
        nr_ = Nr(self.nr.element, vE=vE, vdfE=vdfE, vesc=vesc, vcirc=vcirc, rhosun=rhosun, Ethr=self.nr.Ethr, Eroi=self.nr.Eroi, ω=self.nr.ω)
        binT = nr_.binTot(mdm, 1e-46, 0., self.mock['bin_edges'])
        return binT['Neachbin']

    
    def λsg0(self, mdm, chain_idx):
        mdm_keys = np.array(list(self.λsg0s[chain_idx].keys()))
        mdm_idx = np.argmin(np.abs(mdm_keys - mdm))
        return self.λsg0s[chain_idx][mdm_keys[mdm_idx]]
    
    def λsg(self, mdm, sdm, chain_idx):
        return self.λsg0(mdm, chain_idx)*sdm/1e-46
    
    def nlL(self, mdm, sdm, bl, chain_idx):
        λ = self.λsg(mdm, sdm, chain_idx) + self.exp*self.ΔE*bl
        lL = np.sum(self.nobs*np.log(λ)) - np.sum(λ)
        nlL = -lL
        nlL = 1e32 if np.isnan(nlL) else nlL
        return nlL
    
    def lsdm_func(self, lsdm, mdm):
        if lsdm > -30:
            return 1e32
        sdm = math.pow(10, lsdm)
        nlls = []
        for chain_idx in range(self.chainlen):
            blm = fsolve(self.bl_func, [0.001], args=(mdm, sdm, chain_idx))[0]
            nlls.append(self.nlL(mdm, sdm, blm, chain_idx))
        return np.min(nlls)
    
    def bl_func(self, bl, mdm, sdm, chain_idx):
        λ = self.λsg(mdm, sdm, chain_idx) + self.exp*bl*self.ΔE
        return np.mean(self.nobs/λ) - 1.0
    




class SampleUncertanity(MLE):
    def __init__(self, mock, nr_init, Mdm=np.linspace(1, 10, 20), percentiles=np.arange(5, 96, 2), run_parallel=False, run_globalmin=True, fixed_rhosun=True):
        self.gals = mwgals
        self.percentiles = percentiles
        self.fixed_rhosun = fixed_rhosun
        self.run_parallel = run_parallel
        self.vE = mwld['MW']['vE']
        self.VdfEs, self.Vescs, self.Vcrics, self.Rhosuns  = self.vdf_info()
        super().__init__(mock, nr_init, Mdm, run_globalmin)

    def vdf_info(self):
        VdfEs, Vescs, Vcircs, Rhosuns = [], [], [], []
        for gal in mwgals:
            VdfEs.append(mwld[gal]['vdfEs'])
            Vescs.append(mwld[gal]['vescs'])
            Vcircs.append(mwld[gal]['vcircs'])
            Rhosuns.append(mwld[gal]['rhosuns'])
        return (np.vstack(VdfEs).T, np.concatenate(Vescs), np.concatenate(Vcircs), np.concatenate(Rhosuns))
    
    def filter_and_write(self, Mdm, write=True, datafile='../../Output/rate_prediction.pkl'):
        Mdm = np.around(Mdm, 2)
        λsg0d = pickle.load(open(datafile, 'rb'))
        λsg0s = λsg0d['sample'].copy()
        percentiles = list(λsg0s.keys())
        for per in tqdm(self.percentiles):
            if not per in percentiles:
                λsg0s[per] = {}
                if self.run_parallel:
                    λsg0s[per] = self.parallel_compute(Mdm, per)
                else:
                    for mdm in Mdm:
                        λsg0s[per][mdm] = self.compute_λsg0s(mdm, per)
            else:
                mdm_keys = np.array(list(λsg0s[per].keys()))
                for mdm in Mdm:
                    if np.min(np.abs(mdm_keys - mdm)) > 0.01:
                        λsg0s[per][mdm] = self.compute_λsg0s(mdm, per)               
        if write:
            λsg0d['sample'] = λsg0s
            pickle.dump(λsg0d, open(datafile, 'wb'))
        return λsg0s
    
    def parallel_func(self, mdm):
        return self.compute_λsg0s(mdm, self.percentile_here)

    def parallel_compute(self, Mdm, percentile):
        self.percentile_here = percentile
        num_cores = 6 #cpu_count()
        with Pool(processes=num_cores) as pool:
            results = list(pool.imap(self.parallel_func, Mdm))
        res_dict = {Mdm[i]: results[i] for i in range(len(Mdm))}
        return res_dict
        
    def get_nr(self, percentile):
        vdfE = norm(np.percentile(self.VdfEs, percentile, axis=1), self.vE)
        vesc = np.percentile(self.Vescs, percentile)
        vcirc = np.percentile(self.Vcrics, percentile)
        if self.fixed_rhosun:
            rhosun = p50(mwd['rhosuns'])
        else:
            rhosun = np.percentile(self.Rhosuns, percentile)
        return Nr(self.nr.element, vE=self.vE, vdfE=vdfE, vesc=vesc, vcirc=vcirc, rhosun=rhosun, Ethr=self.nr.Ethr, Eroi=self.nr.Eroi, ω=self.nr.ω)
    
    def compute_λsg0s(self, mdm, percentile):
        nr_ = self.get_nr(percentile=percentile)
        binT = nr_.binTot(mdm, 1e-46, 0., self.mock['bin_edges'])
        return binT['Neachbin']
    
    def λsg0(self, mdm, percentile):
        mdm_keys = np.array(list(self.λsg0s[percentile].keys()))
        mdm_idx = np.argmin(np.abs(mdm_keys - mdm))
        return self.λsg0s[percentile][mdm_keys[mdm_idx]]
    
    def λsg(self, mdm, sdm, percentile):
        return self.λsg0(mdm, percentile)*sdm/1e-46
    
    def nlL(self, mdm, sdm, bl, percentile):
        λ = self.λsg(mdm, sdm, percentile) + self.exp*self.ΔE*bl
        lL = np.sum(self.nobs*np.log(λ)) - np.sum(λ)
        nlL = -lL
        nlL = 1e32 if np.isnan(nlL) else nlL
        return nlL
    
    def lsdm_func(self, lsdm, mdm):
        if lsdm > -30:
            return 1e32
        sdm = math.pow(10, lsdm)
        nlls = []
        for percentile in self.percentiles:
            blm = fsolve(self.bl_func, [0.001], args=(mdm, sdm, percentile))[0]
            nlls.append(self.nlL(mdm, sdm, blm, percentile))
        return np.min(nlls)
    
    def bl_func(self, bl, mdm, sdm, percentile):
        λ = self.λsg(mdm, sdm, percentile) + self.exp*bl*self.ΔE
        return np.mean(self.nobs/λ) - 1.0