import math
from time import perf_counter 
import numpy as np 
from numpy.core.function_base import linspace as linspace
import pandas as pd
from scipy.optimize import fsolve, bisect, minimize 
import sys, os, pickle 
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

sys.path.append('../../')
from project.recoil import Electron as El
from project.recoil import norm, p50, get_vdf_ert

MWlike = pickle.load(open('../../Output/MWlike_dict.pkl','rb'))
mwld = MWlike['vdf_RCfit']
mwd = mwld['MW']
mwgals = MWlike['gals']

norm = lambda f, x: f/np.trapz(f, x) if not np.all(f == 0) else f
p50 = lambda x: np.percentile(x, 50)
pQ = lambda x, Q: np.percentile(x, Q)





class MLE:
    def __init__(self, mock, el_init, Mdm=np.linspace(1,10,20), run_globalmin=True):
        self.mock = mock
        self.el = el_init
        self.nobs = self.mock['binned_Esample']
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

    def get_mwD(self, gal):
        return mwld[gal]
        
    def get_best(self, Mdm):
        with Pool(processes=os.cpu_count()) as pool:
            Sdm, Nll = [], []
            for result in pool.imap(self.min_sdm, Mdm):
                Sdm.append(result[0])
                Nll.append(result[1])
        return (np.array(Sdm), np.array(Nll))
        
    def min_sdm(self, mdm):
        minz = minimize(self.lsdm_func, x0=[-38.], args=(mdm))
        return (math.pow(10, minz.x[0]), minz.fun)

    def limit_fn(self, mdm):
        idx = np.where(self.Mdm == mdm)[0][0]
        if self.Tq[idx] > self.tq_limit:
            return (0, 0)
        def func(lsdm):
            nll = self.lsdm_func(lsdm, mdm)
            tq = 2*(nll - self.nll_min)
            return tq - self.tq_limit
        solupp = bisect(func, np.log10(self.Sdm[idx]), -34.0)
        sollow = bisect(func, -42.0, np.log10(self.Sdm[idx]))
        return (mdm, math.pow(10, solupp), math.pow(10, sollow))

    def get_limits(self, Mdm, tq_limit=5.99):
        if not self.ran_globalmin:
            print (f'Running self.globalmin() for the first time')
            self.globalmin(Mdm)
        self.tq_limit=tq_limit
        self.Mdm = Mdm
        Mdm_lim = []
        Sdm_upp = []
        Sdm_low = []
        with Pool(processes=os.cpu_count()) as pool:
            for result in pool.imap(self.limit_fn, Mdm):
                if 0 in result:
                    continue 
                else:
                    Mdm_lim.append(result[0])
                    Sdm_upp.append(result[1])
                    Sdm_low.append(result[2])
        return (np.array(Mdm_lim), np.array(Sdm_upp), np.array(Sdm_low))
    
    def get_Tgrid(self, Mdm=np.linspace(1,10,20), Sdm = np.logspace(-40., -37, 20)):
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
    
    def get_λsg0s(self, Mdm, datafile = '../../Output/El_rate_prediction.pkl'):
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
    def __init__(self, mock, el_init, Mdm=np.linspace(1,10,20), globalmin=True):
        super().__init__(mock, el_init, Mdm, globalmin)

    def filter_and_write(self, Mdm, write=True, datafile='../../Output/El_rate_prediction.pkl'):
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
        binT = self.el.binTot(mdm, 1e-40, 0.)
        return binT['Neachbin']
    
    def λsg0(self, mdm):
        mdm_keys = np.array(list(self.λsg0s.keys()))
        mdm_idx = np.argmin(np.abs(mdm_keys - mdm))
        return self.λsg0s[mdm_keys[mdm_idx]]
    
    def λsg(self, mdm, sdm):
        return self.λsg0(mdm)*sdm/1e-40
    
    def nlL(self, mdm, sdm, bl):
        λ = self.λsg(mdm, sdm) + self.mock['exposure']*self.mock['binwidth']*bl 
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
        λ = self.λsg(mdm, sdm) + self.mock['exposure']*self.mock['binwidth']*bl 
        return np.mean(self.nobs/λ) - 1.0





class MassModelUncertanity(MLE):
    def __init__(self, mock, el_init, Mdm=np.linspace(1,10,20), globalmin=True, chainlen='all', gal='MW', fixed_rhosun=True):
        self.gal = gal
        self.mwD = self.get_mwD(self.gal)
        self.fixed_rhosun = fixed_rhosun
        if isinstance(chainlen, str) and chainlen == 'all':
            self.chainlen = len(self.mwD['vdfEs'])
        elif isinstance(chainlen, int) and chainlen < 10000:
            self.chainlen = chainlen
        super().__init__(mock, el_init, Mdm, globalmin)

    def filter_and_write(self, Mdm, write=True, parallel = 8, datafile='../../Output/El_rate_prediction.pkl'):
        Mdm = np.around(Mdm, 2)
        λsg0d = pickle.load(open(datafile, 'rb'))
        λsg0s = λsg0d['massmodel'][self.gal].copy()
        chain_keys = list(λsg0s.keys())
        for ci in tqdm(range(self.chainlen)):
            if not ci in chain_keys:
                write = True
                λsg0s[ci] = {}
                if parallel:
                    parallel_λsg0s = self.parallel_compute(Mdm, ci, parallel) 
                    for mdm in Mdm:
                        λsg0s[ci][mdm] = parallel_λsg0s[mdm]
                else:
                    for mdm in Mdm:
                        λsg0s[ci][mdm] = self.compute_λsg0s(mdm, ci)
            else:
                write = False
                mdm_keys = np.array(list(λsg0s[ci].keys()))
                for mdm in Mdm:
                    if np.min(np.abs(mdm_keys - mdm)) > 0.01:
                        write = True
                        λsg0s[ci][mdm] = self.compute_λsg0s(mdm, ci)
            if write:
                λsg0d['massmodel'][self.gal] = λsg0s
                pickle.dump(λsg0d, open(datafile, 'wb'))
        return λsg0s
    
    def parallel_func(self, mdm):
        return self.compute_λsg0s(mdm, self.chain_idx_here)
    
    def parallel_compute(self, Mdm, chain_idx, num_cores):
        self.chain_idx_here = chain_idx
        with Pool(processes=num_cores) as pool:
            results = []
            for result in pool.imap(self.parallel_func, Mdm):
                results.append(result)
        res_dict = {Mdm[i]: results[i] for i in range(len(Mdm))}
        return res_dict
    
    def compute_λsg0s(self, mdm, chain_idx):
        vE = self.mwD['vE']
        vdfE = self.mwD['vdfEs'][chain_idx]
        vesc = self.mwD['vescs'][chain_idx]
        vcirc = self.mwD['vcircs'][chain_idx]
        if self.fixed_rhosun:
            rhosun = self.el.ρ0
        else:
            rhosun = self.mwD['rhosuns'][chain_idx]
        el_ = El(self.el.material, vE=vE, vdfE=vdfE, vesc=vesc, vcirc=vcirc, rhosun=rhosun)
        binT = el_.binTot(mdm, 1e-40, 0.)
        return binT['Neachbin']

    
    def λsg0(self, mdm, chain_idx):
        mdm_keys = np.array(list(self.λsg0s[chain_idx].keys()))
        mdm_idx = np.argmin(np.abs(mdm_keys - mdm))
        return self.λsg0s[chain_idx][mdm_keys[mdm_idx]]
    
    def λsg(self, mdm, sdm, chain_idx):
        return self.λsg0(mdm, chain_idx)*sdm/1e-40
    
    def nlL(self, mdm, sdm, bl, chain_idx):
        λ = self.λsg(mdm, sdm, chain_idx) + self.mock['exposure']*self.mock['binwidth']*bl 
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
        λ = self.λsg(mdm, sdm, chain_idx) + self.mock['exposure']*self.mock['binwidth']*bl 
        return np.mean(self.nobs/λ) - 1.0
    




class SampleUncertanity(MLE):
    def __init__(self, mock, el_init, Mdm=np.linspace(1, 10, 20), percentiles=np.arange(5, 96, 2), run_parallel=True, run_globalmin=True, fixed_rhosun=True):
        self.gals = mwgals
        self.percentiles = np.around(percentiles, 2)
        self.fixed_rhosun = fixed_rhosun
        self.run_parallel = run_parallel
        self.vE = mwld['MW']['vE']
        self.vdf_dict  = self.vdf_info()
        super().__init__(mock, el_init, Mdm, run_globalmin)

    def vdf_info(self):
        Vdfs, Vescs, Vcircs, Rhosuns = [], [], [], []
        for gal in mwgals:
            Vdfs.append(mwld[gal]['vdfs'])
            Vescs.append(mwld[gal]['vescs'])
            Vcircs.append(mwld[gal]['vcircs'])
            Rhosuns.append(mwld[gal]['rhosuns'])
        Vdfs = np.vstack(Vdfs).T
        Vescs = np.concatenate(Vescs)
        Vcircs = np.concatenate(Vcircs)
        Rhosuns = np.concatenate(Rhosuns)
        vdf_dict = {}
        for per in self.percentiles:
            vdf = np.percentile(Vdfs, per, axis=1)
            vesc = np.percentile(Vescs, per)
            vcirc = np.percentile(Vcircs, per)
            rhosun = np.percentile(Rhosuns, per)
            vdfE = get_vdf_ert(vE=mwd['vE'], v=mwd['v'], vdf=vdf, vesc=vesc, vcirc=vcirc)
            vdf_dict[per] = {'vdf': vdf, 'vdfE': vdfE, 'vesc':vesc, 'vcirc':vcirc, 'rhosun':rhosun}
        return vdf_dict

    def filter_and_write(self, Mdm, write=True, datafile='../../Output/rate_prediction.pkl'):
        Mdm = np.around(Mdm, 2)
        λsg0d = pickle.load(open(datafile, 'rb'))
        λsg0s = λsg0d['sample'].copy()
        percentiles = list(λsg0s.keys())
        for per in tqdm(self.percentiles):
            print (per)
            if not per in percentiles:
                print ('No per')
                write = True
                λsg0s[per] = {}
                if self.run_parallel:
                    λsg0s[per] = self.parallel_compute(Mdm, per)
                else:
                    for mdm in Mdm:
                        λsg0s[per][mdm] = self.compute_λsg0s(mdm, per)
            else:
                write = False
                mdm_keys = np.array(list(λsg0s[per].keys()))
                for mdm in Mdm:
                    if np.min(np.abs(mdm_keys - mdm)) > 0.01:
                        print ('no mdm')
                        write = True
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
        
    def get_el(self, percentile):
        vdfE = self.vdf_dict[percentile]['vdfE']
        vesc = self.vdf_dict[percentile]['vesc']
        vcirc = self.vdf_dict[percentile]['vcirc']
        if self.fixed_rhosun:
            rhosun = self.el.ρ0 
        else:
            rhosun = self.vdf_dict[percentile]['rhosun']
        return El(self.el.material, vE=self.vE, vdfE=vdfE, vesc=vesc, vcirc=vcirc, rhosun=rhosun)
    
    def compute_λsg0s(self, mdm, percentile):
        el_ = self.get_el(percentile=percentile)
        binT = el_.binTot(mdm, 1e-40, 0.)
        return binT['Neachbin']
    
    def λsg0(self, mdm, percentile):
        mdm_keys = np.array(list(self.λsg0s[percentile].keys()))
        mdm_idx = np.argmin(np.abs(mdm_keys - mdm))
        return self.λsg0s[percentile][mdm_keys[mdm_idx]]
    
    def λsg(self, mdm, sdm, percentile):
        return self.λsg0(mdm, percentile)*sdm/1e-40
    
    def nlL(self, mdm, sdm, bl, percentile):
        λ = self.λsg(mdm, sdm, percentile) + self.mock['exposure']*self.mock['binwidth']*bl 
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
        λ = self.λsg(mdm, sdm, percentile) + self.mock['exposure']*self.mock['binwidth']*bl 
        return np.mean(self.nobs/λ) - 1.0