import numpy as np
import scipy as sp
import sys,json,os
sys.path.append('../')
sys.path.append('../..')
sys.path.append('../../..')
import project.profiles as pp
from project.tools.constants import Constants as pc

import warnings
warnings.filterwarnings('ignore')

class EI_vdf:
    def __init__(self, model, **kwargs):
        self.dm = model['dm'] if 'dm' in model.keys() else False
        self.disk = model['disk'] if 'disk' in model.keys() else False
        self.bulge = model['bulge'] if 'bulge' in model.keys() else False
        self.gas = model['gas'] if 'gas' in model.keys() else False
        
        if self.dm:
            if self.dm[0] == 'NFW':
                self.lm,self.rs = self.dm[1:3]
            elif self.dm[0] == 'BNFW':
                self.lm,self.rs,self.rc = self.dm[1:4]
                self.rhob,self.rb = pp.rhob_rb_burk(self.lm,self.rs,self.rc)
            else:
                self.dm_func = self.dm[1]
                
        self.r_sat = kwargs.get('r_sat') if 'r_sat' in kwargs.keys() else 50
        self.r_infinity = 1e7
        self.R = np.logspace(-1,7,400)
        self.Mtot = self.mass_tot(self.R)
        self.Mtot_fun = sp.interpolate.interp1d(self.R, self.Mtot, kind = 'cubic', fill_value = 'extrapolate')
        # self.Ptot = np.array([-self.pote_tot(r) for r in self.R])
        self.Ptot = -self.pote_tot(self.R)
        self.Ptot_fun = sp.interpolate.interp1d(self.R, self.Ptot, kind = 'cubic', fill_value = 'extrapolate')
        # self.d2 = np.array([self.d2ρd2ϕ(r) for r in self.R])
        self.d2 = self.d2ρd2ϕ(self.R)
        self.d2ρd2ϕ_func = sp.interpolate.interp1d(self.R, self.d2, kind = 'cubic', fill_value = 'extrapolate')
        
    def density_dm(self,r):
        if self.dm:
            if self.dm[0] == 'NFW':
                return pp.density_nfw(self.lm,self.rs,r)
            elif self.dm[0] == 'BNFW':
                return pp.density_bnfw(self.lm,self.rs,self.rc,r)
            else:
                return self.dm_func(r)
        else:
            return 0
        
    def mass_dm(self,r):
        if self.dm:
            if self.dm[0] == 'NFW':
                return pp.mass_nfw(self.lm,self.rs,r)
            elif self.dm[0] == 'BNFW':
                return pp.mass_bnfw(self.lm,self.rs,self.rc,r)
            else: return 0
        else: return 0
    
    def mass_bary(self,r):
        if self.disk:
            if self.disk[0] == 'EXP':
                mass_disk = pp.mass_exp(self.disk[1],self.disk[2],r)
            elif self.disk[0] == 'POINT':
                mass_disk = self.disk[1]
            else: mass_disk == 0
        else: mass_disk = 0
        
        if self.bulge:
            if self.bulge[0] == 'EXP':
                mass_bulge = pp.mass_exp(self.bulge[1],self.bulge[2],r)
            else: mass_bulge = 0
        else: mass_bulge = 0
        
        if self.gas:
          if self.gas[0] == 'EXP':
            mass_gas = pp.mass_exp(self.gas[1],self.gas[2],r)
          else: mass_gas = 0
        else: mass_gas = 0
        return mass_disk + mass_bulge + mass_gas
    
    def mass_tot(self,r):
        return self.mass_bary(r) + self.mass_dm(r)
    
    def pote_dm(self,r):
        if self.dm:
            if self.dm[0] == 'NFW':
                return pp.potential_nfw(self.lm,self.rs,r)
            elif self.dm[0] == 'BNFW':
                return pp.potential_bnfw(self.lm,self.rs,self.rc,r,rhob_rb = [self.rhob,self.rb])
            else: return 0
        else: return 0
    
    def pote_bary(self,r):
        if self.disk:
            if self.disk[0] == 'EXP':
                pote_disk = pp.potential_exp(self.disk[1],self.disk[2],r)
            elif self.disk[0] == 'POINT':
                pote_disk = pp.potential_point(self.disk[1],r)
            else: pote_disk = 0
        else: pote_disk = 0
        
        if self.bulge:
            if self.bulge[0] == 'EXP':
                pote_bulge = pp.potential_exp(self.bulge[1],self.bulge[2],r)
            else: pote_bulge = 0
        else: pote_bulge = 0
        
        if self.gas:
          if self.gas[0] == 'EXP':
            pote_gas = pp.potential_exp(self.gas[1],self.gas[2],r)
          else: pote_gas = 0
        else: pote_gas = 0
        return pote_disk + pote_bulge
    
    def pote_tot(self,r):
        return self.pote_bary(r) + self.pote_dm(r)
    
    def Vmax(self,r):
        return np.sqrt(2*self.Ptot_fun(r))
    
    def d2ρd2ϕ(self,r):
        self.dρdr = lambda r_: sp.misc.derivative(self.density_dm, r_, dx = 1e-3, order = 3)
        self.dϕdr = lambda r_: -pc.G*self.Mtot_fun(r_)/r_**2
        self.dρdϕ = lambda r_: self.dρdr(r_)/self.dϕdr(r_)
        return -sp.misc.derivative(self.dρdϕ, r, dx = 1e-3, order = 3)

    def get_vdf(self, r, n = 32):
        self.n = n
        vesc = self.Vmax(r)
        v = np.linspace(0,vesc,self.n)
        pote_here = self.Ptot_fun(r)
        E = pote_here-0.5*v**2
        
        
        Rmax = np.ones(self.n)
        for i in range(0,self.n-1,1):
            def eqsn(x):
                return self.Ptot_fun(x) - E[i]
            Rmax[i] = float(sp.optimize.fsolve(eqsn,x0 = r)[0])
        
        I = np.zeros(self.n)
        for i in range(0,self.n-1,1):
            def integrand(x):
                x = np.array(x)
                return self.d2ρd2ϕ_func(x)/np.sqrt(np.abs(E[i]-self.Ptot_fun(x)))
            I[i] = sp.integrate.quad(integrand,Rmax[i],1000*Rmax[i],points = Rmax[i],limit = 10000)[0]
            
        # maximum value of ψ = 0 at r = infinity, minimum value is E (due to escape vel) at r = solve(ψ(r) - E = 0).
        
        # I = np.zeros(self.n)
        # for i in range(0,self.n-1,1):
        #   r_ = np.logspace(np.log10(Rmax[i]+1e-7),np.log10(Rmax[i])+3,10000)
        #   ψ = -self.Ptot_fun(r_)
        #   d2ρd2ψ = self.d2ρd2ϕ_func(r_)
        #   integrand = d2ρd2ψ/np.sqrt(np.abs(E[i]-ψ))
        #   I[i] = np.trapz(integrand,r_)/(np.sqrt(8)*np.pi**2)
        
            
      
            
        vdist = I*(1/(np.sqrt(8)*np.pi**2))/self.density_dm(r) # velocity distribution
        sdist = 4.*np.pi*v**2*vdist # speed distribution
        vdf = np.vstack([v,sdist])
        return vdf.transpose()