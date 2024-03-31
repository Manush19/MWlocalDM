import numpy as np
import scipy as sp
import ultranest
import ultranest.stepsampler as ultrastep

class FitVDF:
  def __init__(self, func,v,vdf,err_vdf,para,prior):
    self.v = v
    self.vdf = vdf
    self.err_vdf = err_vdf
    self.func = func
    self.para = para
    self.N = len(self.para)
    self.prior = prior #only flat priors are allowed here
    
  def get_llike_prior_fn(self):
    def prior_flat(cube):
      params = cube.copy()
      for i in range(self.N):
        p = self.prior[i]
        params[i] = cube[0]*(p[1] - p[0]) + p[0]
      return params
    
    def llike(params):
      f = self.func(self.v,*params)
      llike = -0.5*(((self.vdf - f)/self.err_vdf)**2).sum()
      if np.isnan(llike):
        llike = 1e38
      return llike
    
    return llike,prior_flat
  
  def fit(self,ultraroot = '../Output/Ultra/'):
    llike,prior_fn = self.get_llike_prior_fn()
    sampler = ultranest.ReactiveNestedSampler(self.para,llike,prior_fn,
                                              log_dir = ultraroot, 
                                              resume = 'overwrite')
    nsteps = 2*self.N
    sampler.stepsampler = ultrastep.RegionSliceSampler(nsteps = nsteps)
    result = sampler.run(show_status = True)
    self.means = result['posterior']['mean']
    self.stdev = result['posterior']['stdev']
    return result
  
  def print_results(self):
    for i in range(self.N):
      print ('%s := %.3f +- %.3f'%(self.para[i],self.means[i],self.stdev[i]))
    