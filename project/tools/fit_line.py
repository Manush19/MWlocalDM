import numpy as np
import scipy as sp
import ultranest
import ultranest.stepsampler as ultrastep
from getdist import MCSamples, plots
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'dejavuserif'

purple = (0.25098039215686274, 0.0, 0.29411764705882354, 1.0)
green = (0.0, 0.26666666666666666, 0.10588235294117647, 1.0)



class Fitline:
    def __init__(self, x, y, **kwargs):
        self.x = x
        self.y = y
        key1,key2 = '',''
        if 'xerr' in kwargs.keys():
            key1 = 'x'
            self.xerr = kwargs.get('xerr')
        if 'yerr' in kwargs.keys():
            key2 = 'y'
            self.yerr = kwargs.get('yerr')
        self.key = key1 + key2
        self.paramnames = ['m','c']
        if 'prior' in kwargs.keys():
            self.prior = kwargs.get('prior')
        else:
            self.prior = 'flat'
        if 'm_prior' in kwargs.keys():
            self.m_prior = kwargs.get('m_prior')
        else:
            self.m_prior = [1,0]
        if 'c_prior' in kwargs.keys():
            self.c_prior = kwargs.get('c_prior')
        else:
            self.c_prior = [-1,1]
    
    def getdist_analyze(self,sample):
        getsample = MCSamples(samples = sample.results['samples'],
                              labels = self.paramnames,
                              names = self.paramnames,
                              settings = dict(smooth_scale_2D = 2),
                              sampler = 'nested')
        Means = getsample.means.tolist()
        Sddev = getsample.sddev.tolist()
        return (getsample, Means, Sddev)
    
    def scatter_dex(self,means):
        m,c = means
        line = m*self.x + c
        scatter = np.sqrt(((self.y - line)**2).sum())
        scatter = scatter/np.sqrt(len(self.x) - 1)
        return scatter
    
    def get_llike_prior_fn(self):
        def prior_flat(cube):
            params = cube.copy()
            params[0] = cube[0]*(self.m_prior[1] - self.m_prior[0]) + self.m_prior[0]
            params[1] = cube[1]*(self.c_prior[1] - self.c_prior[0]) + self.c_prior[0]
            return params

        def prior_gauss(cube):
            params = cube.copy()
            params[0] = sp.stats.norm.ppf(cube[0],self.m_prior[0],self.m_prior[1])
            params[1] = sp.stats.norm.ppf(cube[1],self.c_prior[0],self.c_prior[1])
            return params

        def llike_(params):
            m,c = params
            line = m*self.x + c
            llike = -0.5*((self.y - line)**2).sum()
            return llike

        def llike_y(params):
            m,c = params
            line = m*self.x + c
            llike = -0.5*(((self.y - line)/self.yerr)**2).sum()
            return llike

        def llike_xy(params):
            m,c = params
            line = m*self.x + c
            llike = -0.5*(((self.y - line)/(self.yerr + m*self.xerr))**2).sum()
            return llike
        
        if self.key == '':
            llike = llike_
        elif self.key == 'y':
            llike = llike_y
        elif self.key == 'xy':
            llike = llike_xy
        if self.prior == 'flat':
            prior_fn = prior_flat
        elif self.prior == 'gauss':
            prior_fn = prior_gauss
        return llike,prior_fn
        
    
    def fit(self,ultraroot = '../Output/Ultra/line'):
        llike,prior_fn = self.get_llike_prior_fn()
        sampler = ultranest.ReactiveNestedSampler(self.paramnames,llike, prior_fn, log_dir = ultraroot, 
                                                  resume = 'overwrite')
        result = sampler.run(show_status = False)
        
        getsample,means,sddevs = self.getdist_analyze(sampler)
        
        print ('slope = %.2f +- %.2f'%(means[0],sddevs[0]))
        print ('intcp = %.2f +- %.2f'%(means[1],sddevs[1]))
        sig_obs = self.scatter_dex(means)
        print ('sig_obs = %.2f dex'%sig_obs)
        
        self.getsample = getsample
        self.means = means
        self.sddevs = sddevs
        self.sig_obs = sig_obs
        self.func = lambda x_: means[0]*x_ + means[1]
        return (means, sddevs, sig_obs)
    
    def getdict_contour(self,contour_file = None, clr = purple):
        g = plots.get_subplot_plotter()
        g.settings.figure_legend_frame = False
        g.settings.title_limit_fontsize = 13
        g.settings.alpha_filled_add = 0.4
        g.triangle_plot(self.getsample,
                        filled = True,
                        legend_loc = 'upper right',
                        line_args = {'ls':'-','lw':2,'color':clr},
                        contour_colors = [clr],
                        title_limit = 1,
                        markers = self.getsample.means)
        if not contour_file:
            # now = datetime.now()
            # contour_file = now.strftime("../Output/Ultra/line/contours/%d_%m_%Y_%H_%M_%S.png")
            plt.show()
        else:
            plt.savefig(contour_file, bbox_inches = 'tight', dpi = 300)
        print ('triangle (contour) plot saved in %s'%contour_file)

    def plot_line(self, xlabel, ylabel, xscale = 'log', yscale = 'log',
                  clr = purple, width = 6, height = 4,marker = 'o', s = 100, 
                  alpha = 0.4, xlim = None, ylim = None, plot_Exy = True,
                  xy_names = ['logx','logy'], filename = None,ax = None,
                  loc = 'upper left',errorbars = None):
        if not ax:
            fig,ax = plt.subplots(figsize = (width,height))
        if plot_Exy:
            ax.scatter(10**self.x,10**self.y,marker = marker, s = s, alpha = alpha, 
                       edgecolor = 'k',color = clr)
            if errorbars:
                if self.key == 'xy':
                    ax.errorbar(10**self.x,10**self.y,xerr = errorbars[0],
                                yerr = errorbars[1], fmt = 'none', ecolor = 'k', capsize = 2,
                                alpha = alpha)
                elif self.key == 'y':
                    ax.errorbar(10**self.x,10**self.y,yerr = errorbars[0],
                                fmt = '', ecolor = 'k', capsize = 2,alpha = alpha)
                else:
                    print ('Fitline without errorbars')              
        else:
            ax.scatter(self.x,self.y,marker = marker, s = s, alpha = alpha, 
                       edgecolor = 'k',color = clr)
            if errorbars:
                if self.key == 'xy':
                    ax.errorbar(self.x,self.y,xerr = errorbars[0],yerr = errorbars[1],
                                fmt = '', ecolor = 'k', capsize = 2,alpha = alpha)
                elif self.key == 'y':
                    ax.errorbar(self.x,self.y,yerr = errorbars[0],
                                fmt = '', ecolor = 'k', capsize = 2,alpha = alpha)
                else:
                    print ('Fitline without errorbars')
        if not xlim:
            xlim = ax.get_xlim()
            print (xlim)
            if plot_Exy:
                xlim = np.log10(xlim)
        if not ylim:
            ylim = ax.get_ylim()
            if plot_Exy:
                ylim = np.log10(ylim)
        x4line = np.linspace(xlim[0],xlim[1],100)
        y4line = self.means[0]*x4line + self.means[1]
        label = '%s = %.2f%s + %.2f'%(xy_names[1],self.means[0],
                                      xy_names[0],self.means[1])
        # ax.text(0.8,0.1,'$\sigma_{\mathrm{obs}} = %.2f$ dex'%self.sig_obs, transform = ax.transAxes)
        if plot_Exy:
            ax.plot(10**x4line,10**y4line,c = 'k',label = label)
            ax.fill_between(10**x4line,10**(y4line+self.sig_obs),10**(y4line-self.sig_obs), 
                            color = 'grey', alpha = 0.2)
        else:
            ax.plot(x4line,y4line,c = 'k',label = label)
            ax.fill_between(x4line,y4line + self.sig_obs,y4line - self.sig_obs, color = 'grey', 
                            alpha = 0.2)
        ax.set_xlabel(xlabel,size = 14)
        ax.set_ylabel(ylabel,size = 14)
        ax.set_xlim(10**xlim[0],10**xlim[1])
        ax.set_ylim(10**ylim[0],10**ylim[1])
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.legend(prop = {'size':10},loc = loc)
        return ax
        
            
        
        
        
        
        
        