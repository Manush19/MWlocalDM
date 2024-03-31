import numpy as np
import scipy as sp
import scipy.integrate as simps
from project.constants import Constants as c
from scipy import special

def arctan(x):
  single = False
  if isinstance(x, (float,int)): x,single = [x],True
  x = np.array(x)
  indx1 = np.nonzero(x > 1e7)
  indx2 = np.nonzero(x < -1e7)
  x[indx1] = np.pi/2
  x[indx2] = -np.pi/2
  a = np.arctan(x)
  if single: return a[0]
  else: return a
    

### NFW

def r200_nfw(lm):
  return (3*(10**lm)/(4*c.pi*200*c.rho_crit))**(1./3)
  
def c200_nfw(lm,rs):
  return r200_nfw(lm)/rs

def rs_nfw(lm,c2):
  m200 = 10**lm
  return (3.*m200/(200.*4.*np.pi*c.rho_crit))**(1./3.)/c2

def rho0_nfw(lm,rs):
  c200 = c200_nfw(lm,rs)
  g = np.log(1. + c200) - (c200/(1 + c200))
  return (10**lm)/(4*c.pi*(rs**3)*g)

def mass_nfw(lm,rs,r):
  rho0 = rho0_nfw(lm,rs)
  gr = np.log((rs+r)/rs) - (r/(rs+r))
  return 4*c.pi*rho0*gr*(rs**3)

def v_nfw(lm,rs,r):
  mr = mass_nfw(lm,rs,r)
  return np.sqrt(c.G*mr/r)

def density_nfw(lm,rs,r):
  rho0 = rho0_nfw(lm,rs)
  x = r/rs
  return rho0/(x*((1+x)**2))

def potential_nfw(lm,rs,r):
  rho0 = rho0_nfw(lm,rs)
  return -4*c.pi*c.G*rho0*(rs**3)*np.log(1 + (r/rs))/r

### Burkert
                   
def rhob_rb_burk(lm,rs,rc):
  if rc == np.inf or np.isnan(rc) or rc <= 0:
    return np.inf,0
  y = rc/rs
  gg = 4*(np.log(1+y)-(y/(1+y)))*((1+y)**2/y**2)
  rho0 = rho0_nfw(lm,rs)
  def func(x):
      ff = ( np.log((1 + x**2)*(1 + x)**2)  - 2*arctan(x) )*(1+x)*(1 + x**2)/x**3
      return ff - gg
  x = sp.optimize.fsolve(func,x0 = 0.1)[0]
  rb = rc/x
  rhob = rho0*((1+x)*(1+x**2))/(y*(1+y)**2)
  return rhob,rb

def mass_burk(rhob,rb,r):
  kk = np.pi*rhob*rb**3*1e-10
  x = r/rb
  return kk * (np.log((1+x**2)*(1+x)**2) - 2*arctan(x)) * 1e10

def potential_burk(rhob,rb,r,N = 400):
  y,ymax = r/rb,1e7/rb
  I1 = lambda y: np.log(y) - np.log(1+y) - np.log(1+y)/y
  I2 = lambda y: -np.log(1+y**2)/(2.*y) + arctan(y)
  I3 = lambda y: -arctan(y)/y + np.log(y) - np.log(y**2 + 1)/2.
  I  = lambda y: I1(y) + I2(y) - I3(y)
  return 2.*np.pi*rhob*(rb**2)*c.G*(I(y)-I(ymax))

def density_burk(rhob,rb,r):
  y = r/rb
  return rhob/((1+y)*(1+y**2))

def v_burk(rhob,rb,r):
  return np.sqrt(mass_burk(rhob,rb,r)*c.G/r)


### BNFW

def mass_bnfw(lm,rs,rc,r,**kwargs):
  if 'rhob_rb' in kwargs.keys(): rhob,rb = kwargs.get('rhob_rb')  
  else: rhob,rb = rhob_rb_burk(lm,rs,rc)
  single = False
  if isinstance(r, (float,int)): r,single = [r],True
  r = np.array(r)
  m1 = mass_burk(rhob,rb,r[r<rc])
  m2 = mass_nfw(lm,rs,r[r >= rc])
  mass = np.zeros(r.shape)
  mass[r<rc] = m1
  mass[r>=rc] = m2
  if single: return mass[0]
  else: return mass

def v_bnfw(lm,rs,rc,r):
  return np.sqrt(mass_bnfw(lm,rs,rc,r)*c.G/r)

def density_bnfw(lm,rs,rc,r):
  rhob,rb = rhob_rb_burk(lm,rs,rc)
  single = False
  if isinstance(r, (float,int)): r,single = [r],True
  r = np.array(r)
  indx1 = np.nonzero(r[r <= rc])
  indx2 = np.nonzero(r[r > rc])
  d1 = density_burk(rhob,rb,r[indx1])
  d2 = density_nfw(lm,rs,r[indx2])
  
  dens = np.zeros(r.shape)
  dens[indx1] = d1
  dens[indx2] = d2
  if single: return dens[0]
  else: return dens

def potential_bnfw(lm,rs,rc,r,**kwargs):
  if 'rhob_rb' in kwargs.keys(): rhob,rb = kwargs.get('rhob_rb')
  else: rhob,rb = rhob_rb_burk(lm,rs,rc)
  try:
    r = np.concatenate([r,[1e7]])
    single = False
  except:
    r,single = np.concatenate([[r],[1e7]]),True
  pote = np.zeros(r.shape)
  pote[r<rc] = potential_burk(rhob,rb,r[r<rc])
  pote[r>=rc] = potential_burk(rhob,rb,rc) + potential_nfw(lm,rs,r[r>=rc]) - potential_nfw(lm,rs,rc)
  pote = pote[:-1] - pote[-1]
  if single: return pote[-1]
  else: return pote

### Photomeric

def v_disk(vg,vd,yd,r):
  return np.sqrt(vg*np.abs(vg)+yd*vd**2)

def v_bulge(vb,yb,r):
  return np.sqrt(yb*vb**2)

def v_bary(vg,vd,vb,yd,yb,r):
  return np.sqrt(np.abs(vg)*vg + yd*vd**2 + yb*vb**2)

def mass_bary(vg,vd,vb,yd,yb,r):
  v = v_bary(vg,vd,vb,yd,yb,r)
  M = v**2*r
  mbary = [M[0]]
  for i in range(len(M)-1):
    mbary.append(np.abs(M[i+1]-M[i]))
  mbary = np.array(mbary)
  return np.sum(mbary)/c.G

def mass_star(vd,vb,yd,yb,r):
  vg_ = np.zeros(vd.shape)
  return mass_bary(vg_,vd,vb,yd,yb,r)

### Exponential disk

def sig_exp(lmstar,rd):
  return 10**lmstar/(2.*np.pi*rd**2)

def mass_exp(lmstar,rd,r):
  return 10**lmstar * (1 - (np.exp(-r/rd)*(1 + (r/rd))))

def v_exp(lmstar,rd,r):
  mstar = 10**lmstar
  y = r/(2*rd)
  try:
    vsq = 2.*c.G*(mstar/rd)*(y**2)* (special.i0(y)*special.k0(y) - special.i1(y)*special.k1(y))
  except:
    print ('exp disk failure; lmstar = %.3f, rd = %.3f, special1 = %.3f'%(lmstar,
           rd,special.i0(y)*special.k0(y)))
    vsq = 0
  return np.sqrt(vsq)
  
def sig_profile_exp(lmstar,rdstar,r):
	return sig_exp(lmstar,rdstar)*np.exp(-r/rdstar)
	
	
def density_exp(lmstar,rdstar,r):
	return sig_profile_exp(lmstar,rdstar,r)/(2*r)

def potential_exp(lmstar,rd,r):
  pote = lambda r_: 10**lmstar * ((-1/r_) + np.exp(-r_/rd)/r_) * c.G
  result = pote(r) - pote(1e7)
  return result

def potential_point(lmstar,r):
  pote = lambda r_: -c.G * (10**lmstar) / r_
  result = pote(r) - pote(1e7)
  return result

### Scaling relations
def lc200_SR(lm,fb = c.fb, h = c.h):
  m200 = 10**(lm - 12.)
  m200 = m200/(1. - fb)
  return 0.905 - 0.101*(np.log10(m200*h))

def lm200_SR(lc,fb = c.fb, h = c.h):
  return 0.905 - lc - np.log10(h) + np.log10(1-fb) + 12

def lmstar_moster_13(logm200,fb = c.fb):
	m200 = 10**logm200/(1. - fb)
	N10 = 0.0351
	M10 = 11.590
	B10 = 1.376
	C10 = 0.608
	M1 = 10**(M10 - 10.)
	M = m200*1e-10
	mm = M/M1
	ratio = 2.*N10/(mm**(-1.*B10) + mm**C10)
	return np.log10(ratio*m200)

def f_behroozi(x,alp,delta,gam):
    fx = -np.log10(10**(alp*x) + 1.) + delta*(((np.log10(1. + np.exp(x)))**gam)/(1 + np.exp(10**(-x))))
    return fx
                   
def lmstar_behroozi_13(logm200,fb = c.fb):
    eps = 10**-1.777
    M1 = 10**11.514
    alp_behroozi = -1.412
    delta_behroozi = 3.508
    gam_behroozi = 0.316
    M200 = 10**logm200/(1. - fb)
    x = logm200 - 11.514
    logmstar = np.log10(eps*M1) + f_behroozi(x,alp_behroozi,delta_behroozi,gam_behroozi) - f_behroozi(0.,alp_behroozi,delta_behroozi,gam_behroozi)
    return logmstar
  
def lmstar_behroozi_19(logm200,fb = c.fb):
    logm200 = logm200 - np.log10(1.-fb)
    eps = -1.435
    alpha = 1.963
    beta = 0.482
    gamma = 10**-1.034
    delta = 0.411
    logM1 = 12.035
    x = logm200 - logM1
    logmstar = eps - np.log10(10**(-alpha*x) + 10**(-beta*x)) + gamma*(np.exp(-0.5*(x/delta)**2)) + logM1
    return logmstar
  
  
  

  
  
  
#####


# def frac(a,b):
# 	try:
# 		if a == np.inf or a == -np.inf:
# 			return a
# 		elif np.isnan(a):
# 			return np.inf
# 		elif b == 0:
# 			return (a/np.abs(a))*np.inf
# 		else:
# 			return a/b
# 	except:
# 		return a/b

# def Falsepos(f,a,b,max_iter,args = [],eps=1e-5,flag = ''):

# 	i = 0
# 	success = False
# 	while (i <= max_iter):
# 		c = frac(a*f(b,*args) - b*f(a,*args),f(b,*args)-f(a,*args))
# 		fc = f(c,*args)

# 		if i != 0:
# 			if np.abs(fc-fd) <= eps:
# 				if np.abs(fc) <= eps:
# 					success = True
# 					break
		
# 		if fc == 0.:
# 			break
# 		elif fc * f(a,*args) < 0:
# 			b = c
# 		else:
# 			a = c
# 		fd = fc	
# 		i += 1
# 	return {'iterations':i,'root':c,'success':success,'flag':flag}
  
  
# def rhob_rb_burk(logm200,rs,r1):
# 	if r1 == np.inf or np.isnan(r1) or r1 <= 0:
# 		return np.inf,0
# 	y = frac(r1,rs)
# 	gg = 4.*(np.log(1. + y) - frac(y,1.+y))*frac(po(1. + y,2),po(y,2))
# 	rho0 = rho0_nfw(logm200,rs)
# 	def func(x):
# 		if x == np.inf or np.isnan(x) or x <= 0 or x == -np.inf:
# 			return 1
# 		else:
# 			ff = ( ln((1. + po(x,2))*po(1. + x,2)) - 2.*arctan(x))*(1. + x)*(1. + po(x,2))/po(x,3)
# 			return ff - gg
# 	X = Falsepos(func,1.,10.,70)
# 	if X['success'] == True:
# 		X = X['root']
# 	else:
# 		X = np.inf
# 	rb = r1/X
# 	rhob = rho0*frac((1. + X)*(1. + po(X,2)),y*po(1. + y,2))
# 	return rhob,rb
  
# def chkary(a):
# 	if not isinstance(a,np.ndarray):
# 		if isinstance(a,list):
# 			return np.array(a)
# 		else:
# 			return np.array([a])
# 	else:
# 		return a

# def log(x):
# 	x = chkary(x)
# 	if not np.all([x <= 0]) or np.isnan(x):
# 		if len(x) == 1:
# 			return np.log10(x)[0]
# 		else:
# 			return np.log10(x)
# 	else:
# 		return np.ones(len(x))*-np.inf	


# def ln(x):
# 	x = chkary(x)
# 	if not np.all([x <= 0]) or np.all(np.isnan(x)):
# 		if len(x) == 1:
# 			return np.log(x)[0]
# 		else:
# 			return np.log(x)
# 	else:
# 		return np.ones(len(x))*-np.inf
		

# def E(x):
# 	return 10**x

# def po(a,b):
# 	return a**b
