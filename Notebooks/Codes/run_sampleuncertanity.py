import numpy as np
import sys, pickle 
sys.path.append('../../')

from project.recoil import Electron as El, p50, norm, get_vdf_ert 
from project.electron_pll import SampleUncertanity as Samp

MWlike = pickle.load(open('../../Output/MWlike_dict.pkl','rb'))
mwld = MWlike['vdf_RCfit']
mwd = mwld['MW']
mwgals = MWlike['mwgals']


el_init = El('Si', vE=mwd['vE'], vdfE=mwd['vdfE_50'], vesc=p50(mwd['vescs']), vcirc=p50(mwd['vcircs']), rhosun=p50(mwd['rhosuns']))
mdm0, sdm0, bl0 = 5., 1e-38, 1.
mock = el_init.mocksample(mdm0, sdm0, bl0, 3.8, exposure=1, seed=5222)

samp = Samp(mock, el_init=el_init, Mdm=np.linspace(2,6,100))