#!/usr/bin/env python
# coding: utf-8

"""
@author: einfaltleonie
modified to include iminuit by: manushmanju
"""


import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
import operator
import warnings
warnings.filterwarnings('ignore') # to ignore the iminuit warning
                                  # (E VariableMetricBuilder Initial matrix not pos.def.)


# In[109]:


class HoppingResult(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return '\n'.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in sorted(self.items())])
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys())
    


# In[159]:


def zerothstep(m, paraNames, guess0):
    foundmin = False
    count = 0
    output = {}
    message = ''
    x0 = 0
    fun0 = 0
    while foundmin == False:
        i = 0
        for para in paraNames:
            m.values[para] = guess0[i]
            i += 1
        m.migrad()
        fun0 = m.fval
        x0 = []
        for para in paraNames:
           x0.append(m.values[para])
        x0 = np.array(x0)	
        if m.valid or all(x0 != guess0) or np.isfinite(x0).all():
            foundmin = True
            break
        else:
            message = 'Can not evaluate at zeroth guess'
        guess0 = np.zeros(1)
        for para in paraNames:
            ng = np.random.uniform(*m.limits[para])
            guess0 = np.concatenate((guess0,[ng]))
        guess0 = guess0[1:]
        count += 1
        if count > 30:
            message = 'Cannot find good zeroth guess, enter manually'
            break
    output = {'guess0':guess0, 'fun0':fun0, 'x0':x0, 'm':m}
    errorhandler = {'error': operator.not_(foundmin), 'message': message}
    return errorhandler, output

def firstquadstep(m, paraNames, fun0):
    stepsuccess = False
    count = 0
    while stepsuccess == False:
        nextguess = np.zeros(1)
        for para in paraNames:
            ng = np.random.uniform(*m.limits[para])
            nextguess = np.concatenate((nextguess,[ng]))
        nextguess = nextguess[1:]
        xguess = m.fcn(m.values[:])
        if xguess < fun0 and np.isfinite(xguess) == True:
            stepsuccess = True
            break
        else:
            count += 1
        if count == 30:
            break
    return nextguess

def xiknown(m, x, fun, count, paraNames):
    res = []
    for para in paraNames:
	    res.append(m.values[para])
    res = np.array(res)
    xi = np.array([res])
    prev = np.where(np.all(abs(x-xi)<0.1, axis = 1))[0]
    if prev.size:
        count[prev[0]] += 1
    else:
        x = np.concatenate((x, xi), axis = 0)
        fun.append(m.fval)
        count.append(1)
    return x, fun, count
        
        
def firstquadsearch(m, paraNames, x0, fun0, failiter, multicount, terminationflag, precisionflag):
    count = []
    fun = []
    fun.append(fun0), count.append(1)
    x = np.array([x0])
    count_fail = 0

    while count_fail < failiter:
        guess = firstquadstep(m, paraNames, fun0)
        i = 0
        for para in paraNames:
            m.values[para] = guess[i]
            i += 1
        # This can be changed to just, m.values[paraNames] = guess.
        # Apply this simplification everywhere else also.
        m.migrad()
        if m.valid:
            x, fun, count = xiknown(m, x, fun, count, paraNames)
        elif m.valid == False and all(m.values[paraNames] != guess) and np.isfinite(m.values[paraNames]).all() == True:
            x, fun, count = xiknown(m, x, fun, count, paraNames)
            precisionflag = True
        else:
            count_fail += 1
        if max(count) >= multicount:
            break
        if count_fail > failiter:
            terminationflag = True
            break
    return {'x':x, 'fun':fun, 'count':count,
           'termination': terminationflag, 'precision':precisionflag,
           'nfail':count_fail}


# In[128]:


def quadhop(m, paraNames, guess0):
    """
    m has to be given after defining limits and errordef and proper
    fixed parameters. 
    paraNames only contains the parameter names which are not fixed 
    and in order of guess0.
    """
    guess0 = np.array(guess0)
    foundmin = False
    failiter = 5
    multicount = 4
    success = False
    x = 0
    fun = 0
    globalmin = x
    globalfun = x
    count = 0
    precision = 'Not applicable.'
    
    #error flags
    error = False
    precisionflag = False
    terminationflag = False
    
    errorhandler, stepzero = zerothstep(m, paraNames, guess0)
    error = errorhandler.get('error')
    if error == False:
        x0 = stepzero.get('x0')
        fun0 = stepzero.get('fun0')
        m = stepzero.get('m')
    
        quad = firstquadsearch(m, paraNames, x0, fun0, failiter, multicount, terminationflag, precisionflag)
        precisionflag = quad['precision']
        terminationflag = quad['termination']
        x = quad['x']
        count = quad['count']
        fun = quad['fun']
        globalfun = min(fun)
        globalmin = x[fun.index(globalfun),:]

        if terminationflag:
            success = False
            message = 'Terminatin due to too many failed minimization routines.'
        else:
            success = True
            message = 'At least one minima was found a sufficient number of times.'
        if precisionflag:
            precision = 'At some point a termination due to precision loss occured.'
        else:
            precision = 'No precision loss in any of the minimizations.'
    else:
        message = errorhandler.get('message')
    result = HoppingResult(success = success, precision = precision, message = message,
             globalfun = globalfun, fun = fun, minima = x, globalmin = globalmin,
             count = count)    
    return result,m
    

