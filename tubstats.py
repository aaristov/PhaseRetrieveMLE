### tubulin processing tools

import pylab as plb
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp

def linearFit(x,y):
    '''
    Fits 2D linear futction to x and y
    Returns linear_fit(x)
    '''
    linCurve = lambda x,a,b: a*x+b
    popt,pcov = curve_fit(linCurve,x,y)
    curve = linCurve(np.unique(x),*popt)
    #plt.plot(np.unique(x),curve)
    #plt.show()
    plt.plot(x,y,'.')
    plt.plot(np.unique(x),curve,'r-',label='fit')
    return linCurve(x,*popt)

def gaussFitHist(h,num_peaks=1):
    '''
    Fitting gaussian function to the histogram
    input: np.histogram() result,
            number of peaks (1 or 2)
    output: dictionary of parameters for 1 or 2 peaks
    '''

    x = np.array(h[1][1:])-(h[1][1]-h[1][0])/2
    y = h[0]

    n = len(x)                          #the number of data
    mean = x[np.argmax(y)]#sum(x*y)/n                   #note this correction
    sigma = 50#sum(y*(x-mean)**2)/n        #note this correction
    print mean,sigma
    def gaus(x,a,x0,sigma):
        return a*exp(-(x-x0)**2/(2*sigma**2))

    def gaus2(x,a1,a2,x1,x2,sigma):
        return a1*exp(-(x-x1)**2/(2*sigma**2)) + a2*exp(-(x-x2)**2/(2*sigma**2))

    popt,pcov = curve_fit(gaus,x,y,p0=[25,mean,sigma])

    #plt.hist(x,y)
    gg=gaus(x,*popt)
    plt.plot(x,gg,'ro:',label='fit')
    out = dict(a=popt[0],
                x0=popt[1],
                sigma=popt[2])
    if num_peaks==2:
        y=y-gg
        mean = x[np.argmax(y)]#sum(x*y)/n                   #note this correction
        sigma = 50#sum(y*(x-mean)**2)/n        #note this correction
        print mean,sigma
        popt,pcov = curve_fit(gaus,x,y,p0=[25,mean,sigma])
        out['a2']=popt[0]
        out['x02']=popt[1]
        out['sigma2']=popt[2]

        gg=gaus(x,*popt)
        plt.plot(x,gg,'go:')

    return out


def doubleGaussFitHist(curve,num_peaks=2):
    '''
    Fitting gaussian function to the histogram
    input: curve, 2D array
    output: dictionary of parameters for 2 peaks
    '''
    def gaus(x,a,x0,sigma):
        return a*exp(-(x-x0)**2/(2*sigma**2))


    def gaus2(x,a1,a2,x1,x2,sigma):
        return a1*exp(-(x-x1)**2/(2*sigma**2)) + a2*exp(-(x-x2)**2/(2*sigma**2))

    x = curve[:,0]
    xstep = x[1]-x[0]
    xnew = np.arange(x[0],x[-1],xstep/10)
    y = curve[:,1]

    n = len(x)                          #the number of data
    mean1 = x[np.argmax(y)]#sum(x*y)/n                   #note this correction
    sigma = .01#sum(y*(x-mean)**2)/n        #note this correction
    a1 = .8*y.max()
    print 'x1,a1,sigma',mean1,a1,sigma

    peak1gaus = gaus(x,a1,mean1,sigma)
    y2 = y-peak1gaus
    mean2 = x[np.argmax(y2)]
    a2 = .8*y2.max()


    peak2gaus = gaus(x,a2,mean2,sigma)

    plt.plot(x,y,'k-')
    #plt.plot(x,peak1gaus)
    #plt.plot(x,peak2gaus)
    #plt.plot(x,peak1gaus+peak2gaus)
    #plt.plot(x,y2)


    popt,pcov = curve_fit(gaus2,x,y,p0=[a1,a2,mean1,mean2,sigma])

    #plt.hist(x,y)
    gg=gaus2(xnew,*popt)
    g1 = gaus(xnew,popt[0],popt[2],popt[4])
    g2 = gaus(xnew,popt[1],popt[3],popt[4])
    plt.plot(xnew,gg,'r-',label='fit')
    plt.plot(xnew,g1,'b-',label='fit1')
    plt.plot(xnew,g2,'b-',label='fit2')
    plt.title('inter-peak distance: {} nm'.format(np.round(np.abs(popt[2]-popt[3])*1000,1)))

    out = popt


    return out
