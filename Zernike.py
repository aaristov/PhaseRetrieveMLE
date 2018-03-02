import numpy as np
import _pickle as cPickle
from skimage import io
from PhaseRetrieveTools import *
from multiprocessing import Pool, cpu_count, Process, Queue
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.DEBUG)
from scipy.io import loadmat
import math
import time


class PupilMask:
    ''' Defines pupil parameters and parabola based on the size of image and optical params'''

    def __init__(self,
                 img_size:int=128,
                 numerical_aperture:float=1.49,
                 immersion_medium_refractive_index:float=1.515,
                 pixel_size_um:float=.11,  # um
                 wavelength:float=.64,  # um
                 pupilReductionInSize:float=1,  # pupil reduction of size
                 parabolaMultiplier:float=1.,
                 **kwargs):
        self.img_size = img_size
        self.NA = numerical_aperture
        self.oil_n = immersion_medium_refractive_index
        self.px_size = pixel_size_um
        self.qy, self.qx = np.indices((img_size, img_size), dtype='f')
        qy, qx = self.qy, self.qx

        pupil_px_size = 1 / (pixel_size_um * img_size)
        self.pupil_pxs = numerical_aperture / (wavelength * pupil_px_size)
        center = img_size / 2
        self.center = center
        kx = 2 * np.pi * (qx - center) / (pixel_size_um * img_size)
        ky = 2 * np.pi * (qy - center) / (pixel_size_um * img_size)

        self.pupilmask = np.ones_like(qx) * (
                    kx ** 2 + ky ** 2 <= (2 * np.pi * numerical_aperture / wavelength * pupilReductionInSize) ** 2)

        self.xslope, self.yslope = qx / (img_size) * 2 * np.pi / pixel_size_um, qy / (img_size) * 2 * np.pi / pixel_size_um
        # with np.errstate(sqrt ='ignore'):
        parabola0 = np.sqrt(
            (2 * np.pi * immersion_medium_refractive_index / wavelength) ** 2 - (kx * self.pupilmask) ** 2 - (
                        ky * self.pupilmask) ** 2)
        # parabola0[np.isnan(parabola0)] = 0
        self.parabola = parabola0 * parabolaMultiplier


class ZernikePolynomials:
    ''' Generating Zernike modes according to Pupil size'''

    def __init__(self,
                 pupil_mask:PupilMask,
                 num:int):

        assert isinstance(pupil_mask, PupilMask) #'First argument should be a PUpilMask instance'
        assert isinstance(num,int) and num>0 #'Enter positive integer for number of Zernike modes'
        try:
            num = int(num)
            if num <=0:
                raise ValueError('Provide positive number of Zernike modes')
        except ValueError:
            raise ValueError('Provide a valid number of Zernike modes')
        self.num = num
        p = pupil_mask
        self.p = p
        rho = np.array(np.sqrt((p.qx - p.center) ** 2 + (p.qy - p.center) ** 2) / p.pupil_pxs)
        phi = np.arctan2((p.qy - p.center), (p.qx - p.center))
        self.rho = rho
        self.phi = phi

        #R1 = lambda n, m, k, rho: (-1) ** k * np.math.factorial(n - k) / (
        #           np.math.factorial(k) * np.math.factorial((n + m) / 2 - k) * np.math.factorial(
        #        (n - m) / 2 - k)) * rho ** (n - 2 * k)
        #Rnm = lambda n, m, rho: np.sum(np.array([R1(n, m, k, rho) for k in range(0, (n - m) / 2 + 1)]), axis=0)
        ## Rnm = lambda n,m,rho: np.array([R1(n,m,k,rho) for k in range(0,(n-m)/2+1)])
        self.Znm1 = lambda n, m: self.Rnm(n, m) * np.cos(m * phi)
        self.Znm2 = lambda n, m: self.Rnm(n, -m) * np.sin(-m * phi)
        self.zernInd = self.genZernInd()
        self.zernStack = self.genZernStack()

    def Rnm(self,n,m):
        out=[]
        for k in range(0, int((n - m) / 2 + 1)):
            out.append(self.R1(n, m, k))
            return np.sum(out,axis=0)

    def R1(self,n,m,k):
        return (-1) ** k * np.math.factorial(n - k) \
               / (np.math.factorial(k) * np.math.factorial((n + m) / 2 - k)
               * np.math.factorial((n - m) / 2 - k)) * self.rho ** (n - 2 * k)

    def switchZnm(self, n, m):
        if m >= 0:
            return self.Znm1(n, m)
        else:
            return self.Znm2(n, m)

    def genZernInd(self):
        i = 0
        n = i
        nmStack = []
        while i < self.num:

            for m in np.arange(-n, n + 1, 2,dtype='int'):
                nmStack.append((n, m))
                i += 1
            n += 1
        self.zernInd = np.array(nmStack,dtype='int')
        return np.array(nmStack,dtype='int')

    def genZernStack(self):
        out = np.array([self.switchZnm(*z) for z in self.zernInd])
        self.zernStack = out
        return out

    def plotZern(self):
        num = self.num
        ind = self.zernInd
        zern = self.zernStack

        r = np.max(ind)
        logging.info('Max order {}, num ind {}'.format(r, len(zern)))
        for i in range(r + 1):
            ind1 = ind[ind[:, 0] == i]
            zern1 = zern[ind[:, 0] == i]
            fig = plt.figure(figsize=(15, 2))
            for k in range(len(zern1)):
                fig.add_subplot(1, len(ind1), k + 1)
                plt.imshow(zern1[k] * self.p.pupilmask)
                plt.axis('off')
                plt.title(ind1[k])
            # plt.tight_layout()
            plt.show()


class PupilFunction(object):
    '''Phase storage object to keep zernike weights updated and to generate psf

    zernWights = A, B, z2....zn
    A - photon number
    B - background'''

    def __init__(self,
                 zernike_polynomials:ZernikePolynomials,
                 pupil:PupilMask,
                 zernWeights:[]=None,
                 pupilWeights:[]=None,
                 ):
        # super(ZernPhase,self).__init__()
        self.pupilmask = pupil.pupilmask
        self.parabola = pupil.parabola
        self.retrieved_pupil = pupil.pupilmask

        self.xslope = pupil.xslope
        self.yslope = pupil.yslope

        self.zernArray = zernike_polynomials.zernStack
        # select circular modes for pupil
        self.pupilZernikeArray = self.zernArray[zernike_polynomials.zernInd[:, 1] == 0]

        if zernWeights is None:
            self.zernWeights = np.zeros(len(zernike_polynomials.zernStack) + 2)
        else:
            self.zernWeights = zernWeights

        if pupilWeights is None:
            self.pupilZernikeWeights = np.zeros(len(self.pupilZernikeArray))
            self.pupilZernikeWeights[0] = 1
        else:
            self.pupilZernikeWeights = pupilWeights

        self.updatePhase()
        self.updatePupil()

        self.gaussSmooth = 0

    def gen_PSF(self, x, y, z, a, b, size):
        # print 'z',z
        # print (self.ret_phase-self.parabola*z-self.xslope*x-self.yslope*y).shape
        ampl, phase = (
            decomp1f(np.abs(self.retrieved_pupil), self.ret_phase - self.parabola * z - self.xslope * x - self.yslope * y))
        # print ampl.shape
        if self.gaussSmooth:
            ampl = ndi.gaussian_filter(ampl, self.gaussSmooth)
        return b + a * norm_sum(cropCenter(ampl, size) ** 2)

    def genPSFarray(self, zvect, size):
        # plt.imshow(self.gen_PSF(.0,.0,zvect[0],100,0,size=32))
        # plt.show()
        a, b = self.zernWeights[0], self.zernWeights[1]
        self.genArray = np.array([self.gen_PSF(0, 0, z, a, b, size) for z in zvect])
        return self.genArray

    def updatePupil(self):
        weights3D = self.pupilZernikeWeights[:].reshape(len(self.pupilZernikeWeights[:]), 1, 1)
        self.retrieved_pupil = self.pupilmask * (np.sum(weights3D * self.pupilZernikeArray[:], axis=0))

    def updatePhase(self):
        weights3D = self.zernWeights[2:].reshape(len(self.zernWeights[2:]), 1, 1)
        self.ret_phase = np.sum(weights3D * self.zernArray[:], axis=0)

    def updateWeight(self, i, val):
        self.zernWeights[i] += val
        self.updatePhase()

    def updatePupilWeight(self, i, val):
        self.pupilZernikeWeights[i] += val
        self.updatePupil()

    def getWeights(self):
        return self.zernWeights.copy()


def ll_test(I, F):
    '''test likelihood and corr_coeff of model stack vs raw stack
    returns MSE of errors'''
    ppp = []
    ccc = []
    # f2 - raw stack
    f2 = I
    # f1 - generated stack
    f1 = F
    # for i in range(1):
    for i in range(len(f2)):
        # for each image in raw stack
        # compute likelihood with generated stack
        ppp.append([LE(f2[i], aa) for aa in f1])
        ccc.append([corr_coeff(f2[i], aa) for aa in f1])
        # plt.imshow(flatten_stack(f1))

        # plt.show()
        # plt.imshow(flatten_stack(f2))
        # plt.show()
    ppp = (np.array(ppp))
    ccc = (np.array(ccc))
    ll_errors = ppp.argmin(axis=1) - np.arange(len(ppp))
    cc_errors = ccc.argmax(axis=1) - np.arange(len(ccc))
    cc_MSE = np.mean(cc_errors ** 2)
    ll_MSE = np.mean(ll_errors ** 2)
    return ll_MSE, cc_MSE


def ll_test_vect(I, F):
    '''
    test likelihood and corr_coeff of model stack vs raw stack
    Returns:
    Likelihood curves
    likelihood errors
    corr_coef curves
    corr_coef errors
    '''
    ppp = []
    ccc = []
    # f2 - raw stack
    f2 = I
    # f1 - generated stack
    f1 = F
    # for i in range(1):
    for i in range(len(f2)):
        # for each image in raw stack
        # compute likelihood with generated stack
        ppp.append([LE(f2[i], aa) for aa in f1])
        ccc.append([corr_coeff(f2[i], aa) for aa in f1])
        # plt.imshow(flatten_stack(f1))

        # plt.show()
        # plt.imshow(flatten_stack(f2))
        # plt.show()
    ppp = (np.array(ppp))
    ccc = (np.array(ccc))
    ll_errors = ppp.argmin(axis=1) - np.arange(len(ppp))
    cc_errors = ccc.argmax(axis=1) - np.arange(len(ccc))
    cc_MSE = np.mean(cc_errors ** 2)
    ll_MSE = np.mean(ll_errors ** 2)
    return ppp, ll_errors, ccc, cc_errors


def plotBias(fVect, ppp, ccc):
    '''
    fVect  - z vector
    ppp - likelihood fit from ll_test_vect()
    ccc - corr-coef from ===

    '''
    fig = plt.figure(figsize=(7, 5))
    fig.add_subplot(2, 2, 1)
    plt.plot(fVect, ppp.T)
    plt.plot(fVect[ppp.argmin(axis=1)], ppp.min(axis=1), 'ro')
    plt.title('likelihoods')

    fig.add_subplot(2, 2, 3)
    plt.plot(fVect, (fVect[1] - fVect[0]) * (ppp.argmin(axis=1) - np.arange(len(ppp))))
    plt.title('ll min index error')
    plt.xlabel('z,um')
    plt.ylabel('z err,um')

    fig.add_subplot(2, 2, 2)
    plt.plot(fVect, ccc.T)
    plt.plot(fVect[ccc.argmax(axis=1)], ccc.max(axis=1), 'ro')
    plt.title('correlation coefficients')

    fig.add_subplot(2, 2, 4)
    plt.plot(fVect, (fVect[1] - fVect[0]) * (ccc.argmax(axis=1) - np.arange(len(ccc))))
    plt.title('cc max index error')
    plt.xlabel('z,um')
    plt.ylabel('z err,um')

    plt.tight_layout()
    plt.show()


class FitZern:
    '''gradient descent for PSF fitting using phase Zernike modes '''

    def __init__(self,
                 zStack,
                 zVector,
                 myPhase,
                 fitPupil=True):

        zStack = np.array(zStack)
        zStack[zStack == 0] = 1
        self.zStack = zStack
        self.size = zStack.shape[-1]
        self.zVector = zVector
        self.ph = myPhase
        self.fitPupil = fitPupil
        self.delta = .2
        self.gamma = 1.
        self.Lh = []
        self.Wh = []
        self.gammas = np.ones_like(self.ph.zernWeights)
        self.pupilGammas = np.ones_like(self.ph.pupilZernikeWeights)
        self.epochs = 0
        I = zStack
        tmp = [np.ravel(I[:, 0:3]), np.ravel(I[:, -4:-1]), np.ravel(I[:, :, 0:3]), np.ravel(I[:, :, -4:-1])]

        bgMean = np.min(np.ravel(tmp))
        a = np.sum(I - bgMean) / len(zVector)
        self.ph.zernWeights[1] = bgMean
        self.ph.zernWeights[0] = a

        # print 'A',self.ph.zernWeights[0],'bg',self.ph.zernWeights[1]

    def fitOneZern(self, i, debug):
        '''compute derivative for i weight of zernike array and update weight'''
        # print 'i = ',i
        g = -self.gammas[i]
        d = self.delta
        if i == 0:
            d = d * 20
        self.ph.updateWeight(i, d)
        tmp = self.ph.genPSFarray(self.zVector, self.size)
        # plt.imshow(flatten_stack(tmp))
        # print 'zStack and genStack shapes',self.zStack.shape, tmp.shape
        s3 = LE(self.zStack, tmp)

        self.ph.updateWeight(i, -2 * d)
        s1 = LE(self.zStack, self.ph.genPSFarray(self.zVector, self.size))

        self.ph.updateWeight(i, d)
        s2 = LE(self.zStack, self.ph.genPSFarray(self.zVector, self.size))

        der1 = (s3 - s1) / (2 * d)
        der2 = np.abs(s3 - 2 * s2 + s1) / (d ** 2)

        if debug:
            logging.debug("s1: {}".format(s1))
            logging.debug("s2: {}".format(s2))
            logging.debug("s3: {}".format(s3))
            logging.debug("der1: {}, der2: {}".format(der1, der2))
            logging.debug("likelihood: {}".format(s2))

        if der2 != 0:
            dw = der1 / der2
            if debug:
                logging.debug("dw*gamma: {}".format(dw))

            # if dw>2: dw = 2
            # elif dw<-2: dw = -2

            if i == 0 or i == 1:
                g = 10 * g

            dw = dw * g
            if i == 0 or i == 1:
                while self.ph.zernWeights[i] + dw < 0:
                    dw /= 10
            if debug:
                logging.debug("finally dw: {}".format(dw))
            self.ph.updateWeight(i, dw)
            LL = LE(self.zStack, self.ph.genPSFarray(self.zVector, self.size))
            it = 0
            while LL > s2 and it < 6:
                self.ph.updateWeight(i, -dw)
                g = g / 10
                dw = der1 / der2 * g

                self.ph.updateWeight(i, dw)
                LL = LE(self.zStack, self.ph.genPSFarray(self.zVector, self.size))
                it += 1
                if debug:
                    logging.debug("g reduction: {}".format(g))
            self.LL = LL
            # self.gammas[i] = g # save gamma


        else:
            if debug: logging.debug('skipping')
            pass

    def fitOnePupilZern(self, i, debug):
        '''compute derivative for i weight of pupil zernike array and update weight'''
        # print 'i = ',i
        g = -self.pupilGammas[i]
        d = self.delta
        # if i ==0:
        #    d=d*20
        self.ph.updatePupilWeight(i, d)
        tmp = self.ph.genPSFarray(self.zVector, self.size)
        # plt.imshow(flatten_stack(tmp))
        # print 'zStack and genStack shapes',self.zStack.shape, tmp.shape
        s3 = LE(self.zStack, tmp)

        self.ph.updatePupilWeight(i, -2 * d)
        s1 = LE(self.zStack, self.ph.genPSFarray(self.zVector, self.size))

        self.ph.updatePupilWeight(i, d)
        s2 = LE(self.zStack, self.ph.genPSFarray(self.zVector, self.size))

        der1 = (s3 - s1) / (2 * d)
        der2 = np.abs(s3 - 2 * s2 + s1) / (d ** 2)

        if debug:
            logging.debug("s1: {}".format(s1))
            logging.debug("s2: {}".format(s2))
            logging.debug("s3: {}".format(s3))
            logging.debug("der1: {}, der2: {}".format(der1, der2))
            logging.debug("likelihood: {}".format(s2))

        if der2 != 0:
            dw = der1 / der2
            if debug:
                logging.debug("dw*gamma: {}".format(dw))

            # if dw>2: dw = 2
            # elif dw<-2: dw = -2

            # if i ==0 or i==1:
            #    g=10*g

            dw = dw * g
            if i == 0:
                while self.ph.pupilZernikeWeights[i] + dw < 0:
                    dw /= 10
            if debug: print('finally dw', dw)
            self.ph.updatePupilWeight(i, dw)
            LL = LE(self.zStack, self.ph.genPSFarray(self.zVector, self.size))
            it = 0
            while LL > s2 and it < 6:
                self.ph.updatePupilWeight(i, -dw)
                g = g / 10
                dw = der1 / der2 * g

                self.ph.updatePupilWeight(i, dw)
                LL = LE(self.zStack, self.ph.genPSFarray(self.zVector, self.size))
                it += 1
                if debug: print('g reduction', g)
            self.LL = LL
            # self.gammas[i] = g # save gamma


        else:
            if debug: print('skipping')
            pass

    def getChi2(self):
        try:
            # print 'chi2 minima',self.ph.genArray.min(), self.zStack.min()
            tmp = np.sum((self.ph.genArray - self.zStack) ** 2 / (self.ph.genArray))
            return tmp
        except ValueError:
            print(traceback.print_exc())

    def fitAll(self, nEpochs=10, debug=0, parallel=0, stop=1e-4, plot=0, size=32):
        epoch = 0
        fitPupil = 0
        kkk = self.ph.genPSFarray(self.zVector, self.size)
        LL = LE(self.zStack, kkk)
        self.Lh.append(LL)
        chi2 = self.getChi2()
        count = 0
        if debug:
            print('initial pupil weights ', self.ph.pupilZernikeWeights)
            print('initial phase weights ', self.ph.zernWeights)
        if parallel:
            pass
        try:
            while epoch < nEpochs:
                if epoch < 2 and self.epochs < 2:
                    zMax = 6
                else:
                    zMax = len(self.ph.zernWeights)
                if parallel:
                    for i in range(0, 2):
                        Process(target=self.fitOneZern, args=(i, debug,))

                        # self.fitOneZern(i,debug)
                else:
                    if fitPupil:
                        for i in range(0, len(self.ph.pupilZernikeWeights)):
                            if debug: print('fitting pupil {}'.format(i))
                            self.fitOnePupilZern(i, debug)
                    for i in range(0, zMax):
                        if debug: print('fitting phase {}'.format(i))
                        self.fitOneZern(i, debug)
                if debug:
                    print('new pupil weights ', self.ph.pupilZernikeWeights)
                    print('new phase weights ', self.ph.zernWeights)

                self.Wh.append(self.ph.getWeights())

                if not parallel:
                    self.Lh.append(self.LL.copy())
                epoch += 1
                self.epochs += 1
                chi2tmp = self.getChi2()
                dL = (self.LL - self.Lh[-2]) / self.LL

                ### plot
                if plot:
                    fig = plt.figure(figsize=(20, 4))
                    plt.imshow(flatten_stack(cropCenter(self.ph.genArray, size)), interpolation='none')
                    plt.show()
                ###

                ll_MSE, cc_MSE = ll_test(self.zStack, self.ph.genArray)

                # if chi2tmp<chi2:
                if np.abs(dL) > stop:
                    # saveWeights = self.ph.zernWeights.copy()
                    # savePupilWeights = self.ph.pupilZernikeWeights.copy()

                    chi2 = chi2tmp
                    saveChi = chi2
                    print('epoch ', self.epochs, 'chi2', chi2, 'LL', self.LL, 'dL', dL, 'll,cc_MSE', ll_MSE,
                          cc_MSE)  # 'gammas ', self.gammas
                    count = 0
                elif fitPupil == 0 and self.fitPupil:
                    fitPupil = 1
                    print('activating pupil fit')
                elif count < 3:
                    count += 1
                    # print 'chi2 increasing, keep trying',count, 'chi2', chi2, 'LL',self.LL
                    print('Likelihood change is less than ', stop, ', keep trying', count, 'chi2', chi2, 'LL', self.LL,
                          'dL', dL, 'll,cc_MSE', ll_MSE, cc_MSE)
                else:
                    # self.ph.zernWeights = saveWeights
                    # self.ph.pupilZernikeWeights = savePupilWeights

                    # print 'stopping because of chi2',saveChi
                    print('stopping because of LL', dL)
                    break
        except Exception as e:
            print(e)
            print('epoch ', self.epochs, 'chi2', chi2, 'LL', self.LL, 'dL', dL, 'll,cc_MSE', ll_MSE,
                  cc_MSE)  # 'gammas ', self.gammas

            # break

        if epoch == nEpochs:
            # print 'end of epochs, chi2',chi2tmp
            print('end of epochs, dL', dL)


class PhaseFitWrap(object):
    def __init__(self, stack=None, NA=1.49, zStep=.1, skip=0, n_oil=1.51, px_size=.11, wl=.64, zern_num=36, smooth=0,
                 fitPupil=True, fileName=None, plot=False, par_mul=1.):
        if fileName:
            myDict = cPickle.load(open(fileName, 'r'))
            img_size = myDict["pupilAmp"].shape[0]
            self.size = img_size
            self.wl = myDict['wl']
            self.NA = myDict['NA']
            self.px_size = myDict['px_size']
            self.n_oil = myDict['n_oil']
            self.ret_pupil = myDict['pupilAmp']
            self.ret_phase = myDict['pupilPhase']
            self.smooth = myDict['gaussSmooth']
            self.stack = myDict['stack']
            self.zStep = myDict['zStep']
            self.zVect = myDict['zVect']
            self.zVect0 = myDict['zVect0']
            self.zernWeights = myDict['zernWeights']
            self.zern_num = len(self.zernWeights) - 2
            self.pupilZernikeWeights = myDict['pupilZernikeWeights']
            self.psfArray = myDict['psfArray']

        else:
            self.NA = NA  # /1.5
            self.n_oil = n_oil
            self.px_size = px_size
            self.zern_num = zern_num
            self.zernWeights = None
            self.pupilZernikeWeights = None
            self.wl = wl
            self.px_size = px_size

        if stack is not None:
            self.stack = (norm_sum(stack)) * stack.sum() / len(stack)
            self.stack0 = self.stack
            self.size = stack.shape[-1]
            self.zStep = zStep
            l = len(stack)
            if l % 2:
                # print l
                self.zVect0 = np.arange(-((l - 1) / 2) * zStep, ((l - 1) / 2 + .5) * zStep, zStep)
            else:
                self.zVect0 = np.arange(-l / 2 * zStep, (l / 2) * zStep, zStep)
            print('full stack ', self.stack.shape)
            if skip:
                self.stack = stack[::skip]
                self.zVect = self.zVect0[::skip]
                print('new stack with skipping', self.stack.shape)
            else:
                self.zVect = self.zVect0

        self.smooth = smooth

        self.pupil = PupilMask(img_size=self.size, numerical_aperture=self.NA, immersion_medium_refractive_index=self.n_oil,
                               pixel_size_um=self.px_size, wavelength=self.wl, parabolaMultiplier=par_mul)
        # self.pupil.parabola = self.pupil.parabola*1.5
        self.zern = ZernikePolynomials(self.pupil, zern_num)
        self.zernPhase = PupilFunction(ZernikeObject=self.zern,
                                       pupil=self.pupil,
                                       zernWeights=self.zernWeights,
                                       pupilWeights=self.pupilZernikeWeights)
        self.zzz = FitZern(self.stack, self.zVect, self.zernPhase, fitPupil)
        self.psfArray = self.zernPhase.genPSFarray(self.zVect, self.size)[:]
        self.fullPsfArray = self.zernPhase.genPSFarray(self.zVect0, self.size)[:]

        if plot:
            plt.figure(figsize=(10, 2))
            plt.imshow(flatten_stack(self.stack))
            plt.title('raw stack with skip')
            plt.show()

    def run(self):
        self.zzz.fitAll(20, parallel=0, debug=0, stop=1e-4, plot=0)
        if self.smooth:
            self.zernPhase.gaussSmooth = self.smooth
        self.psfArray = self.zernPhase.genPSFarray(self.zVect, self.size)[:]
        self.fullPsfArray = self.zernPhase.genPSFarray(self.zVect0, self.size)[:]

    def smooth(self, s):
        self.zernPhase.gaussSmooth = s
        self.psfArray = self.zernPhase.genPSFarray(self.zVect, self.size)[:]
        self.fullPsfArray = self.zernPhase.genPSFarray(self.zVect0, self.size)[:]

    def showBias(self):
        # fullPsfArray = self.zernPhase.genPSFarray(self.zVect0,self.size)[:]
        # print self.stack0.shape, self.fullPsfArray.shape
        ppp, ppp1, ccc, ccc1 = ll_test_vect(self.stack0, self.fullPsfArray)

        plotBias(self.zVect0, ppp, ccc)

    def showResult(self):
        fig = plt.figure(figsize=(8, 2))

        fig.add_subplot(1, 3, 1)
        plt.imshow(self.zernPhase.retrieved_pupil)
        plt.title('retrieved pupil')
        plt.plot(self.size / 2 - self.zernPhase.retrieved_pupil[self.pupil.center] * 10, 'w')
        plt.colorbar()

        fig.add_subplot(1, 3, 2)
        plt.imshow(self.zernPhase.ret_phase * (self.zernPhase.pupilmask))
        plt.title('retrieved phase')
        plt.colorbar()

        fig.add_subplot(1, 3, 3)
        plt.bar(np.arange(len(self.zernPhase.zernWeights[2:])), self.zernPhase.zernWeights[2:])
        plt.title('Zernike weights')

        plt.tight_layout()
        plt.show()

        fig = plt.figure(figsize=(10, 1))
        plt.imshow(flatten_stack(self.stack), cmap='gray')
        plt.axis('off')
        plt.title('raw data')
        plt.colorbar()
        plt.show()

        fig = plt.figure(figsize=(10, 1))
        plt.imshow(flatten_stack(self.psfArray), cmap='gray')
        plt.axis('off')
        plt.title('model')
        plt.colorbar()
        plt.show()

        f1, f2 = self.stack, self.psfArray
        fig = plt.figure(figsize=(10, 1))
        plt.imshow(flatten_stack((f1 - f2) / np.sqrt(f2)), interpolation='none', cmap='gray')
        plt.title('residuals')
        plt.axis('off')
        plt.colorbar()
        plt.show()

        fig = plt.figure(figsize=(6, 4))
        fig.add_subplot(2, 2, 1)
        plt.plot(self.zVect, centroid(f2, axis=(1, 2))[0])
        plt.plot(self.zVect, centroid(f2, axis=(1, 2))[1])

        plt.plot(self.zVect, centroid(f1, axis=(1, 2))[0], 'ob')
        plt.plot(self.zVect, centroid(f1, axis=(1, 2))[1], 'og')

        plt.title('Centroid')
        # plt.legend('xyXY')

        fig.add_subplot(2, 2, 2)
        plt.plot(self.zVect, stdev(f2, axis=(1, 2))[0])
        plt.plot(self.zVect, stdev(f2, axis=(1, 2))[1])

        plt.plot(self.zVect, stdev(f1, axis=(1, 2))[0], 'ob')
        plt.plot(self.zVect, stdev(f1, axis=(1, 2))[1], 'og')

        plt.title('Std xy')
        plt.legend(['x model', 'y model', 'x image', 'y image'], loc=(1, .7))

        fig.add_subplot(2, 2, 3)
        plt.plot(self.zVect, f1.sum(axis=(1, 2)))
        plt.plot(self.zVect, f2.sum(axis=(1, 2)))
        plt.title('Sum')
        plt.xlabel('z,um')

        fig.add_subplot(2, 2, 4)
        plt.plot(self.zVect, f1.max(axis=(1, 2)))
        plt.plot(self.zVect, f2.max(axis=(1, 2)))
        plt.title('Max')
        plt.grid()
        plt.legend(['data', 'model'], loc=(1, .7))
        plt.xlabel('z,um')

        plt.tight_layout()

        plt.show()

    def showCRLB(self, num_phot, bg, ymax=0):

        crlb = ModelPhaseCRLB(self.zVect0.min(), self.zVect0.max(), self.zStep, num_phot, bg, self.zernPhase,
                              size=self.size)
        crlb.runPhase()

        plt.plot(crlb.x_abs, crlb.xCRLB * 1000, crlb.x_abs, crlb.yCRLB * 1000, crlb.x_abs, crlb.zCRLB * 1000)
        plt.title('CRLB, photons: {}, bg: {}'.format(num_phot, bg))
        plt.grid()
        plt.legend('xyz')
        plt.xlabel('z, um')
        plt.ylabel('std(err), nm')
        if ymax:
            plt.ylim([0, ymax])
        plt.show()

    def showCRLB1(self, num_phot, bg, ymax=0):

        crlb = ModelPhaseCRLB(self.zVect0.min(), self.zVect0.max(), self.zStep, num_phot, bg, self.zernPhase,
                              size=self.size)
        crlb.runPhase()

        plt.plot(crlb.x_abs, crlb.CRLB[:, 0] * 1000, crlb.x_abs, crlb.CRLB[:, 1] * 1000, crlb.x_abs,
                 crlb.CRLB[:, 2] * 1000)
        plt.title('CRLB, photons: {}, bg: {}'.format(num_phot, bg))
        plt.grid()
        plt.legend('xyz')
        plt.xlabel('z, um')
        plt.ylabel('std(err), nm')
        if ymax:
            plt.ylim([0, ymax])
        plt.show()

    def dump(self, path):
        '''cPickles oblect to the path'''
        myDict = dict(wl=self.wl,
                      NA=self.NA,
                      px_size=self.px_size,
                      n_oil=self.n_oil,
                      pupilAmp=self.zernPhase.retrieved_pupil,
                      pupilPhase=self.zernPhase.ret_phase,
                      zernWeights=self.zernPhase.zernWeights,
                      pupilZernikeWeights=self.zernPhase.pupilZernikeWeights,
                      gaussSmooth=self.zernPhase.gaussSmooth,
                      zVect0=self.zVect0,
                      zVect=self.zVect,
                      zStep=self.zStep,
                      stack=self.stack,
                      psfArray=self.psfArray)
        cPickle.dump(myDict, open(path, 'wb'))


class CropBeads(object):
    def __init__(self, path, zStep, crop_size, convert2photons=None, gauss=None):
        fileName = sorted(glob.glob(path))
        self.path = path
        print('found ', len(fileName))
        frame0 = io.imread(fileName[0])
        if len(frame0.shape) == 3:
            full_frame_exp_stack = frame0  # [:,52:84,52:84]
            print('reading stack with dimentions', full_frame_exp_stack.shape)
        elif len(frame0.shape) == 2:
            print('Reading file list')
            tmp = []
            for t in fileName:
                tmp.append(io.imread(t))
            full_frame_exp_stack = np.array(tmp)
            print(full_frame_exp_stack.shape)

        if convert2photons:
            print('converting photons')
            full_frame_exp_stack = convert2photons(full_frame_exp_stack)

        z_proj = full_frame_exp_stack.mean(axis=0)
        if gauss:
            z_proj1 = ndi.gaussian_filter(z_proj, gauss)
        else:
            z_proj1 = z_proj
        peaks = peak_local_max(z_proj1, min_distance=crop_size, exclude_border=False)
        print('Showing detection on the flattened stack')
        plt.imshow(z_proj1)
        plt.plot(peaks[:, 1], peaks[:, 0], 'r.')
        plt.show()

        x, y = peaks[:, 1], peaks[:, 0]
        Icoordinates_selected = []
        s = crop_size / 2
        self.crops = []
        self.Icoordinates_selected = []
        for x1, y1 in zip(x, y):
            I = full_frame_exp_stack[:, int(y1 - s):int(y1 + s), int(x1 - s):int(x1 + s)]
            if I.mean(0).shape == (crop_size, crop_size):  # and \
                # len(peak_local_max(I.mean(0),min_distance=crop_size/5,exclude_border=False))==1:

                self.crops.append(I)
                self.Icoordinates_selected.append([y1, x1])

        self.Icoordinates_selected = np.array(self.Icoordinates_selected)

        l = len(full_frame_exp_stack)
        if l % 2:
            # print l
            zVect = np.arange(-((l - 1) / 2) * zStep, ((l - 1) / 2 + .5) * zStep, zStep)
        else:
            zVect = np.arange(-l / 2 * zStep, (l / 2) * zStep, zStep)

        print('found {} non overlapping crops'.format(len(self.Icoordinates_selected)))

        plt.imshow(z_proj)
        if len(self.Icoordinates_selected):
            plt.plot(self.Icoordinates_selected[:, 1], self.Icoordinates_selected[:, 0], 'g.')
        plt.show()

        n = 0

        for i in self.crops:
            plt.figure(figsize=(10, 2))
            plt.imshow(flatten_stack(i[::len(full_frame_exp_stack) / 10]), interpolation='none')
            plt.colorbar()
            # plt.title(len(peak_local_max(i.mean(0),min_distance=3)))
            # plt.title(int(i.mean(0).sum()))
            plt.title(n)
            plt.show()
            n += 1

            wx = stdev(i, axis=(1, 2), rmBg=1)[0]
            wy = stdev(i, axis=(1, 2), rmBg=1)[1]

            plt.figure(figsize=(4, 4))
            plt.plot(zVect, wx)
            plt.plot(zVect, wy)

            plt.plot(zVect, wx - wy)

            plt.legend(['std x', 'std y', 'std(x-y)'])
            plt.grid()
            plt.show()


class ReconstructionMLE:
    def __init__(self,
                 folder,
                 myFit,
                 crop_size=32,
                 min_photons=100,
                 min_distance=16,
                 ERF=LE,
                 bg_kernel=10,
                 x_thr=.2,
                 fileList=None,
                 photon_convert=None):
        self.folder = folder
        self.psfArray = myFit.psfArray
        self.px = myFit.px_size
        self.zVect = myFit.zVect
        self.zernPhase = myFit.zernPhase
        self.ERF = ERF
        self.crop_size = crop_size
        self.min_photons = min_photons
        self.min_distance = min_distance
        self.x_thr = x_thr
        self.bg_kernel = bg_kernel
        self.preCrop = lambda f: f
        self.pool = ActivePool()
        self.jobs = []
        self.sem = multiprocessing.Semaphore(cpu_count())
        self.workers = []
        self.found = np.empty((0, 6))

        if fileList == None:
            self.testFileName = sorted(glob.glob(folder + '*.tif'))
        else:
            self.testFileName = fileList
        print('found {} files in the folder'.format(len(self.testFileName)))

        if photon_convert == None:
            self.photon_convert = lambda f: f
        else:
            self.photon_convert = photon_convert

        self.log = []

        self.xc = XcorrMLE(self.psfArray,
                           self.zVect,
                           self.zernPhase,
                           MLE_ERF=LE,
                           crop_size=self.crop_size,
                           x_thr=self.x_thr,
                           min_photons=self.min_photons,
                           pool=0,
                           min_distance=self.min_distance,
                           bg_kernel=self.bg_kernel)

    def stop(self):
        for j in self.workers:
            j.terminate()
            print(self.workers)

    def updateFolder(self):
        self.testFileName = sorted(glob.glob(self.folder + '*.tif'))
        print('found {} files in the folder'.format(len(self.testFileName)))

    def poolMLE(self, frame):
        out = self.xc.run(frame)
        return out

    def progress(self):
        print('{} frames, {} particles'.format(len(self.pool.exitCodes), len(self.pool.arr)))

    def worker(self):
        name = multiprocessing.current_process().name
        # print 'start worker {}'.format(name)
        px = self.px
        try:
            while 1:
                i = self.qIn.get(True, 5)
                with self.pool.lock:
                    frame = self.readFrame(i)
                out = self.poolMLE(frame)
                with self.pool.lock:
                    for o in out:
                        self.pool.arr.append(
                            [int(i), (-o.x + o.X * px) * 1000, (-o.y + o.Y * px) * 1000, (o.z) * 1000, int(o.a),
                             int(o.b)])
                    self.pool.exitCodes.append(multiprocessing.current_process().exitcode)
                    print('\r{} frames, {} particles'.format(len(self.pool.exitCodes), len(self.pool.arr)))
        except Queue.Empty:
            return

    def collectData(self):
        try:
            if len(self.pool.arr) > len(self.found):
                self.found = np.append(self.found, self.pool.arr[self.found.shape[0]:], axis=0)

        except Exception as e:
            print(e)
        return self.found.copy()

    def parallelProcessing(self, start=None, end=None):

        try:
            if end == None:
                end = len(self.testFileName)
            if start == None:
                start = 0
            data = np.arange(start, end)
            self.qIn = multiprocessing.Queue()
            for i in data:
                self.qIn.put(i)
                print('\r', i)
            print('created file queue')

            if len(self.workers) > 0:
                for i in range(len(self.workers)):
                    j = self.workers.pop()
                    j.terminate()

            for i in range(cpu_count()):
                self.workers.append(multiprocessing.Process(target=self.worker, name=str(i), args=()))

            for j in self.workers:
                try:
                    j.daemon = True
                    j.start()
                except AssertionError:
                    j.terminate()

            print('started {} workers'.format(len(self.workers)))
        except Exception as e:
            traceback.print_exc()
            print(e)

    def readFrame(self, frameNum):
        return self.preCrop(self.photon_convert(io.imread(self.testFileName[frameNum])))

    def testPredetections(self, frameNum):
        frame = self.readFrame(frameNum)
        print(frame.shape)

        out = self.xc.run(frame)

        plt.figure(figsize=(12, 3))
        plt.subplot(1, 4, 1)
        plt.imshow(frame)
        plt.title('original frame')

        plt.subplot(1, 4, 2)
        plt.imshow(self.xc.img_wo_bg)
        plt.title('bg subtraction')

        plt.subplot(1, 4, 3)
        self.xc.showXcorr()
        plt.title('xcorr frame')

        plt.subplot(1, 4, 4)
        self.xc.showDetections()
        plt.title('original frame localizations')

        for i in range(len(self.xc.crops)):
            I = self.xc.crops[i].I
            xI = self.zernPhase.gen_crop_init(self.xc.crops[i])
            F = self.zernPhase.gen_crop(self.xc.crops[i])
            R = (I - F) / np.sqrt(F)
            R1 = R.copy()
            R1[F < F.min() + .1 * (F.max() - F.min())] = 0

            plt.figure(figsize=(10, 2))

            plt.subplot(1, 5, 1)
            plt.imshow(I, interpolation='none')
            plt.title('raw crop \n mean %i, var %i' % (I.mean(), np.var(I)))
            plt.colorbar()

            plt.subplot(1, 5, 2)
            plt.imshow(xI, interpolation='none')
            plt.title('xcorr result')
            plt.colorbar()

            plt.subplot(1, 5, 3)
            plt.imshow(F, interpolation='none')
            plt.title('MLE result')
            plt.colorbar()

            plt.subplot(1, 5, 4)
            plt.imshow(R, interpolation='none')
            plt.title('residuals %.2f' % np.std(R))
            plt.colorbar()

            plt.subplot(1, 5, 5)
            plt.imshow(R1, interpolation='none')
            plt.title('residuals %.2f' % np.std(R1[R1 != 0]))
            plt.colorbar()
            plt.tight_layout()

            plt.show()

    def setCrop(self, xmin, xmax, ymin, ymax):
        self.preCrop = lambda f: f[ymin:ymax, xmin:xmax]


class ZernPhaseLt(PupilFunction, object):
    def __init__(self, **kwargs):
        '''
        Zernike phase for network manager.
        Expecting a dictionary argument:
        dict(NA = myFit.NA,
              px = myFit.px,
              n_oil = myFit.n_oil,
              pupilAmp = myFit.zernPhase.ret_pupil,
              pupilPhase = myFit.zernPhase.ret_phase,
              zernWeights = myFit.zernPhase.zernWeights,
              gaussSmooth = myFit.zernPhase.gaussSmooth)
        '''
        img_size = kwargs["pupilAmp"].shape[0]
        self.pupil = PupilMask(img_size=img_size, numerical_aperture=kwargs['NA'], pixel_size_um=kwargs['px'],
                               immersion_medium_refractive_index=kwargs['n_oil'])
        self.ret_pupil = kwargs['pupilAmp']
        self.ret_phase = kwargs['pupilPhase']
        self.parabola = self.pupil.parabola
        self.xslope = self.pupil.xslope
        self.yslope = self.pupil.yslope
        self.gaussSmooth = kwargs['gaussSmooth']
