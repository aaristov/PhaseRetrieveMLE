import numpy as np
import _pickle as cPickle
from skimage.feature import match_template

import glob

from SPsample import *
import traceback

from multiprocessing import Pool,cpu_count
import threading
#from pathos.multiprocessing import ProcessingPool as Pool
#from pathos.multiprocessing import cpu_count
#global pupil_init

#global s # pool
def rmMeanBg(arr):
    '''
    Removes background according to mean oat the edge of the image stack
    '''
    if len(np.array(arr).shape)==3:
        bg = arr[:,:,:arr.shape[-1]/10].mean()
    if len(np.array(arr).shape)==2:
        bg = arr[:,:arr.shape[-1]/10].mean()
    newarr = np.array(arr)-bg
    newarr[newarr<0]=0
    return newarr


def centroid(arr,axis,rmBg=0):
    ind = np.indices(arr.shape)
    if rmBg:
        arr=rmMeanBg(arr)
    cx = np.sum((ind[-1])*arr,axis=axis)/arr.sum(axis=axis)
    cy = np.sum((ind[-2])*arr,axis=axis)/arr.sum(axis=axis)
    return cx,cy
def variance(arr,axis,rmBg=0):
    ind = np.indices(arr.shape)
    cx,cy = centroid(arr,axis,rmBg)
    if rmBg:
        arr=rmMeanBg(arr)
    if cx.shape:
        vx = np.sum(arr*(ind[-1]- np.resize(cx,(len(cx),1,1)))**2,axis=axis)/np.sum(arr,axis=axis)
        vy = np.sum(arr*(ind[-2]- np.resize(cx,(len(cx),1,1)))**2,axis=axis)/np.sum(arr,axis=axis)
    else:
        vx = np.sum(arr*(ind[-1]- np.resize(cx,(ind[-1].shape)))**2,axis=axis)/np.sum(arr,axis=axis)
        vy = np.sum(arr*(ind[-2]- np.resize(cx,(ind[-2].shape)))**2,axis=axis)/np.sum(arr,axis=axis)

    return vx,vy

def stdev(arr,axis,rmBg=0):
    return np.sqrt(variance(arr,axis,rmBg))


def propagate(I0,dz,smooth):
    E0=np.sqrt(I0)
    #E0_fft= (np.fft.fft2(E0))
    E0_fft= np.fft.fftshift(np.fft.fft2(E0))
    #E1 = np.fft.ifft2(np.fft.ifftshift(np.exp(-1j*dz*parabola1/(2*k)) * np.fft.fftshift(E0_fft)))
    phase = np.angle(E0_fft)
    #plt.imshow(phase)
    #plt.show()
    if smooth:
        phase = ndi.filters.gaussian_filter(phase,smooth)

    #plt.imshow(phase)
    #plt.show()
    E0_fft1 = gauss_fft1 * np.exp(1j*(phase*pupilmask))
    #E1 = np.fft.ifft2(np.fft.ifftshift(np.exp(-1j*dz*parabola1/(2*k)) * np.fft.fftshift(np.real(gauss_fft1) + 1j*(np.imag(E0_fft)))))
    E1 = np.fft.ifft2(np.fft.ifftshift(np.exp(-1j*dz*parabola1/(2*k)) * E0_fft1))
    #return E1**2,np.angle(np.fft.fftshift(np.fft.fft2(E1)))
    return E1**2,np.angle(E0_fft1)#np.fft.fftshift(np.fft.fft2(E1)))

def replace_ampl(I1,I2):
    E1=np.sqrt(I1)
    E2=np.sqrt(I2)

    #E1.real = E2
    return (np.abs(E2) * np.exp( 1j*np.angle(E1)))**2

def phase_retrieve(stack,zvector,iterations=15,zstep=.5,debug=0,smooth=5):
    '''
    Performs iterative phase retrieval:

    stack shoud have odd number of layers
    iterations = number of turns
    zstep in um between stack layers
    '''
    plt.imshow(stack.reshape(stack.shape[0]*stack.shape[1],stack.shape[2]).T,interpolation='none')
    #PSF_pad = np.zeros((stack.shape[0],512,512))
    #PSF_pad[:,256-16:256+16,256-16:256+16]=(stack)
    #print PSF_pad.shape
    SP_fft = np.abs(np.fft.fftshift(np.fft.fft2(np.sqrt(stack[3]))))

    stack_size = stack.shape[0]
    stack_center=stack_size//2
    rolling_vector = np.concatenate([np.arange(stack_size)[stack_center:-1],np.arange(stack_size)[:0:-1],np.arange(stack_size)[:stack_center+1]])
    rolling_vector_z = np.concatenate([zvector[stack_center:-1],zvector[:0:-1],zvector[:stack_center+1]])
    #print 'Stack size: %d, center: %d'%(stack_size,stack_center)
    #print rolling_vector
    #print rolling_vector_z
    I0=stack[rolling_vector[-1]]
    prev_index=rolling_vector[-1]
    SE = []
    prev_phase = []
    fig = plt.figure()
    for iter in range(iterations):
        #print '\riteration %d'%(iter+1),
        for step in range(len(rolling_vector)-1):
            #print rolling_vector[step]
            index=rolling_vector[step]
            I1=(stack[index])
            I01, phase = propagate(I0,(index-prev_index)*zstep,smooth)

            if debug:
                #print phase
                fig=plt.figure(figsize=(10,2))
                fig.add_subplot(1,5,1)
                plt.imshow(np.abs(I0)[256-7:256+8,256-7:256+8],interpolation='none')
                plt.title('step %d E0'%rolling_vector[step])
                plt.axis('off')


                fig.add_subplot(1,5,2)
                plt.imshow(np.abs(I01)[256-7:256+8,256-7:256+8],interpolation='none')
                plt.title('I1')
                plt.axis('off')

                fig.add_subplot(1,5,3)
                plt.imshow(np.abs(np.abs(np.fft.fftshift(np.fft.fft2(np.sqrt(I01))))),interpolation='none')
                plt.colorbar()
                plt.title('E1 real')
                plt.axis('off')

                fig.add_subplot(1,5,4)
                plt.imshow((phase),interpolation='none')
                plt.colorbar()
                plt.title('E1 imag')
                plt.axis('off')

                fig.add_subplot(1,5,5)
                plt.imshow(I1[256-7:256+8,256-7:256+8],interpolation='none')
                plt.title('E1 from stack')
                plt.axis('off')
                plt.tight_layout()
                plt.show()


            I0 = replace_ampl(I01,I1)
            prev_index=index
        #plt.imshow((phase))
        if type(prev_phase).__module__ == np.__name__:
            SE.append(np.sum((phase-prev_phase)**2*pupilmask))
        prev_phase=phase

        scan = SP_z_scan(gauss_fft1,phase,-0.4,0.4,.1)
        #plt.figure(figsize=(20,3))
        plt.imshow(np.reshape(scan,(scan.shape[1]*scan.shape[0],scan.shape[2])).T,interpolation='none')
        plt.show()
    plt.clf()
    plt.plot(SE)
    plt.show()

    return phase

def SP_z_scan(pupil_init,phase,start,stop,step,parabola):
    scan = []
    scanPhase = []

    for z in np.arange(start,stop,step):
        #print z
        #tmp=(np.fft.ifft2(np.fft.ifftshift( SP_fft *np.exp(1j*phase-1j*z*parabola1/(2*k)))))[256-15:256+16,256-15:256+16]**2
        #tmp=(np.fft.ifft2(np.fft.ifftshift( SP_fft *np.exp(1j*phase-1j*z*parabola1/(2*k)))))**2
        #tmp=np.fft.ifft2(np.fft.ifftshift( SP_fft *np.exp(1j*phase-1j*z*parabola/(2*k))))
        tmp=np.fft.ifft2(np.fft.fftshift( pupil_init *np.exp(1j*phase-1j*z*parabola)))
        scan.append(np.abs(tmp))
        scanPhase.append(np.angle(tmp))
    return np.array((scan)),np.array((scanPhase))




def phase_retrieve_parallel_f(stack,parabola,mask,iterations=2,zstep=.1,phaseinit=0,initialZ=0,stop = 10,debug=0):
    if len(stack.shape) == 3:
        stackCenter=len(stack)/2
        deltaZ = np.zeros(len(stack))
        for s in range(len(stack)):
            deltaZ[s]=(s-stackCenter)*zstep
        defocus = parabola*np.fft.fftshift(deltaZ).reshape(len(deltaZ),1,1)
    else:
        defocus = 0

    saveAmpli=np.zeros_like(stack[0])
    phaseFmean=smooth_phase(phaseinit,mask)
    phaseFprop = np.zeros(stack.shape)
    phaseF = np.zeros(stack.shape)
    ampliF = np.zeros(stack.shape)
    saveAmpliFI = 0
    savePhaseFI = 0
    stab_AmpliF,stab_PhaseF = [],[]
    t = time.time()
    for iter in range(iterations):
        #propagation

        if debug:
            plt.imshow(smooth_phase(phaseFmean,mask))
            plt.show()
        phaseFprop=phaseFmean-defocus
        #phaseFprop=phaseFmean+defocus
        ampli,phase=decomp12(mask,phaseFprop)
        #image replacement
        ampliF,phaseF=decomp22(stack,phase)
        phaseF=phaseF+defocus
        #phaseF=phaseF-defocus

        saveAmpli=ampliF.mean(axis=0)
        #averaging phase
        cos=np.cos(phaseF).mean(axis=0)
        sin=np.sin(phaseF).mean(axis=0)
        phaseFmean=np.arctan2(sin,cos)
        #print '\r',iter,

        if debug:
            #print len(ampli.shape)

            plt.imshow(flatten_stack(ampli))
            plt.show()

        #stack_diffI.append(stack_diff)

        stab_AmpliF.append(np.abs((saveAmpliFI-saveAmpli)**2*(mask>0)).mean())
        stab_PhaseF.append(np.abs((savePhaseFI-phaseFmean)**2*(mask>0)).mean())
        savePhaseFI, saveAmpliFI = phaseFmean, saveAmpli


        #print 'Phase variance',stab_PhaseF[-1],
        if stab_PhaseF[-1]<stop:
            #print 'converged'
            break
    return saveAmpli,phaseFmean,np.array(stab_AmpliF), np.array(stab_PhaseF),0

def phase_retrieve_parallel_float_pupil(stack,parabola,mask,iterations=2,zstep=.1,phaseinit=0,initialZ=0,stop = 10,debug=0):
    if debug:
        import matplotlib.pyplot as plt
    if len(stack.shape) == 3:
        stackCenter=len(stack)/2
        deltaZ = np.zeros(len(stack))
        for s in range(len(stack)):
            deltaZ[s]=(s-stackCenter)*zstep
        defocus = parabola*np.fft.fftshift(deltaZ).reshape(len(deltaZ),1,1)
    else:
        defocus = 0

    saveAmpli=np.ones_like(stack[0])*mask
    phaseFmean=smooth_phase(phaseinit,mask)
    #phaseFprop = np.zeros(stack.shape)
    phaseF = np.zeros(stack.shape)
    ampliF = np.zeros(stack.shape)
    saveAmpliFI = 0
    savePhaseFI = 0
    stab_AmpliF,stab_PhaseF = [],[]
    t = time.time()
    for iter in range(iterations):
        #propagation

        if debug:

            plt.imshow(smooth_phase(phaseFmean,mask))
            plt.show()
        phaseFprop=phaseFmean-defocus
        #phaseFprop=phaseFmean+defocus
        ampli,phase=decomp12(saveAmpli,phaseFprop)
        #image replacement
        ampliF,phaseF=decomp22(stack,phase)
        phaseF=phaseF+defocus
        #phaseF=phaseF-defocus

        saveAmpli=ampliF.mean(axis=0)
        #averaging phase
        cos=np.cos(phaseF).mean(axis=0)
        sin=np.sin(phaseF).mean(axis=0)
        phaseFmean=np.arctan2(sin,cos)
        #print '\r',iter,

        if debug:
            #print len(ampli.shape)

            plt.imshow(flatten_stack(ampli-stack))
            plt.show()

        #stack_diffI.append(stack_diff)

        #stab_AmpliF.append(np.abs((saveAmpliFI-saveAmpli)**2*(mask>0)).mean())
        #stab_PhaseF.append(np.abs((savePhaseFI-phaseFmean)**2*(mask>0)).mean())
        stab_PhaseF1 = (np.abs((savePhaseFI-phaseFmean)**2*(mask>0)).mean())
        savePhaseFI, saveAmpliFI = phaseFmean, saveAmpli
        stab_AmpliF.append(saveAmpli)
        stab_PhaseF.append(phaseFmean)

        #print 'Phase variance',stab_PhaseF1,
        if stab_PhaseF1 < stop:
            #print 'converged'
            break
    return saveAmpli,phaseFmean,np.array(stab_AmpliF), np.array(stab_PhaseF),0

def phase_retrieve_parallel_float_pupil_smooth_phase(stack,vector,parabola,mask,iterations=2,zstep=.1,phaseinit=0,initialZ=0,stop = 10,debug=0):
    stack = stack/stack.sum(axis=(-2,-1)).reshape(len(stack),1,1)

    if len(stack.shape) == 3:
        '''
        stackCenter=len(stack)/2
        s=len(stack)
        print s
        #deltaZ = np.zeros(len(stack))
        #for s in range(len(stack)):
        #    deltaZ[s]=(s-stackCenter)*zstep
        #
        deltaZ = np.arange((-stackCenter)*zstep,(s-stackCenter)*zstep,zstep)
        '''
        deltaZ = vector
        #print deltaZ
        defocus = parabola*np.fft.fftshift(deltaZ).reshape(len(deltaZ),1,1)

    else:
        defocus = 0
    if debug:
        import matplotlib.pyplot as plt
        #print 'DeltaZ len', len(deltaZ)
    #saveAmpli=np.ones_like(stack[0])*mask
    saveAmpli=mask
    #phaseFmean=smooth_phase(phaseinit,mask)
    phaseFmean=np.ones_like(mask)*phaseinit
    #phaseFprop = np.zeros(stack.shape)
    phaseF = np.zeros(stack.shape)
    ampliF = np.zeros(stack.shape)
    saveAmpliFI = 0
    savePhaseFI = 0
    stab_AmpliF,stab_PhaseF = [],[]
    t = time.time()
    for iter in range(iterations):
        #propagation

        phaseFprop=phaseFmean-defocus
        #phaseFprop=phaseFmean+defocus
        ampli,phase=decomp1(saveAmpli,phaseFprop)

        if debug:
            #print 'phase shape ', phase.shape
            fig = plt.figure()
            fig.add_subplot(121)
            #print 'phaseFmean shape ',phaseFmean.shape
            plt.imshow(phaseFmean)
            fig.add_subplot(122)
            plt.imshow(saveAmpli)

            plt.show()
        #image replacement
        ampliF,phaseF=decomp2(stack,phase)
        phaseF=phaseF+defocus
        #phaseF=phaseF-defocus

        saveAmpli=ampliF.mean(axis=0)#*(mask>0)
        #averaging phase
        cos=np.cos(phaseF).mean(axis=0)
        sin=np.sin(phaseF).mean(axis=0)
        phaseFmean=np.arctan2(sin,cos)
        #print '\r',iter,

        if debug:
            #print len(ampli.shape)

            #plt.imshow(flatten_stack(ampli-stack))
            plt.imshow(flatten_stack(ampli)**2)
            plt.title('Generated stack, {}'.format(ampli[0].sum()))
            plt.show()
            plt.imshow(flatten_stack(ampli-stack)**2)
            plt.title('difference')
            plt.show()

        #stack_diffI.append(stack_diff)

        #stab_AmpliF.append(np.abs((saveAmpliFI-saveAmpli)**2*(mask>0)).mean())
        #stab_PhaseF.append(np.abs((savePhaseFI-phaseFmean)**2*(mask>0)).mean())
        stab_PhaseF1 = (np.abs((savePhaseFI-phaseFmean)**2*(mask>0)).mean())
        savePhaseFI, saveAmpliFI = phaseFmean, saveAmpli
        #stab_AmpliF.append(saveAmpli)
        #stab_PhaseF.append(phaseFmean)

        #print 'Phase variance',stab_PhaseF1,
        if stab_PhaseF1 < stop:
            #print 'converged'
            break
    #return saveAmpli,phaseFmean,np.array(stab_AmpliF), np.array(stab_PhaseF),0
    return saveAmpli,phaseFmean,0, 0,0


def phase_retrieve_parallel_float_pupil_smooth_phase_pool(stack,parabola,mask,pool,iterations=2,zstep=.1,phaseinit=0,initialZ=0,stop = 10,debug=0):

    s=pool
    #print s
    stack = stack/stack.sum(axis=(-2,-1)).reshape(len(stack),1,1)
    if debug:
        import matplotlib.pyplot as plt
    if len(stack.shape) == 3:
        stackCenter=len(stack)/2
        s1=len(stack)
        #deltaZ = np.zeros(len(stack))
        #for s in range(len(stack)):
        #    deltaZ[s]=(s-stackCenter)*zstep
        #
        deltaZ = np.arange(-(s1-stackCenter)*zstep,(s1-stackCenter)*zstep,zstep)
        #print deltaZ
        defocus = parabola*np.fft.fftshift(deltaZ).reshape(len(deltaZ),1,1)
        defocus1 = parabola*(deltaZ).reshape(len(deltaZ),1,1)
    else:
        defocus = 0

    #saveAmpli=np.ones_like(stack[0])*mask
    saveAmpli=mask
    #phaseFmean=smooth_phase(phaseinit,mask)
    phaseFmean=np.ones_like(mask)*phaseinit
    #phaseFprop = np.zeros(stack.shape)
    phaseF = np.zeros(stack.shape)
    ampliF = np.zeros(stack.shape)
    saveAmpliFI = 0
    savePhaseFI = 0
    stab_AmpliF,stab_PhaseF = [],[]
    t = time.time()


    def parallel_fit(x): # x = zip(defocus1,stack)
        defocus = x[0]
        stack = x[1]
        phaseFprop=phaseFmean-defocus
        ampli,phase=decomp1(saveAmpli,phaseFprop)
        ampliF,phaseF=decomp2(stack,phase)
        return(ampliF,phaseF+defocus)

    try:
        #s = Pool(cpu_count())
        #s.restart()
        #print 'Found ',s.ncpus,' cores'
        #res  = s.map(lambda x: x**2, range(4))
        #print res

        for iter in range(iterations):
            #propagation


            phaseFprop=phaseFmean-defocus
            #phaseFprop=phaseFmean+defocus
            ampli,phase=decomp1(saveAmpli,phaseFprop)

            if debug:
                fig = plt.figure()
                fig.add_subplot(121)
                #print phaseFmean.shape
                plt.imshow(phaseFmean)
                fig.add_subplot(122)
                plt.imshow(saveAmpli)

                plt.show()
            #image replacement
            ampliF,phaseF=decomp2(stack,phase)
            phaseF=phaseF+defocus
            #phaseF=phaseF-defocus

            res = s.map(parallel_fit,(list(defocus1),list(stack)))
            np_res = np.array(res)
            ampliF = np_res[:,0]
            phaseF = np_res[:,1]

            saveAmpli=ampliF.mean(axis=0)
            #averaging phase
            cos=np.cos(phaseF).mean(axis=0)
            sin=np.sin(phaseF).mean(axis=0)
            phaseFmean=np.arctan2(sin,cos)
            #print '\r',iter,

            if debug:
                plt.imshow(phaseFmean)
                plt.title('Found phase' )
                plt.show()
                ''''''
                #print len(ampli.shape)

                #plt.imshow(flatten_stack(ampli-stack))
                plt.imshow(flatten_stack(ampli)**2)
                plt.title(ampli[0].sum())
                plt.show()
                plt.imshow(flatten_stack(ampli-stack)**2)
                #plt.imshow(flatten_stack(ampli))
                plt.show()

            #stack_diffI.append(stack_diff)

            #stab_AmpliF.append(np.abs((saveAmpliFI-saveAmpli)**2*(mask>0)).mean())
            #stab_PhaseF.append(np.abs((savePhaseFI-phaseFmean)**2*(mask>0)).mean())
            stab_PhaseF1 = (np.abs((savePhaseFI-phaseFmean)**2*(mask>0)).mean())
            savePhaseFI, saveAmpliFI = phaseFmean, saveAmpli
            stab_AmpliF.append(saveAmpli)
            stab_PhaseF.append(phaseFmean)

            #print 'Phase variance',stab_PhaseF1,
            if stab_PhaseF1 < stop:
                #print 'converged'
                break

    except:
        print(traceback.print_exc())
    #finally:
        #s.close()
        #s.join()
    return saveAmpli,phaseFmean,np.array(stab_AmpliF), np.array(stab_PhaseF),0


def propagation(ampliF,phaseF,deltaZ,parabola):
    imF=ampliF*np.exp(1j*phaseF)
    #propF=imF*np.exp(-1j*deltaZ*parabola/(2*k))
    propF=imF*np.exp(-1j*deltaZ*parabola)
    #print "prop : ",deltaZ
    return np.abs(propF),np.angle(propF)

def decomp1(ampli,phase=0):
    ''' 2D FFT from camera to pupil'''
    im=np.abs(ampli)*np.exp(1j*phase)
    imfft=np.fft.fftshift(np.fft.fft2(im))
    ampliF=np.abs(imfft)
    phaseF = np.angle(imfft)
    return ampliF,phaseF

def decomp1f(ampli,phase=0): # faster inplementation wo angle
        ''' 2D FFT from camera to pupil'''
        im=np.abs(ampli)*np.exp(1j*phase)
        imfft=np.fft.fftshift(np.fft.fft2(im))
        ampliF=np.abs(imfft)
        phaseF = 0#np.angle(imfft)
        return ampliF,phaseF

def decomp12(ampli,phase=0): # pupil to image
    im=np.abs(ampli)*np.exp(1j*phase)
    imfft=(np.fft.ifft2(np.fft.ifftshift(im)))
    ampliF=np.abs(imfft)
    phaseF = np.angle(imfft)
    return ampliF,phaseF

def decomp2(ampliF,phaseF):
    ''' 2D FFT from pupil to camera'''
    im=ampliF*np.exp(1j*phaseF)
    imfft=np.fft.ifft2(np.fft.ifftshift(im))
    ampli=np.abs(imfft)
    phase = np.angle(imfft)
    return ampli,phase

def decomp22(ampliF,phaseF): # image to pupil
    im=ampliF*np.exp(1j*phaseF)
    imfft=np.fft.fftshift(np.fft.fft2(im))
    ampli=np.abs(imfft)
    phase = np.angle(imfft)
    return ampli,phase

smooth_phase = lambda phase,pupil_init: np.angle(pupil_init*np.exp(1j*phase))


def flatten_stack(stack,direction='h'):
    '''Select the right direction to flatten: v for vertical and h for horizontal'''
    stack = np.array(stack)
    if len(stack.shape) == 3:
        if direction=='h':
            return np.reshape(np.transpose(np.array(stack),axes=(0,2,1)),(stack.shape[1]*stack.shape[0],stack.shape[2])).T
        elif direction=='v':
            return np.reshape(np.transpose(np.array(stack),axes=(0,1,2)),(stack.shape[1]*stack.shape[0],stack.shape[2]))
        else:
            pass
            #print 'Select the right direction to flatten: v for vertical and h for horizontal'
    else:
        return stack
"""
class ModelPhaseCRLB:

    def __init__(self,minZ, maxZ,zstep,num_phot,bg,pupilmask,phase,sph,xslope,yslope):
        self.phase = phase
        self.num_phot = num_phot
        self.bg = bg
        self.pupilmask = pupilmask
        self.sph = sph
        self.xslope = xslope
        self.yslope = yslope
        self.minZ=minZ;
        self.maxZ=maxZ;
        #self.phase=phase;
        self.taille=phase.shape[0];
        self.zstep = zstep
        self.step = 1.

    def runPhase(self):
        nb=(int)((self.maxZ-self.minZ)/self.zstep);
        zrange = np.arange(self.minZ,self.maxZ,self.zstep)
        zrangel = len(zrange)

        self.x_abs= zrange;
        self.xCRLB= np.zeros(zrangel);
        self.yCRLB= np.zeros(zrangel);
        self.zCRLB= np.zeros(zrangel);
        for u in range(zrangel):
            self.xCRLB[u]=(self.crlbX(0, 0, zrange[u], .001));
            self.yCRLB[u]=(self.crlbY(0, 0, zrange[u], .001));
            self.zCRLB[u] = (self.crlbZ(0, 0, zrange[u], .01));
            #self.x_abs.append(u);
            #IJ.log("crlb "+zCRLB[k]);
        self.xCRLB = np.array(self.xCRLB)
        self.yCRLB = np.array(self.yCRLB)
        self.zCRLB = np.array(self.zCRLB)



    def crlbZ(self,x, y, z,hdec):
        f2 = self.computeF(x, y, z+hdec);
        f1 = self.computeF(x, y, z);
        f0 = self.computeF(x, y, z-hdec);
        truc=(f2-f0)/(2*(hdec));
        res=np.sum((1/f1)*truc*truc);

        return np.sqrt(1/res);

    def crlbX(self,x, y, z,hdec):
        f2 = self.computeF(x+hdec, y, z);
        f1 = self.computeF(x, y, z);
        f0 = self.computeF(x-hdec, y, z);
        truc=(f2-f0)/(2*(hdec));
        res=np.sum((1/f1)*truc*truc);
        #res[res==0]=1e-5
        return np.sqrt(1/res);


    def crlbY(self,x, y, z,hdec):
        f2 = self.computeF(x, y+hdec, z);
        f1 = self.computeF(x, y, z);
        f0 = self.computeF(x, y-hdec, z);
        truc=(f2-f0)/(2*(hdec));
        res=np.sum((1/f1)*truc*truc);
        #res[res==0]=1e-5
        return np.sqrt(1/res);


    def computeF(self,x, y, z):
        f=(self.dataGenerator(x,y,z));
        #f=np.random.poisson(self.dataGenerator(x,y,z)).astype('float64');

        '''
        for i in range(self.taille):
            for ii in range(self.taille):
                f[i][ii]*=a;
                f[i][ii]+=b;
                if (f[i][ii]<0):
                    f[i][ii]=0.000000001;
        '''
        #f[f<=0] = 1e-10


        return f;

    def dataGenerator(self,x,y,z):
        return gen_PSF23(x,y,z,self.num_phot,self.bg,self.pupilmask,self.phase,self.sph,self.xslope,self.yslope)
"""


class ModelPhaseCRLB_old(object):

    def __init__(self,minZ, maxZ,zstep,num_phot,bg,my_phase,size=64,units='um'):
        self.phase = my_phase.ret_phase
        self.num_phot = num_phot
        self.bg = bg
        self.pupilmask = my_phase.ret_pupil
        self.sph = my_phase.parabola
        self.xslope = my_phase.xslope
        self.yslope = my_phase.yslope
        self.minZ=minZ;
        self.maxZ=maxZ;
        #self.phase=phase;
        self.taille=self.phase.shape[0];
        if units == 'um':
            self.zstep = zstep
        elif units == 'nm':
            self.zstep = zstep/1000.
        self.step = 1.
        self.dataGenerator = my_phase.gen_PSF
        self.size=size

    def runPhase(self):
        nb=(int)((self.maxZ-self.minZ)/self.zstep);
        zrange = np.arange(self.minZ,self.maxZ,self.zstep)
        zrangel = len(zrange)

        self.x_abs= zrange;
        self.xCRLB= np.zeros(zrangel);
        self.yCRLB= np.zeros(zrangel);
        self.zCRLB= np.zeros(zrangel);
        for u in range(zrangel):
            self.xCRLB[u]=(self.crlbX(0, 0, zrange[u], .001));
            self.yCRLB[u]=(self.crlbY(0, 0, zrange[u], .001));
            self.zCRLB[u] = (self.crlbZ(0, 0, zrange[u], .01));
            #self.x_abs.append(u);
            #IJ.log("crlb "+zCRLB[k]);
        self.xCRLB = np.array(self.xCRLB)
        self.yCRLB = np.array(self.yCRLB)
        self.zCRLB = np.array(self.zCRLB)

    def dump(self,path):
        '''cPickles oblect to the path'''
        cPickle.dump(self.__dict__,open(path,'wb'))

    def load(self,path):
        tmp_dict = cPickle.load(open(path,'rb'))
        self.__dict__.update(tmp_dict)

    def crlbZ(self,x, y, z,hdec):
        f2 = self.computeF(x, y, z+hdec);
        f1 = self.computeF(x, y, z);
        f0 = self.computeF(x, y, z-hdec);
        truc=(f2-f0)/(2*(hdec));
        res=np.sum((1/f1)*truc*truc);

        return np.sqrt(1/res);

    def crlbX(self,x, y, z,hdec):
        f2 = self.computeF(x+hdec, y, z);
        f1 = self.computeF(x, y, z);
        f0 = self.computeF(x-hdec, y, z);
        truc=(f2-f0)/(2*(hdec));
        res=np.sum((1/f1)*truc*truc);
        #res[res==0]=1e-5
        return np.sqrt(1/res);


    def crlbY(self,x, y, z,hdec):
        f2 = self.computeF(x, y+hdec, z);
        f1 = self.computeF(x, y, z);
        f0 = self.computeF(x, y-hdec, z);
        truc=(f2-f0)/(2*(hdec));
        res=np.sum((1/f1)*truc*truc);
        #res[res==0]=1e-5
        return np.sqrt(1/res);


    def computeF(self,x, y, z):
        f=(self.dataGenerator(x,y,z,self.num_phot,self.bg,self.size));
        #f=np.random.poisson(self.dataGenerator(x,y,z)).astype('float64');

        '''
        for i in range(self.taille):
            for ii in range(self.taille):
                f[i][ii]*=a;
                f[i][ii]+=b;
                if (f[i][ii]<0):
                    f[i][ii]=0.000000001;
        '''
        #f[f<=0] = 1e-10


        return f;

class ModelPhaseCRLB(object):

    def __init__(self,minZ, maxZ,zstep,num_phot,bg,my_phase,size=64,units='um'):
        self.phase = my_phase.ret_phase
        self.num_phot = num_phot
        self.bg = max([bg,1.])
        #print self.bg
        self.pupilmask = my_phase.ret_pupil
        self.sph = my_phase.parabola
        self.xslope = my_phase.xslope
        self.yslope = my_phase.yslope
        self.minZ=minZ;
        self.maxZ=maxZ;
        #self.phase=phase;
        self.taille=self.phase.shape[0];
        if units == 'um':
            self.zstep = zstep
        elif units == 'nm':
            self.zstep = zstep/1000.
        self.step = 1.
        self.dataGenerator = my_phase.gen_PSF
        self.size=size

    def runPhase(self):
        nb=(int)((self.maxZ-self.minZ)/self.zstep);
        zrange = np.arange(self.minZ,self.maxZ,self.zstep)
        zrangel = len(zrange)

        self.x_abs= zrange;
        self.CRLB= np.zeros((zrangel,5));

        for u in range(zrangel):
            aa = np.empty((5,self.size,self.size))

            aa[0] = self.crlbX(0, 0, zrange[u], .001);
            aa[1] = self.crlbY(0, 0, zrange[u], .001);
            aa[2] = self.crlbZ(0, 0, zrange[u], .01);
            aa[3] = self.crlbA(zrange[u], 10.);
            aa[4] = self.crlbB(zrange[u], .1);
            M = self.computeF(0, 0, zrange[u]);

            #aa = np.array([x,y,z,a,b])

            #print aa.shape
            bb = aa.reshape((len(aa),1,self.size,self.size))

            cc = np.sum(aa*bb/M,axis=(2,3))
            self.CRLB[u] = np.sqrt(np.linalg.inv(cc).diagonal())
            #self.CRLB[u] = 1./np.sqrt(cc.diagonal())
        self.xCRLB = self.CRLB[:,0]
        self.yCRLB = self.CRLB[:,1]
        self.zCRLB = self.CRLB[:,2]


    def dump(self,path):
        '''cPickles oblect to the path'''
        cPickle.dump(self.__dict__,open(path,'wb'))

    def load(self,path):
        tmp_dict = cPickle.load(open(path,'rb'))
        self.__dict__.update(tmp_dict)

    def crlbZ(self,x, y, z,hdec):
        f2 = self.computeF(x, y, z+hdec);
        #f1 = self.computeF(x, y, z);
        f0 = self.computeF(x, y, z-hdec);
        truc=(f2-f0)/(2*(hdec));
        return truc

    def crlbX(self,x, y, z,hdec):
        f2 = self.computeF(x+hdec, y, z);
        #f1 = self.computeF(x, y, z);
        f0 = self.computeF(x-hdec, y, z);
        truc=(f2-f0)/(2*(hdec));
        return truc


    def crlbY(self,x, y, z,hdec):
        f2 = self.computeF(x, y+hdec, z);
        f1 = self.computeF(x, y, z);
        f0 = self.computeF(x, y-hdec, z);
        truc=(f2-f0)/(2*(hdec));
        return truc


    def crlbA(self,z,hdec):
        f2 = self.computeAB(z,self.num_phot+hdec,self.bg);
        f1 = self.computeAB(z,self.num_phot,self.bg);
        f0 = self.computeAB(z,self.num_phot-hdec,self.bg);
        truc=(f2-f0)/(2*(hdec));
        return truc


    def crlbB(self,z,hdec):
        f2 = self.computeAB(z,self.num_phot,self.bg+hdec);
        f1 = self.computeAB(z,self.num_phot,self.bg);
        f0 = self.computeAB(z,self.num_phot,self.bg-hdec);
        truc=(f2-f0)/(2*(hdec));
        return truc


    def computeF(self,x, y, z):
        f=(self.dataGenerator(x,y,z,self.num_phot,self.bg,self.size));
        return f;

    def computeAB(self,z,a,b):
        f=(self.dataGenerator(0,0,z,a,b,self.size));
        return f;

    #def dataGenerator(self,x,y,z):
    #    return gen_PSF23(x,y,z,self.num_phot,self.bg,self.pupilmask,self.phase,self.sph,self.xslope,self.yslope)




def corr_coeff(img1,img2):
    return np.corrcoef(np.ravel(img1),np.ravel(img2))[1,0]


def corr_curve(img1,img2):
    '''
    correlation coefficient layer by layer for two z-stacks
    '''

    if img1.shape != img2.shape:
        print('Different shapes!', img1.shape,img2.shape)
        pass
    else:
        h,l,w=img1.shape
        cc1=img1.reshape(h,l*w)
        cc2=img2.reshape(h,l*w)
        cc = np.corrcoef(cc1,cc2)
        return cc[np.arange(h),np.arange(h)+h]


#def gen_PSF(x,y,z):
#    #zind = np.int(30+z*10)
#    x,y,z = np.array(x), np.array(y), np.array(z)
#    #print x.shape
#    #print z.mean()
#    zind = np.array(my_interp(z.mean()),dtype='int')
#
#    if z.shape:
#        z = np.array(z).reshape(np.fft.fftshift(z).shape[0],1,1)
#    if x.shape:
#        x = np.array(x).reshape(x.shape[0],1,1)
#    if y.shape:
#        y = np.array(y).reshape(y.shape[0],1,1)
#    #print zind
#    #print np.array(phaseFS)[zind]
#    #delta_z = -(z-sample.zvector[phase_z_vector[zind]])
#    #print delta_z
#    ampl,phase = (decomp2(np.abs(pupil_init),phaseFS[zind]-parabola*z-xslope*x-yslope*y))
#    if ampl.shape[0]==3:
#        ampl  = ampl[:,16:48,16:48]
#    else:
#        ampl = ampl[16:48,16:48]
#    return ampl


def gen_PSF1(x,y,z,phase_ind,a,b,pupil_init1, phaseFS,parabola, xslope, yslope ):
    #print phase_ind
    ampl,phase = (decomp2(np.abs(pupil_init1),phaseFS[phase_ind]-parabola*z-xslope*x-yslope*y))

    ampl = ampl[16:48,16:48]
    ampl /= ampl.sum()
    #ampl -=ampl.mean()
    #ampl[ampl<0] = 0
    ampl = a*ampl+b+.01
    ampl[ampl<=0] = 1e-5
    return ampl

def gen_PSF2(x,y,z,phase_ind,a,b,pupil_init1, phaseFS,parabola, xslope, yslope ):

    ampl,phase = (decomp2(np.abs(pupil_init1),phaseFS[phase_ind]-parabola*z-xslope*x-yslope*y))
    #print phase_ind
    #ampl = ampl[16:48,16:48]**2
    ampl = ampl**2
    ampl /= ampl.sum()
    #ampl -=ampl.mean()
    #ampl[ampl<0] = 0
    ampl = a*ampl+b+.01
    ampl[ampl<=0] = 1e-5
    return ampl

def gen_PSF22(x,y,z,a,b,pupil_init, phase,parabola, xslope, yslope ):

    ampl,phase = (decomp1(np.abs(pupil_init),phase-parabola*z-xslope*x-yslope*y))
    ampl = ampl**2
    ampl /= ampl.sum()
    ampl = a*ampl+b+.01
    return ampl



def norm(tmp):
    '''Standart deviation normalization'''
    tmp-=np.mean(tmp)
    tmp/=np.std(tmp)
    return tmp

def norm01(tmp):
    '''0-1 normalization'''
    tmp=np.array(tmp,dtype='float32')
    if len(tmp.shape)==3:
        tmp1=tmp - tmp.min(axis=(-1,-2)).reshape(len(tmp),1,1)
        tmp1/=np.max(tmp,axis=(-1,-2)).reshape(len(tmp),1,1)
    elif len(tmp.shape)==2:
        tmp1 = tmp - tmp.min()
        tmp1/=np.max(tmp)
    return tmp1

def norm_sum(tmp):
    '''1-photon normalization'''
    tmp=np.array(tmp,dtype='float32')
    if len(tmp.shape)==3:
        tmp1=tmp - tmp.min(axis=(-1,-2)).reshape(len(tmp),1,1)
        tmp1/=np.sum(tmp1,axis=(-1,-2)).reshape(len(tmp),1,1)
    elif len(tmp.shape)==2:
        tmp1 = tmp - tmp.min()
        tmp1/=np.sum(tmp1)
    return tmp1


#norm_sum=lambda I: norm01(I)/np.sum(norm01(I))

def cropCenter(stack,size):
    stack = np.array(stack)
    w,h = stack.shape[-1],stack.shape[-2]
    c1,c2 = w/2,h/2
    #print w,h,c1,c2
    if size>w:
        print('wrong size')
        return 1
    elif len(stack.shape) == 2:
        #print '2D'
        return stack[c1-size/2:c1+size/2,c2-size/2:c2+size/2]
    elif len(stack.shape) == 3:
        #print '3D'
        return stack[:,c1-size/2:c1+size/2,c2-size/2:c2+size/2]

class r_phase(object):
    '''
    Object containing pahse retrieve information for firther use in PSF generation
    '''
    def __init__(self, pupil=0,
                 parabola=0,
                 ret_pupil=0,
                 ret_phase=0,
                 xslope=0,
                 yslope=0):
        #self.mask = mask
        self.pupil = pupil
        self.parabola = parabola
        self.ret_pupil = ret_pupil*(pupil>0)
        self.ret_phase = ret_phase
        self.xslope = xslope
        self.yslope = yslope
        self.size = 32
        self.gaussSmooth = 0


    def dump(self,path):
        '''cPickles oblect to the path'''
        cPickle.dump(self.__dict__,open(path,'wb'))

    def load(self,path):
        cPickle.load(open(path,'rb'))
        self.__dict__.update(tmp_dict)

    def gen_PSF(self, x,y,z,a,b):
        #print 'z',z
        #print (self.ret_phase-self.parabola*z-self.xslope*x-self.yslope*y).shape
        ampl,phase = (decomp1f(np.abs(self.ret_pupil),self.ret_phase-self.parabola*z-self.xslope*x-self.yslope*y))
        if self.gaussSmooth:
            ampl = ndi.gaussian_filter(ampl,self.gaussSmooth)
        return b+a*norm_sum(cropCenter(ampl,self.size)**2)


    def genPupilPSF(self, x,y,z,a,b):
        #print 'z',z
        #print (self.ret_phase-self.parabola*z-self.xslope*x-self.yslope*y).shape
        ampl,phase = (decomp1f(np.abs(self.pupil),self.ret_phase-self.parabola*z-self.xslope*x-self.yslope*y))
        return b+a*norm_sum(cropCenter(ampl,self.size)**2)

    def gen_crop(self,SingleCrop):
        i = SingleCrop
        return self.gen_PSF(i.x,i.y,i.z,i.a,i.b,len(i.I))

    def gen_crop_init(self,SingleCrop):
        i = SingleCrop
        return self.gen_PSF(0,0,i.Z,i.A,i.B,len(i.I))

class SingleCrop(object):
    '''Object containing image I with preliminary coordinates A,B,X,Y,Z
    and localization results a,b,x,y,z,
    where a and b are number of photons and level of bg.
    The latter parameters will be estimated automatically after adding the image.
    '''
    def __init__(self,I,X=0,Y=0,Z=0):
        self.I = I #intensity pattern
        self.X = X #crop x location
        self.Y = Y #crop y location
        self.Z = Z #crop z predetection
        self.x = 0
        self.y = 0
        self.z = Z
        #self.b = 50
        #self.a = 3500

        #np.min(ndi.gaussian_filter(I,2))+1
        s=len(I)/10
        tmp = np.ravel([I[0:s+1],I[-2-s:-1],I[:,0:s+1].T,I[:,-s-2:-1].T])
        try:
            bgMean =  np.mean((tmp))
        except ValueError:
            print('I.shape',I.shape)
            bgMean = I.mean()
        #bgMed = np.median(tmp)
        a = np.sum(I-bgMean)
        if a<=0:
            a=0
        self.a = a
        self.b = bgMean
        self.A = self.a
        self.B = self.b

def cropCenter(stack,size):
    stack = np.array(stack)
    w,h = stack.shape[-1],stack.shape[-2]
    c1,c2 = w/2,h/2
    #print w,h,c1,c2
    if size>w:
        print('wrong size')
        return 1
    elif len(stack.shape) == 2:
        #print '2D'
        return stack[c1-size/2:c1+size/2,c2-size/2:c2+size/2]
    elif len(stack.shape) == 3:
        #print '3D'
        return stack[:,c1-size/2:c1+size/2,c2-size/2:c2+size/2]

# error functions:
LE = lambda I,F: np.mean(-np.log(F)*I + F,dtype = np.float64)
SEF = lambda I,F: np.sum((I - F)**2)


class MLE(object):
    '''
    Class for localization using maximum likelihood estimator using retrieved phase (r_phase object)
    and a crop (SingleCrop object)
    '''
    def __init__(self,ERF,my_phase,my_crop,gamma=.2):
        self.gen_PSF = my_phase.gen_PSF
        self.crop = my_crop
        self.ERF = ERF
        self.crop.I[self.crop.I<=0] = 1e-5
        my_phase.size = len(my_crop.I)
        self.size = len(my_crop.I)
        self.gamma = gamma
        self.run()
        #return self.crop

    def calc_ERF(self):
        crop = self.crop
        gen = self.gen_PSF
        ERF = self.ERF
        return ERF(crop.I,gen(crop.x,crop.y,crop.z,crop.a,crop.b,self.size))

    def derivZ(self,dz):
        crop = self.crop
        gen = self.gen_PSF
        LF = self.ERF
        I = crop.I
        Fz = []
        x,y,z,a,b = crop.x,crop.y,crop.z,crop.a,crop.b
        Fz.append(gen(x,y,z-dz,a,b,self.size))
        Fz.append(gen(x,y,z,a,b,self.size))
        Fz.append(gen(x,y,z+dz,a,b,self.size))

        LF1,LF2,LF3 =LF(I,Fz[0]),LF(I,Fz[1]),LF(I,Fz[2])
        self.der_z = -(LF1 - LF3)/2/dz
        self.der_z2 = np.abs(LF1 -2*LF2+ LF3)/dz**2
        self.LF2 = LF2


    def derivA(self,da):
        crop = self.crop
        gen = self.gen_PSF
        LF = self.ERF
        I = crop.I
        Fz = []
        x,y,z,a,b = crop.x,crop.y,crop.z,crop.a,crop.b
        Fz.append(gen(x,y,z,a-da,b,self.size))
        Fz.append(gen(x,y,z,a,b,self.size))
        Fz.append(gen(x,y,z,a+da,b,self.size))

        LF1,LF2,LF3 =LF(I,Fz[0]),LF(I,Fz[1]),LF(I,Fz[2])
        self.der_a = -(LF1 - LF3)/2/da
        self.der_a2 = np.abs(LF1 -2*LF2+ LF3)/da**2
        self.LF2 = LF2
        if np.isnan(LF1) or np.isnan(LF2) or np.isnan(LF3):
            raise ValueError('Log is none, a {}, da {}'.format(a,da))
            sys.exit(1)


    def derivB(self,db):
        crop = self.crop
        gen = self.gen_PSF
        LF = self.ERF
        I = crop.I
        Fz = []
        x,y,z,a,b = crop.x,crop.y,crop.z,crop.a,crop.b
        Fz.append(gen(x,y,z,a,b-db,self.size))
        Fz.append(gen(x,y,z,a,b,self.size))
        Fz.append(gen(x,y,z,a,b+db,self.size))

        LF1,LF2,LF3 =LF(I,Fz[0]),LF(I,Fz[1]),LF(I,Fz[2])
        self.der_b = -(LF1 - LF3)/2/db
        self.der_b2 = np.abs(LF1 -2*LF2+ LF3)/db**2
        self.LF2 = LF2




    def derivX(self,dx):
        crop = self.crop
        gen = self.gen_PSF
        LF = self.ERF
        I = crop.I
        Fz = []
        x,y,z,a,b = crop.x,crop.y,crop.z,crop.a,crop.b
        Fz.append(gen(x-dx,y,z,a,b,self.size))
        Fz.append(gen(x,y,z,a,b,self.size))
        Fz.append(gen(x+dx,y,z,a,b,self.size))

        LF1,LF2,LF3 =LF(I,Fz[0]),LF(I,Fz[1]),LF(I,Fz[2])
        self.der_x = -(LF1 - LF3)/2/dx
        self.der_x2 = np.abs(LF1 -2*LF2+ LF3)/dx**2
        self.LF2 = LF2




    def derivY(self,dy):
        crop = self.crop
        gen = self.gen_PSF
        LF = self.ERF
        I = crop.I
        Fz = []
        x,y,z,a,b = crop.x,crop.y,crop.z,crop.a,crop.b
        Fz.append(gen(x,y-dy,z,a,b,self.size))
        Fz.append(gen(x,y,z,a,b,self.size))
        Fz.append(gen(x,y+dy,z,a,b,self.size))

        LF1,LF2,LF3 =LF(I,Fz[0]),LF(I,Fz[1]),LF(I,Fz[2])
        self.der_y = -(LF1 - LF3)/2/dy
        self.der_y2 = np.abs(LF1 -2*LF2+ LF3)/dy**2
        self.LF2 = LF2


    def run(self,iterations = 50,dx=.01,dy=.01,dz=.02,da=10.,db=.1,stop = 1e-4):
        crop = self.crop
        z = crop.z
        conv=0
        self.zh = []
        self.ah = []
        self.bh = []
        self.xh = []
        self.yh = []
        self.LL = []
        self.gh = []
        self.zh.append(crop.z)
        self.ah.append(crop.a)
        self.bh.append(crop.b)
        gamma = self.gamma
        #zh = np.zeros((len(sample.zvector),iterations))
        for iter in range(iterations):
            #print 'iter', iter


            self.derivA(da)
            gammatmp=gamma*5

            oldA=crop.a
            oldL= self.LF2#self.calc_ERF()
            for hj in range(4):

                #print 'subiter', hj,
                #print 'der' ,self.der_a,'der2',self.der_a2
                if self.der_a2!=0:
                    #print 'updating crop.z'
                    atmp = crop.a-gammatmp*self.der_a/self.der_a2
                    if atmp >0:
                        crop.a = atmp
                newLL = self.calc_ERF()
                if newLL<=oldL:
                    #print 'breaking'
                    self.gh.append(gammatmp)
                    break
                else:
                    #print 'restoring z'
                    crop.a=oldA
                    gammatmp/=10

            self.derivZ(dz)
            gammatmp=gamma
            oldZ=crop.z
            oldL= self.LF2#self.calc_ERF()
            for hj in range(4):

                #print 'subiter', hj,
                #print 'gamma',gammatmp,self.der_z/self.der_z2
                if self.der_z2!=0:
                    #print 'updating crop.z'
                    ztmp= crop.z-gammatmp*self.der_z/self.der_z2

                    if abs(ztmp)<= 3:
                        crop.z= ztmp


                newLL = self.calc_ERF()
                if newLL<=oldL:
                    #print 'breaking'
                    self.gh.append(gammatmp)
                    break
                else:
                    #print 'restoring z'
                    crop.z=oldZ
                    gammatmp/=10



            self.derivB(db)
            gammatmp=gamma
            oldB=crop.b
            #print 'oldB',oldB
            oldL= self.LF2#self.calc_ERF()
            for hj in range(4):

                #print 'subiter', hj,
                #print 'derB',self.der_b,self.der_b2
                if self.der_b2!=0:
                    #print 'updating crop.z'
                    crop.b= crop.b-gammatmp*self.der_b/self.der_b2
                    if crop.b<=0: crop.b = 1
                    #print 'newb ',crop.b
                newLL = self.calc_ERF()
                if newLL<=oldL:
                    #print 'breaking'
                    self.gh.append(gammatmp)
                    break
                else:
                    #print 'restoring z'
                    crop.b=oldB
                    gammatmp/=10


            self.derivX(dx)
            gammatmp=gamma
            oldX=crop.x
            oldL= self.LF2#self.calc_ERF()
            for hj in range(4):

                #print 'subiter', hj,
                #print 'gamma',gammatmp,self.der_z/self.der_z2
                if self.der_x2!=0:
                    #print 'updating crop.z'
                    xtmp = crop.x-gammatmp*self.der_x/self.der_x2
                    if abs(xtmp)<3:
                        crop.x = xtmp
                newLL = self.calc_ERF()
                if newLL<=oldL:
                    #print 'breaking'
                    self.gh.append(gammatmp)
                    break
                else:
                    #print 'restoring z'
                    crop.x=oldX
                    gammatmp/=10


            self.derivY(dy)
            gammatmp=gamma
            oldY=crop.y
            oldL= self.LF2#self.calc_ERF()
            for hj in range(4):

                #print 'subiter', hj,
                #print 'gamma',gammatmp,self.der_z/self.der_z2
                if self.der_y2!=0:
                    #print 'updating crop.z'
                    ytmp = crop.y-gammatmp*self.der_y/self.der_y2
                    if abs(ytmp)<3:
                        crop.y = ytmp
                newLL = self.calc_ERF()
                if newLL<=oldL:
                    #print 'breaking'
                    self.gh.append(gammatmp)
                    break
                else:
                    #print 'restoring z'
                    crop.y=oldY
                    gammatmp/=10

            self.zh.append(crop.z)
            self.ah.append(crop.a)
            self.bh.append(crop.b)
            self.xh.append(crop.x)
            self.yh.append(crop.y)
            self.LL.append(newLL)
            crit = np.abs(self.LL[iter]-self.LL[iter-1])
            #print 'crit',crit
            if iter>4 and crit<stop:
                #zh = zh[:iter]
                #if debug:
                #print 'converged ',
                crop.conv=1
                break
        self.crop.zh = self.zh
        self.crop.ah = self.ah
        self.crop.bh = self.bh
        self.crop.xh = self.xh
        self.crop.yh = self.yh
        self.crop.LL = self.LL
        #self.crop.I = 0

class XcorrMLE:
    '''
    Class making xcorr predetection and MLE localization
    # template 3D stack of PSF for xcorr predetection
    # vector - z coordinates of the vector
    # MLE_phase - MLE class instance
    # img_thr - bg subtraction before xcorr
    # img = img - img_thr*std(img)
    # padding (bool) for xcorr
    # bg_kernel - smoothing kernel for bg subtraction before xcorr
    # min_distance - for peak detection density
    # x_thr is a cutoff for xcorr max projection (0-1),default = .2
    '''
    def __init__(self,
                psfArray,
                zVect,
                MLE_phase,
                padding=True,
                bg_kernel = 5,
                img_thr = 1,
                min_distance = 10,
                crop_size = 32,
                min_photons  = 1000,
                x_thr=.2,
                MLE=MLE,
                MLE_ERF = LE,
                MLE_gamma = 1.,
                pool = 0,
                **kwargs
                ):
        self.template = psfArray
        self.zvector = zVect
        self.myPhase = MLE_phase

        self.padding = padding
        self.pad_size = len(self.template)/2
        self.bg_kernel=bg_kernel
        self.img_thr = img_thr
        self.x_thr = x_thr
        self.min_distance = min_distance
        self.cropSize = crop_size
        self.min_photons = min_photons


        self.MLE = MLE
        self.ERF = MLE_ERF
        self.MLE_gamma = MLE_gamma
        self.pool = pool
        self.debug = 0

    def run(self,frame):
        '''
        thr - minimal xcorr amplitude (0-1)

        '''
        self.xcorr(frame)
        self.detMax()
        self.makeCrops()

        if any(self.crops):
            if self.pool:
                l = threading.Lock()
                thrList = []
                for crop in self.crops:
                    thrList.append(threading.Thread(target=MLE,args=(self.ERF,self.myPhase,crop,self.MLE_gamma)))
                for t in thrList:
                    t.start()
                for t in thrList:
                    t.join()
            else:
                for crop in self.crops:
                    MLE(self.ERF,self.myPhase,crop,self.MLE_gamma)
        return self.crops


    def xcorr(self,frame):
        img00=np.array(frame, dtype='float32')
        if self.padding:
            self.img = np.pad(np.array(frame),(self.pad_size,),'mean')
            self.pad_shift = (self.pad_size,self.pad_size)
        else:
            self.img = np.array(frame)
            self.pad_shift = [0,0]

        img0=img00-ndi.filters.gaussian_filter(img00,self.bg_kernel)
        img0=img0-self.img_thr*img0.std()
        img0[img0<0]=0
        self.img_wo_bg = img0


        out = np.array([match_template(self.img_wo_bg,tmp,pad_input=self.padding,mode='mean') for tmp in self.template])
        #out = np.array([match_template(img00,tmp,pad_input=self.padding,mode='mean') for tmp in self.template])
        self.Xout= out
        self.shift = .5*(np.array((img00).shape)-np.array((out[0]).shape))
        #return out

    def detMax(self):
        zc = self.Xout
        xymaxproj = zc.max(axis=0) - zc.mean()
        xymaxproj -= self.x_thr
        xymaxproj[xymaxproj<0]=0
        self.xymaxproj = xymaxproj
        self.Xcoordinates = np.array(peak_local_max(xymaxproj, min_distance=self.min_distance))
        if len(self.Xcoordinates):
            self.Icoordinates = self.Xcoordinates + self.shift + self.pad_shift - [1,1]
            self.zcorrs = self.Xout[:,self.Xcoordinates[:,0],self.Xcoordinates[:,1]]
            self.zinits = self.zvector[self.zcorrs.argmax(axis=0)]
        else:
            self.Icoordinates = []

    def showXcorr(self):
        xc=self
        plt.imshow(xc.xymaxproj)
        plt.scatter(xc.Xcoordinates[:,1],xc.Xcoordinates[:,0],marker = '.',color = 'r',)

    def showDetections(self):
        xc=self
        plt.imshow(xc.img,interpolation='none')
        plt.plot(xc.Icoordinates_selected[:,1],xc.Icoordinates_selected[:,0],'r.')


    def makeCrops(self):
        if len(self.Icoordinates):
            x,y = self.Icoordinates[:,1],self.Icoordinates[:,0]
            Icoordinates_selected = []
            s = self.cropSize/2
            self.crops=[]
            for x1,y1,z in zip(x,y,self.zinits):
                I=self.img[int(y1-s+1):int(y1+s+1),int(x1-s+1):int(x1+s+1)]
                if self.debug: print('I.shape',I.shape)
                if I.shape == (self.cropSize,self.cropSize):
                    tmp = SingleCrop(I,X=x1 -self.shift[0] - self.pad_shift[0],Y=y1 -self.shift[1] - self.pad_shift[1],Z=z)
                    if self.debug: print('xc makecrops: shape is good')
                    if tmp.A>self.min_photons:

                        if self.debug: print('xc makecrops: photons is good')
                        self.crops.append(tmp)
                        Icoordinates_selected.append([y1,x1])
                    else:

                        if self.debug: print('photons found',tmp.A)
            self.Icoordinates_selected = np.array(Icoordinates_selected)
        else:
            self.crops = []

### Parallel tools


import multiprocessing
import time
from multiprocessing import cpu_count


class ActivePool(object):
    def __init__(self):
        #super(ActivePool, self).__init__()
        self.mgr = multiprocessing.Manager()
        self.arr = self.mgr.list()
        self.active = self.mgr.list()
        self.lock = multiprocessing.Lock()
        self.exitCodes = self.mgr.list()
        self.jobs = self.mgr.list()

    def makeActive(self, name):
        with self.lock:
            self.active.append(name)
    def makeInactive(self, name):
        with self.lock:
            self.active.remove(name)
    def __str__(self):
        with self.lock:
            self.lenArr = len(self.arr)
            return str(self.active)

def worker(fun,s, pool,fOpen,i,px):
    name = multiprocessing.current_process().name
    with s:
        #pool.jobs.pop()
        #pool.makeActive(name)
        #print 'Now running: %s' % str(pool)
        out = fun(fOpen(i))
        for o in out:
            pool.arr.append([int(i),-o.x+o.X*px,-o.y+o.Y*px,o.z, o.a, o.b])
        #pool.makeInactive(name)
        with pool.lock:
            pool.exitCodes.append(multiprocessing.current_process().exitcode)
            print('\r{} frames, {} particles'.format(len(pool.exitCodes),len(pool.arr)))

def gen_PSF23(x,y,z,a,b,pupil_init, phase,parabola, xslope, yslope ):

    ampl,phase = (decomp1(np.abs(pupil_init),phase-parabola*z-xslope*x-yslope*y))

    return b+a*norm_sum(ampl**2)

#LF=lambda I,F: np.sum(-np.log((F**2)**(I)/scipy.misc.factorial(I)*np.exp(-F**2)))
#LF=lambda I,F: np.sum(-np.log((F**I/scipy.misc.factorial(I)*np.exp(-F))))
#np.math.factorial()
#def derivZ(I,x0,y0,z0,phase_ind,dz,a,b):

def calc_ERF(ERF,phase_ind):
    return ERF(I,gen_PSF1(x,y,z,phase_ind,a,b))

def derivZ(dz,LF,phase_ind):
    #global x,y,z
    global pupil_init, phaseFS, parabola, xslope, yslope
    Fz = []
    Fz.append(gen_PSF1(x,y,z-dz,phase_ind,a,b,pupil_init, phaseFS, parabola, xslope, yslope))
    Fz.append(gen_PSF1(x,y,z,phase_ind,a,b,pupil_init, phaseFS, parabola, xslope, yslope))
    Fz.append(gen_PSF1(x,y,z+dz,phase_ind,a,b,pupil_init, phaseFS, parabola, xslope, yslope))

    LF1,LF2,LF3 =LF(I,Fz[0]),LF(I,Fz[1]),LF(I,Fz[2])
    der_z = -(LF1 - LF3)/2/dz
    der_z2 = (LF1 -2*LF2+ LF3)/dz**2
    return der_z,der_z2,LF2

#def derivA(I,x0,y0,z0,zind,a,da,b):
def derivA(da,LF,phase_ind):
    Fa=[]
    Fa.append(gen_PSF1(x,y,z,phase_ind,a-da,b,pupil_init, phaseFS, parabola, xslope, yslope))
    Fa.append(gen_PSF1(x,y,z,phase_ind,a,b,pupil_init, phaseFS, parabola, xslope, yslope))
    Fa.append(gen_PSF1(x,y,z,phase_ind,a+da,b,pupil_init, phaseFS, parabola, xslope, yslope))
    #plt.imshow(Fa[0])
    #plt.show()

    LF1,LF2,LF3 =LF(I,Fa[0]),LF(I,Fa[1]),LF(I,Fa[2])
    #print LF1
    der1 = -(LF1 - LF3)/2/da
    der2 = (LF1 -2*LF2+ LF3)/da**2
    return der1,der2,LF2
def derivB(db,LF,phase_ind):
    Fa=[]
    #print phase_ind
    Fa.append(gen_PSF1(x,y,z,phase_ind,a,b-db,pupil_init, phaseFS, parabola, xslope, yslope))
    Fa.append(gen_PSF1(x,y,z,phase_ind,a,b,pupil_init, phaseFS, parabola, xslope, yslope))
    Fa.append(gen_PSF1(x,y,z,phase_ind,a,b+db,pupil_init, phaseFS, parabola, xslope, yslope))
    #plt.imshow(Fa[0])
    #plt.show()

    LF1,LF2,LF3 =LF(I,Fa[0]),LF(I,Fa[1]),LF(I,Fa[2])
    #print LF1
    der1 = -(LF1 - LF3)/2/db
    der2 = (LF1 -2*LF2+ LF3)/db**2
    return der1,der2,LF2

def SP_MLE(img,my_interp,z0=0.,dz=.05,A=1,da=10,db = 1e-5,gamma = .05,iterations=500,stop=1e-4,ERF=LE,refine = 0,debug=1):
    global x,y,z,a,b,I,phase_ind

    phase_ind=np.array(my_interp(z0),dtype='int')

    if debug: print('using phase index ', phase_ind)
    if not refine:
        b=img.mean(axis=1).min()
        a=img.sum()-b*img.shape[0]*img.shape[1]
    if debug: print('a, b, z0',a,b, z0)
    x,y,z,I = 0,0,z0,img
    #a=1
    #b=0
    #da=0.1
    #db=0.1
    conv=0
    zh = np.zeros(iterations)
    ah = np.zeros(iterations)
    bh = np.zeros(iterations)
    LL = np.zeros(iterations)
    #zh = np.zeros((len(sample.zvector),iterations))
    for iter in range(iterations):


        derZ = derivZ(dz,ERF,phase_ind)
        gammatmp=gamma
        oldZ=z
        oldL=calc_ERF(ERF,phase_ind)
        for hj in range(4):
            if derZ[1]!=0:
                z= z-gammatmp*derZ[0]/(derZ[1])
            if calc_ERF(ERF,phase_ind)<=oldL:
                break
            else:
                z=oldZ
                gammatmp/=10


        derA = derivA(da,ERF,phase_ind)
        gammatmp=gamma
        oldA=a
        oldL=calc_ERF(ERF,phase_ind)
        for hj in range(4):
            if derA[1] != 0:
                a= a-gammatmp*derA[0]/(derA[1])
            if calc_ERF(ERF,phase_ind)<=oldL:
                break
            else:
                a=oldA
                gammatmp/=10

        derB = derivB(db,ERF,phase_ind)
        gammatmp=gamma
        oldB=b
        oldL=calc_ERF(ERF,phase_ind)
        for hj in range(4):
            if derB[1] != 0:
                b= b-gammatmp*derB[0]/(derB[1])
            if calc_ERF(ERF,phase_ind)<=oldL:
                break
            else:
                b=oldB
                gammatmp/=10


        derB = derivB(db,ERF,phase_ind)
        if derB[1] != 0:
            b = b-gamma*derB[0]*1e-0/(derB[1])
        else:
            b=b
        if debug:
            print('\r',iter)
        #print derZ[0],z,derZ[2]
        #print derZ[0],derA[0],derB[0],z,a,b,derA[2]
        zh[iter] = z
        ah[iter] = a
        bh[iter] = b
        LL[iter] = derZ[2]
        if math.isnan(a):
            print('Error NaN')
            break
        if iter>10 and np.abs(LL[iter]-LL[iter-1])<stop:
            #zh = zh[:iter]
            if debug: print('converged ')
            conv=1
            break
    if debug: print('z=%.3f (err = %.3f), ampl=%.2f, bg=%.3f'%(z,(z-z0),a,b))
    return z, zh, ah, bh, LL[np.nonzero(LL)]

def search_z(img,z0,debug=0):
    z, zh, ah,bh, LL = SP_MLE(img,z0=z0,dz=.01,A=1,da=0.01,db=1e-3,gamma = 1., iterations=1000,stop=1e-4,ERF=LE,debug=debug)
    count=0
    #while np.abs(z-z0)>.1 and count<4:
    #    z0=z
    #    count+=1
    #    if debug:
    #        print 'z err ',z-sample.zvector[zind]
    #        print 'running refinement step'
    #    z, zh, ah,bh, LL = SP_MLE(img,z0=z,dz=.01,A=1,da=0.01,db=1e-3,gamma = 1., iterations=1000,stop=1e-4,ERF=SEF,refine=1,debug=debug)
    return z,LL[-1]

def likelihood_matrix(stack,ERF=LE):
    z=stack.shape[0]
    cc=np.zeros((z,z))
    ll=np.zeros((z,z))
    for i in range(z):
        for j in range(z):
            cc[i,j]=np.corrcoef(np.ravel(stack[i]),np.ravel(stack[j]))[0,1]
            ll[i,j] = ERF(stack[i]+.01,stack[j]+.01)
            #cc=np.corrcoef(np.ravel(stack[i]),np.ravel(stack[j]))
            #print i,j,cc
    return cc,ll

def likelihood_matrix1(stack,zvector,my_interp,pupil_init1, phaseFS,parabola, xslope, yslope ,ERF=LE):
    z=stack.shape[0]
    #cc=np.zeros((z,z))
    ll=np.zeros((z,z))
    for i in np.arange(z):
        st=  stack[i][16:48,16:48]/stack[i][16:48,16:48].sum()
        for j in np.arange(z):
            #cc[i,j]=np.corrcoef(np.ravel(stack[i]),np.ravel(stack[j]))[0,1]
            #print j, sample.zvector[j], my_interp(sample.zvector[j])
            psf = gen_PSF1(0,0,zvector[j],int(my_interp(zvector[np.maximum(i,0)])),1,0,pupil_init1, phaseFS,parabola, xslope, yslope )
            #print psf.sum(),stack[i][16:48,16:48].sum()
            #plt.imshow(psf)
            #plt.show()
            #plt.imshow(stack[i][16:48,16:48])
            #plt.show()
            ll[i,j] = LE(st+.01,psf/psf.sum()+.01)
            print('\r',i,)
            #cc=np.corrcoef(np.ravel(stack[i]),np.ravel(stack[j]))
            #print i,j,cc
    return ll

def saveVISP3D(table_fxyzib,path,selection=None,convert_um2nm = True):
    mul=1
    if convert_um2nm:
        mul=1000
    table_xyzif = np.zeros((len(table_fxyzib),5))
    table_xyzif[:,0:3] = table_fxyzib[:,1:4]*mul #xyz
    table_xyzif[:,3] = table_fxyzib[:,4] #intensity
    table_xyzif[:,4] = table_fxyzib[:,0] #frame num
    if selection:
        np.savetxt(path,table_xyzif[selection])
    else:
        np.savetxt(path,table_xyzif[:])

def saveTS(table_fxyzib,path):
    np.savetxt(path,table_fxyzib,delimiter=',',
           comments='',header='frame,x[nm],y[nm],z[nm],intensity[photons],background[photons]')


class BeadsStats(object):
    def __init__(self,data,data_structure = 'fxyzib'):
        self.data = data
        self.data_structure=data_structure


    def doSegment(self,bandwidth,subset='xy'):
        from sklearn.cluster import MeanShift
        pos = np.array([self.data_structure.index(x) for x in subset],'int')
        #print pos
        subdata= self.data[:,pos]
        self.subdata = subdata
        try:
            ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
            ms.fit(subdata)
            self.labels = ms.labels_
            self.cluster_centers = ms.cluster_centers_

            self.labels_unique = np.unique(self.labels)
            self.n_clusters_ = len(self.labels_unique)

            print("number of estimated clusters : %d" % self.n_clusters_)
            print('Used datset with the shape',subdata.shape)
            print('found',self.labels.shape,'groups')
            print('number of groups', self.labels_unique.shape)
        except Exception as e:
            traceback.print_exc()
            print(e)

    def plotSegment(self):
        try:
            self.n_clusters_
        except:
            print('no clustering done. Try .doSegment(1,\'xy\')')

        from itertools import cycle

        plt.figure(1)
        plt.clf()
        X = self.subdata
        colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
        for k, col in zip(range(self.n_clusters_), colors):
            my_members = self.labels == k
            cluster_center = self.cluster_centers[k]
            plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
            plt.plot(cluster_center[0], cluster_center[1], '+', markerfacecolor=col,
                     markeredgecolor='k', markersize=2)
        plt.title('Estimated number of clusters: %d' % self.n_clusters_)
        plt.xlabel('x, um')
        plt.ylabel('y, um')
        plt.show()

    def plotGroupXY(self,group):
        subdata = self.data[self.labels==group]
        x = self.data_structure.index('x')
        y = self.data_structure.index('y')

        plt.plot(subdata[:,x],subdata[:,y],'k.')
        plt.xlabel('x, um')
        plt.ylabel('y, um')
        plt.title('group %i'%group)
        plt.show()

    def doStats(self,z_step,n_frames,phase, background=0, group=None, units = 'um'):
        '''
        dataset should include localizations of beads fixed at the same z position for n_frames, then moved for z_step
        '''

        ### preselect localization group (one bead)
        if not np.isnan(group):
            subdata = self.data[self.labels==group]
        else:
            subdata = self.data[:]

        #recall positions of data

        f = self.data_structure.index('f')
        z = self.data_structure.index('z')
        a = self.data_structure.index('i')
        x = self.data_structure.index('x')
        y = self.data_structure.index('y')

        t = np.floor(subdata[:,f]/n_frames)
        #print t

        try:
            b = self.data_structure.index('b')
        except:
            b = None

        ### make stats for every step in z
        found_sorted=[]
        for i in np.unique(t):
            #print i
            found_sorted.append(subdata[t==i])
        print('found %i steps'%len(found_sorted))
        #found_sorted = np.array(found_sorted)
        found_std = np.array([tmp[:].std(axis=0) for tmp in found_sorted])
        found_mean = np.array([tmp[:].mean(axis=0) for tmp in found_sorted])
        print('found %i stats'%len(found_std))
        #plt.plot(tmp[np.abs(tmp-tmp.mean())<tmp.std()*3])
        #plt.show()

        ### compute CRLB
        num_phot = subdata[:,a].mean()
        if b:
            bg = subdata[:,b].mean()
        else:
            bg = background
        crlb = ModelPhaseCRLB2(-2,2,abs(z_step),num_phot,bg,phase,units=units)
        crlb.runPhase()

        print('t shift',np.median(t))
        plt.figure(figsize=(6,3))
        plt.plot(subdata[:,f],subdata[:,z],'.')
        plt.plot(subdata[:,f],(t-np.median(t))*z_step,'r-')
        #plt.plot(found[:,2],'+')
        plt.grid()
        plt.xlabel('frame number')
        plt.ylabel('z, um')
        plt.title('z(um)')
        plt.legend(['localizations','ground truth'])


        plt.figure(figsize=(10,3))

        plt.subplot(1,3,1)
        plt.plot(subdata[:,f],subdata[:,x],'b.')
        #plt.plot(found[:,2],'+')
        plt.grid()
        plt.xlabel('frame number')
        plt.ylabel('x, um')
        plt.title('x drift (um)')

        plt.subplot(1,3,2)
        plt.plot(subdata[:,f],subdata[:,y],'g.')
        #plt.plot(found[:,2],'+')
        plt.grid()
        plt.xlabel('frame number')
        plt.ylabel('y, um')
        plt.title('y drift (um)')

        plt.subplot(1,3,3)
        plt.plot(subdata[:,f],subdata[:,z]-((t-np.median(t))*z_step),'r.')
        #plt.plot(found[:,2],'+')
        plt.grid()
        plt.xlabel('frame number')
        plt.ylabel('z, um')
        plt.title('z drift (um)')
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(6,3))
        plt.subplot(1,2,1)
        plt.plot(subdata[:,f],subdata[:,a],'.')
        #plt.plot(found[:,2],'+')
        plt.grid()
        plt.xlabel('frame number')
        plt.ylabel('intensity, photons')
        plt.title('photon numbers')

        if b:
            plt.subplot(1,2,2)
            plt.plot(subdata[:,f],subdata[:,b],'.')
            #plt.plot(found[:,2],'+')
            plt.grid()
            plt.xlabel('frame number')
            plt.ylabel('background, photons')
            plt.title('background')

        plt.tight_layout()
        plt.show()
        #plt.legend(['localizations','ground truth'])




        plt.plot(crlb.x_abs,crlb.CRLB[:,0]*1000,crlb.x_abs,crlb.CRLB[:,1]*1000,crlb.x_abs,crlb.CRLB[:,2]*1000)
        plt.title('CRLB, photons: {}, bg: {}'.format(num_phot,bg))
        plt.grid()
        plt.xlabel('z, um')
        plt.ylabel('std(err), nm')
        plt.ylim(0,100)
        #plt.show()
        if units == 'um':
            mul = 1
        elif units == 'nm':
            mul = 1000
        p=(np.unique(t)-15)*z_step/mul
        plt.plot(p,found_std[:,x]*1000/mul,'bo')
        plt.plot(p,found_std[:,y]*1000/mul,'go')
        plt.plot(p,found_std[:,z]*1000/mul,'ro')
        plt.legend('xyzXYZ')

        plt.show()


### end
