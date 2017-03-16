#!/usr/bin/python
#

#import theano
#from theano import function, config, shared, sandbox
#import theano.tensor as T
#import theano.sandbox.cuda.dnn
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
#from skimage import data, img_as_float
import time

import cPickle
import numpy as np

#import keras
#from keras.datasets import mnist
#from keras.models import Sequential
#from keras.models import model_from_json
#from keras.layers.core import Dense, Dropout, Activation, Flatten
#from keras.layers.convolutional import Convolution2D, MaxPooling2D
#from keras.utils import np_utils
#from keras.callbacks import EarlyStopping

from PIL import Image
import scipy
#from scipy.interpolate import RegularGridInterpolator

import matplotlib.pyplot as plt
import datetime

def noisy(img,bg,amp):
    img1=img.copy()
    img1=img/img.max()
    img1*=amp*(np.random.rand()+.5)
    img1+=bg*(np.random.rand()+.5)
    img1=np.random.poisson(img1)
    img1=np.array(img1,dtype='float32')
    img1-=img1.mean()
    img1/=np.std(img1)
    return img1

def GDF(img,n1=1.5,n2=4.,thr=.02):
    '''Gaussian difference filter'''
    img=np.array(img,dtype='float32')
    out = ndi.filters.gaussian_filter(img,n1)-ndi.filters.gaussian_filter(img,n2)
    #out = np.array(diff)
    thrs=np.mean(out)+thr*np.std(out)
    out = out-thrs
    out[out<0]=0
    return out

def norm(tmp):
    '''Standart deviation normalization'''
    tmp-=np.mean(tmp)
    tmp/=np.std(tmp)
    return tmp

def norm01(tmp):
    '''0-1 normalization'''
    tmp=np.array(tmp,dtype='float32')
    tmp-=tmp.min()
    tmp/=np.max(tmp)
    return tmp

class SPsampleClass:
    def __init__(self,px_size=.15,zstep=.1):
        self.px_size=px_size
        #self.zrange = zrange
        self.zstep = zstep



    def loadPSFfromTiff(self,path,numPass,numSteps=0,zstep = .1, gdf=1,gdf_low=1.5,gdf_high=4.0,gdf_thr=0,mean = 0,norm=0):
        try:
            self.zstep = zstep
            im = Image.open(path)
            if not numSteps: numSteps=im.n_frames/numPass
            PSF=[]
            rawPSF = []
            rawPSF1 = []
            ii=0
            for k in range(numPass):
                #print i
                for i in range(numSteps):
                    im.seek(i+k*numSteps)
                    img=np.array(im, dtype='float32')
                    img1=img/float(np.sum(img))
                    rawPSF.append(img1)
                    ii+=1
                    print "\r\rProcessing frame %d/%d"%(ii,numSteps*numPass),
                rawPSF1.append(rawPSF)
                rawPSF=[]

            if numPass>0:
                PSF=np.mean(rawPSF1,axis=0)
            normPSF=[]
            for k in PSF:
                if gdf:
                    tmp = GDF(k, gdf_low, gdf_high, gdf_thr)
                elif mean:
                    tmp = k-np.mean(k)-gdf_thr*np.std(k)
                else:
                    tmp=k

                tmp[tmp<0] = 0
                if norm:
                    normPSF.append(tmp/tmp.sum())
                else:
                    normPSF.append(tmp)
                #normPSF.append(tmp/tmp.max())
            self.PSF=np.array(normPSF)

            self.PSF_rows, self.PSF_cols = self.PSF.shape[1],self.PSF.shape[2]
            print '\nPSF is loaded with dimensions ',PSF.shape

            self.avPSF=PSF.mean(axis=0)
            self.zframes=self.PSF.shape[0]
            self.zrange=self.zstep*(self.zframes-1)
            self.zvector = np.arange(-self.zrange/2,self.zrange/2+self.zstep,self.zstep)

            #cPickle.dump(self.avPSF, open("avPSF.cPickle", 'wb'))
            print 'avPSF is created with dimensions ',self.avPSF.shape

            #x = np.linspace(-self.avPSF.shape[0]/2*self.px_size*1000,(self.avPSF.shape[0]/2-1)*self.px_size*1000,self.avPSF.shape[0])
            #y = np.linspace(-self.avPSF.shape[1]/2*self.px_size*1000,(self.avPSF.shape[1]/2-1)*self.px_size*1000,self.avPSF.shape[1])
            #z = np.linspace(-self.zrange/2*1000,self.zrange/2*1000,self.PSF.shape[0])
            x = np.linspace(-self.avPSF.shape[0]/2,(self.avPSF.shape[0]/2-1),self.avPSF.shape[0])
            y = np.linspace(-self.avPSF.shape[1]/2,(self.avPSF.shape[1]/2-1),self.avPSF.shape[1])
            z = np.arange(self.zframes)
            self.Interpolator =  RegularGridInterpolator((z, x, y), np.array(self.PSF),bounds_error=False,fill_value=0)
            print 'Interpolator is created '

            cPickle.dump((self.PSF,self.avPSF,self.Interpolator), open("PSF.cPickle", 'wb'))
            print 'cPickled all into PSF.cPickle'

        except IOError as e:
            print 'IOError :', e
            pass

    def interpolate(self,y,x,z):
        try:
            self.Interpolator
        except NameError:
            print "loading Interpolator"
            self.loadPSFfromcPickle()
        #else:
        #      print "sure, it was defined."
        yshift=0#-270
        #ys = np.linspace(-self.avPSF.shape[0]/2*self.px_size*1000-y+yshift,(self.avPSF.shape[0]/2-1)*self.px_size*1000-y+yshift,self.avPSF.shape[0])
        ys = np.linspace(-self.avPSF.shape[0]/2-y+yshift,(self.avPSF.shape[0]/2-1)-y+yshift,self.avPSF.shape[0])
        xshift=0
        #xs = np.linspace(-self.avPSF.shape[0]/2*self.px_size*1000-x+yshift,(self.avPSF.shape[0]/2-1)*self.px_size*1000-x+yshift,self.avPSF.shape[0])
        xs = np.linspace(-self.avPSF.shape[0]/2-x-xshift,(self.avPSF.shape[0]/2-1)-x-xshift,self.avPSF.shape[1])

        return self.Interpolator( [ [z, x0,y0] for x0,y0 in \
                          np.array(np.meshgrid( xs, ys)).reshape(2,self.avPSF.shape[0]*self.avPSF.shape[1]).transpose()])\
                        .reshape(self.avPSF.shape[0],self.avPSF.shape[1]).transpose()


    def loadPSFfromcPickle(self,path='PSF.cPickle'):
        self.PSF, self.avPSF, self.Interpolator = cPickle.load(open(path, 'rb'))
        self.PSF_rows, self.PSF_cols = self.PSF.shape[1],self.PSF.shape[2]
        self.zframes=self.PSF.shape[0]
        self.zrange=self.zstep*(self.zframes-1)
        self.zvector = np.arange(-self.zrange/2,self.zrange/2+self.zstep,self.zstep)
        print '\nPSF is loaded with dimensions ',self.PSF.shape

    def showPSF(self):
        try:
            self.PSF
        except AttributeError:
            print "No PSF found, loading PSF from cPickle"
            self.loadPSFfromcPickle()
        finally:
            fig=plt.figure(figsize=(15,10*self.PSF.shape[0]/60))
            for i in range(self.PSF.shape[0]):

                ax = fig.add_subplot(self.PSF.shape[0]/10+1,11,i+1)
                plt.imshow(self.PSF[i], cmap=plt.cm.gray, interpolation='none')
                plt.title(str(i+1))
                plt.axis('off')
            plt.tight_layout()
            plt.show()

    def showPSFxz(self):
        try:
            self.PSF
        except AttributeError:
            print "No PSF found, loading PSF from cPickle"
            self.loadPSFfromcPickle()
        finally:
            fig=plt.figure(figsize=(15,10))
            for i in range(self.PSF.shape[2]):

                ax = fig.add_subplot(4,8,i+1)
                plt.imshow(self.PSF[:,:,i], cmap=plt.cm.gray, interpolation='none')
                plt.title(str(i+1))
                plt.axis('off')
            plt.tight_layout()


    def showPSFyz(self):
        try:
            self.PSF
        except AttributeError:
            print "No PSF found, loading PSF from cPickle"
            self.loadPSFfromcPickle()
        finally:
            fig=plt.figure(figsize=(15,10))
            for i in range(self.PSF.shape[1]):

                ax = fig.add_subplot(4,8,i+1)
                plt.imshow(self.PSF[:,i], cmap=plt.cm.gray, interpolation='none')
                plt.title(str(i+1))
                plt.axis('off')
            plt.tight_layout()


    def genSPTrainSet(self,trainnum,xr=2.,yr=2.,bg=0.,ampl=100.):
        Y_train=np.random.rand(trainnum,3)
        # x
        Y_train[:,0]-=.5
        Y_train[:,0]*=xr
        # y
        Y_train[:,1]-=.5
        Y_train[:,1]*=yr
        # z
        Y_train[:,2]-=.5
        Y_train[:,2]*=self.zrange


        #print y_train[2]
        X_train=[]
        for i in range(trainnum):
            img=self.interpolate(Y_train[i,0]*1000,Y_train[i,1]*1000,Y_train[i,2]*1000)
            #img=noisy(img,gauss=.12,maxi=100,poisson=True)
            if ampl:
                img=noisy(img,bg,ampl)
            #img = GDF(norm(img),1.5,4,.0)
            img=norm(img)
            X_train.append(img)
            print "\rProcessing frame %s"%(i+1),
        print '\n'
        return np.array(X_train).reshape(trainnum,1,self.PSF.shape[1],self.PSF.shape[2]).astype('float32'),\
                   np.array(Y_train)

    def genSPTrainTestSet(self,trainnum,testnum,xr=2.,yr=2.,bg=0.,ampl=100.):
        self.X_train, self.Y_train = self.genSPTrainSet(trainnum,xr,yr,bg,ampl)
        self.X_test, self.Y_test = self.genSPTrainSet(testnum,xr,yr,bg,ampl)
        cPickle.dump((self.X_train, self.Y_train, self.X_test, self.Y_test), open("trainset.cPickle", 'wb'))
        print 'cPickled self.X_train, self.Y_train, self.X_test, self.Y_test into trainset.cPickle'

    def loadSPTrainTestSet(self,path='trainset.cPickle'):
        self.X_train, self.Y_train, self.X_test, self.Y_test = cPickle.load(open(path, 'rb'))


    def compileCNNnet(self,nb_epoch = 12,nb_filters = 32,nb_conv = 9):
        #batch_size = 200
        #nb_classes = 10
        nb_epoch = nb_epoch

        # input image dimensions

        # number of convolutional filters to use
        nb_filters = nb_filters
        # convolution kernel size
        nb_conv = nb_conv

        X_train = self.X_train.astype('float32')
        X_test = self.X_test.astype('float32')
        Y_train=self.Y_train
        Y_test=self.Y_test
        print('X_train shape:', X_train.shape)
        print('Y_train shape:', Y_train.shape)
        print(X_train.shape[0], 'train samples')
        print(X_test.shape[0], 'test samples')


        self.early_stopping = EarlyStopping(monitor='val_loss', patience=6)

        self.history = LossHistory()

        self.model = Sequential()

        self.model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                                border_mode='valid',
                                input_shape=(1, self.PSF_rows, self.PSF_cols)))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
        self.model.add(Activation('relu'))

        self.model.add(Dropout(0.5))

        self.model.add(Flatten())

        self.model.add(Dense(3))
        self.model.add(Activation('linear'))

        self.model.compile(loss='mean_squared_error', optimizer="rmsprop")

    def trainCNNnet(self,):

        self.model.fit(self.X_train, self.Y_train, batch_size=200, nb_epoch=200, show_accuracy=True, verbose=1, validation_split=0.2, callbacks=[self.early_stopping,self.history])
        self.model.optimizer.lr.set_value(0.0001)
        self.model.fit(self.X_train, self.Y_train, batch_size=200, nb_epoch=200, show_accuracy=True, verbose=1, validation_split=0.2, callbacks=[self.early_stopping,self.history])
        #model.fit(X_train, Y_train, batch_size=1344, nb_epoch=2, show_accuracy=True, verbose=1, validation_split=0.2, callbacks=["""early_stopping,""""history])
        score = self.model.evaluate(self.X_test, self.Y_test, show_accuracy=True, verbose=0)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])
        self.model.save_weights('weights.h5')
        json_string = self.model.to_json()
        open('models.json', 'w').write(json_string)
        print 'Saved model and weights into models.json weights.h5'

    def loadCNNnet(self,model_location='models/model_architecture-2015-02-08.json',weights_location='weights/weights2015-02-09-noisy0-100-loss=0.013.h5'):
        self.model = model_from_json(open(model_location).read())
        self.model.load_weights(weights_location)

    def processStack(self,path,debug=0,limit=0):

        debug=debug
        frame=Image.open(path)
        print 'Found a frame of %d x %d pixels'%(np.array(frame).shape[0],np.array(frame).shape[1])
        counter=0
        if not limit:
            limit=1000000

        avPSF4d=self.avPSF.reshape(1,1,self.PSF_rows,self.PSF_cols)
        imgx=T.ftensor4('imgx')
        x = shared(np.asarray(avPSF4d, config.floatX))
        xcorr = function([imgx], theano.sandbox.cuda.dnn.dnn_conv(imgx, x,conv_mode='cross',border_mode='full'))


        self.found=[]
        try:
            while counter<limit:
                frame.seek(counter)
                counter+=1

                img0=np.array(frame, dtype='float32')
                img1=img0.copy()
                imgf=GDF(norm(img0),1.,3.,2)
                if debug:
                    fig = plt.figure(figsize=(15,5))
                    ax=fig.add_subplot(1,3,1)
                    plt.imshow(img0,interpolation='none')
                    ax=fig.add_subplot(1,3,2)
                    plt.imshow(imgf,interpolation='none')
                    plt.colorbar()

                #plt.imshow(imgf)
                #plt.colorbar()
                #break
                img=norm01(img0).reshape(1,1,img0.shape[0],img0.shape[1])
                r = xcorr(img) # xcorr result on GPU
                #r.append(f())

                a=np.asarray(r) #xcorr result in MEM

                locdist=10
                im = (a[0,0])
                #plt.imshow(im)
                #break
                if debug:
                    print 'xcorr shape, max', im.shape, im.max()
                    #fig = plt.figure()
                    ax=fig.add_subplot(1,3,3)
                    plt.imshow(im,interpolation='none')
                    plt.colorbar()
                    plt.tight_layout()
                    plt.show()

                coordinates = (peak_local_max(im, min_distance=locdist))
                if debug: print coordinates

                #for k in range(len(coordinates)):
                if len(coordinates):
                #b.append(np.array(a))
                    stack=[]
                    cc=[]
                    for c in coordinates:

                        cc.append(np.array(c))
                        if img1.shape[0]>32:
                            tmp=np.array(img1[c[0]:c[0]+self.PSF_rows,c[1]:c[1]+self.PSF_cols])
                        else:
                            tmp = imgf
                        tmp=norm(tmp)
                        stack.append(tmp)
                        #crops.append(tmp)
                        #print c
                        #plt.imshow(tmp)
                        #plt.show()
                    stack=np.array(stack)
                    cc=np.array(cc)
                    if debug: print stack.shape
                    prediction=self.model.predict(stack.reshape(stack.shape[0],1,stack.shape[1],stack.shape[2]))
                    if debug: print cc,prediction
                    #cropshifts.append(np.array(prediction))
                    b = np.zeros((cc.shape[0],3))

                    b[:,0] = cc[:,1]*self.px_size
                    b[:,1] = cc[:,0]*self.px_size
                    #print b
                    #print "pred", prediction


                    ff=b+prediction
                    #print ff

                    if len(self.found):
                        self.found=np.append(self.found,ff,axis=0)
                        #cropshifts=np.append(cropshifts,prediction,axis=0)
                    else:
                        self.found=ff
                        #cropshifts=prediction
                if debug:
                    print 'Found ', len(coordinates)
                    fig=plt.figure(figsize=(len(coordinates)*5,10))
                    i=0
                    for c in range(len(coordinates)):

                        ax = fig.add_subplot(len(coordinates),3,i+1)
                        plt.imshow(norm01(stack[c]), cmap=plt.cm.gray, interpolation='none')
                        plt.title(str(np.sum(stack[c])))
                        plt.colorbar()
                        ax = fig.add_subplot(len(coordinates),3,i+2)
                        estimate=(self.interpolate(prediction[c,0]*1000,prediction[c,1]*1000,prediction[c,2]*1000))
                        diff = norm01(stack[c])-norm01(GDF(norm(estimate),1.,3.,2))
                        plt.imshow(norm01(GDF(norm(estimate),1.,3.,2)), cmap=plt.cm.gray, interpolation='none')
                        plt.title(str(np.sum(estimate)))
                        #plt.imshow(generate(0,0,(c-10)*200,Interpolator), cmap=plt.cm.gray, interpolation='none')
                        plt.colorbar()
                        ax = fig.add_subplot(len(coordinates),3,i+3)
                        plt.imshow(diff,cmap=plt.cm.gray, interpolation='none')
                        plt.title(str(np.sum(np.sqrt(np.power(diff,2)))))
                        plt.colorbar()
                        i+=3

                    #plt.tight_layout()
                    plt.show()
                print "\rProcessing frame %d"%(counter),
            print '\nFound %d particles'%self.found.shape[0]

        except EOFError:
            print 'No more frames'

#print 'SPsample library is loaded'
