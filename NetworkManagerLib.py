### implementation of network manager for palm reconstruction

import multiprocessing
import Queue
from multiprocessing.managers import BaseManager,DictProxy
import cPickle
from Zernike import *
import socket
import sys,os,csv

class NetworkManager(object):
    def __init__(self,filePath = None, myFitPath='PSF/2016-12-22-PSF/ast0 psf 100 nm_1/myFit.pkl',
                        myParamsPath=None,ip='',port=50000,authkey = 'abracadabra'):
        mgr = multiprocessing.Manager()

        if myParamsPath is None:
            myParams = dict(min_photons=100,
                        x_thr=.1,
                        min_distance=5,
                        crop_size=32,
                        bg_kernel=10,
                        myFitPath = myFitPath)
        else:
            myParams = cPickle.load(open(myParamsPath))

        status = dict(hosts = [],
                    frames = 0,
                    molecules = 0)

        self.queue = mgr.Queue()
        myParamDict = mgr.dict(myParams)
        statusDict = mgr.dict(status)
        self.activeList = mgr.list()
        lock = mgr.Lock()
        self.results = mgr.list()
        self.molecules = 0
        self.frames=mgr.list()
        self.found = np.empty((0,6))
        self.stop = False

        class QueueManager(multiprocessing.managers.BaseManager): pass
        QueueManager.register('get_queue', callable=lambda:self.queue)
        QueueManager.register('get_params', callable=lambda:myParamDict)
        QueueManager.register('get_lock', callable=lambda:lock)
        QueueManager.register('get_active', callable=lambda:self.activeList)
        QueueManager.register('put_results', callable=lambda:self.results)
        QueueManager.register('put_frames', callable=lambda:self.frames)
        QueueManager.register('status',statusDict,DictProxy)

        if filePath is not None:
            self.loadFiles(filePath)
            self.filePath = filePath

        print 'Server is ready!'

        tt = threading.Thread(target=self.collectionRoutines,args=())
        tt.daemon=True
        tt.start()

        tt = threading.Thread(target=self.printStatus,args=())
        tt.daemon=True
        tt.start()

        tt = threading.Thread(target=self.updateFolder,args=())
        tt.daemon=True
        tt.start()

        m = QueueManager(address=(ip, port), authkey=authkey)

        s = m.get_server()
        self.server = s
        s.serve_forever()

    def collectionRoutines(self):
        path = self.folder[:-5]+'myLoctmp.csv'
        header='"frame","x [nm]","y [nm]","z [nm]","intensity [photons]","background [photons]"\n'

        try:
            f = open(path,'r')
        except IOError:
            with open(path,'w') as f:
                f.write(header)

        try:
            while not self.queue.empty():
                try:
                    if len(self.results)>len(self.found):
                        a=self.results[len(self.found):]
                        #np.savetxt()
                        self.found=np.append(self.found,a,axis=0)
                        self.molecules=len(self.found)

                        with open(path,'a') as f:
                            wr = csv.writer(f)
                            wr.writerows(a)


                    else:
                        time.sleep(1)

                except Exception as e:
                    traceback.print_exc()
                    print a
                    print e


            self.saveResults()
            #self.server.shutdown(self.server)
            t = 0
            print '\n',
            while len(self.activeList)>0:
                print '\rWaiting for workers to disconnect {} sec ({} left)'.format(t,len(self.activeList)),
                sys.stdout.flush()
                t+=1
                time.sleep(1)
            print '\rWaiting for workers to disconnect {} sec ({} left)'.format(t,len(self.activeList)),
            print '\n exiting'
            sys.stdout.flush()
            os._exit(0)


        except Exception as e:
            print e
            self.saveResults()
            sys.exit(1)


    def loadFiles(self,folder):
        self.folder = folder
        print 'reading file list from ',folder,'*.tif'
        sys.stdout.flush()
        fileList = sorted(glob.glob(folder+'*.tif'))
        self.fileList = fileList
        for i in range(len(fileList)):
            self.queue.put((i,fileList[i]))
            #print '\rloading files ',i,
        print 'loaded {} files '.format(i+1)
        self.length = i+1
        print '\n'

    def updateFolder(self):
        while not self.stop:
            try:
                newFileList = glob.glob(self.folder+'*.tif')
                if len(newFileList)>len(self.fileList):
                    diff = newFileList
                    for f in self.fileList:
                        diff.remove(f)
                    for i in range(len(diff)):
                        self.queue.put((self.length,diff[i]))
                        self.length +=1
                        self.fileList.append(diff[i])
                time.sleep(60)

            except:
                traceback.print_exc()
                break


    def saveResults(self):
        print 'saving results:'
        sys.stdout.flush()
        path = self.folder[:-5]
        self.path = path+'myLoc.csv'
        try:
            saveTS(self.found,open(self.path,'w'))
            print  'results save in {}'.format(self.path)
            sys.stdout.flush()
        except Exception as e:
            print 'Saving failed, {}'.format(e)
            sys.stdout.flush()

    def printStatus(self):
        while len(self.frames)<self.length:
            print '\r {}/{} frames, {} molecules, {} workers'.format(len(self.frames),
                                                        self.length,
                                                        self.molecules,
                                                        len(self.activeList)),
            sys.stdout.flush()
            time.sleep(1)



class NetworkReconstructor(object):
    def __init__(self,ip='',port=50000,authkey = 'abracadabra'):

        class QueueManager(BaseManager): pass

        QueueManager.register('get_queue')
        QueueManager.register('get_params')
        QueueManager.register('get_lock')
        QueueManager.register('get_active')
        QueueManager.register('put_results')
        QueueManager.register('status')
        QueueManager.register('put_frames')

        try:
            m = QueueManager(address=(ip,port), authkey=authkey)
            m.connect()
        except:
            print 'Server not found, exiting'
            sys.stdout.flush()
            os._exit(1)

        self.queue = m.get_queue()
        self.get_params = m.get_params()
        self.lock = m.get_lock()
        self.res = m.put_results()
        self.frames = m.put_frames()
        self.workers = m.get_active()
        self.status = m.status

        self.readParams()
        self.defineMLE()

        self.name = socket.gethostname()


    def readParams(self):
        self.params = dict(self.get_params.items())



    def defineMLE(self):
        self.myFit = PhaseFitWrap(fileName=self.params['myFitPath'],plot=False)
        self.xc = XcorrMLE(self.myFit.psfArray,
                      self.myFit.zVect,
                      self.myFit.zernPhase,
                      **self.params)

    def poolMLE(self,frame):
        out = self.xc.run(frame)
        return out

    def worker(self):


        self.workers.append(self.name)
        #print 'start worker {}'.format(name)
        px = self.myFit.px_size
        try:
            while 1:
                i,path = self.queue.get(True,10)
                #print i,path
                #with self.lock:
                frame = self.readFrame(path)
                out = self.poolMLE(frame)
                tmp = []

                self.lock.acquire()
                for o in out:
                    self.res.append([int(i),(-o.x+o.X*px)*1000,(-o.y+o.Y*px)*1000,(o.z)*1000, int(o.a), int(o.b)])
                self.frames.append(i)
                self.lock.release()
                    #print '\r{} frames, {} particles'.format(len(self.pool.exitCodes),len(self.pool.arr)),


        except Queue.Empty:
            try:
                self.workers.remove(self.name)
            except (EOFError,IOError):
                pass
            return 0

        except Exception as e:
            try:
                self.workers.remove(self.name)
            except (EOFError,IOError):
                pass

            return 1

    def readFrame(self,path):
        return io.imread(path)

    def addWorker(self,join=True):
        '''add a single worker'''
        w=multiprocessing.Process(target=self.worker,args=())
        #self.workers.append(w.name)

        w.daemon = True
        w.start()

        name = socket.gethostname()
        print self.name, multiprocessing.cpu_count()
        sys.stdout.flush()

        if join: w.join()
        #print 'started {} workers'.format(self.workers.count(1))

    def addManyWorkers(self,join=True):
        ''' add a number of workers = cpu_count'''
        ww = []
        for i in range(multiprocessing.cpu_count()):
            w= multiprocessing.Process(target=self.worker,args=())
            ww.append(w)
            #self.workers.append(w.name)

            w.daemon = True
            w.start()
        try:
            print self.name, multiprocessing.cpu_count(),len(ww)
            sys.stdout.flush()
            for w in ww:
                if join: w.join()
        except Exception as e:
            return 1
        #print 'started {} workers'.format(self.workers.count(1))

    def loadFiles(self,folder):
        self.folder = folder
        fileList = sorted(glob.glob(folder+'*.tif'))
        for i in range(len(fileList)):
            self.queue.put((i,fileList[i]))
            print '\r',i,
