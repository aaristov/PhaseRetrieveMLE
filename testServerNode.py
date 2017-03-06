from NetworkManagerLib import *
import json,os,sys

ipv4 = os.popen('ip addr show eth2').read().split('inet')[1].split('/')[0]
myDict = dict(ip = ipv4,host = socket.gethostname())
print myDict
json.dump(json.dumps(myDict),open('server.dict','w'))

s = NetworkManager(ip=ipv4,filePath=sys.argv[1],
                  myFitPath='../data/2017-03-02-tub/SP-fit.pkl')
