from NetworkManagerLib import *
from Zernike import *
import json

myDict = json.loads(json.load(open('server.dict')))

c = NetworkReconstructor(ip = myDict['ip'])

c.addManyWorkers()
