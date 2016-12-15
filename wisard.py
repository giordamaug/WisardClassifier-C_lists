# -*- coding: utf-8 -*-
"""
WiSARD C Library Wrapper

Created on Tue Sep  3 16:41:54 2013

@author: maurizio
"""

from ctypes import *
import os
import random
import platform
if platform.system() == 'Linux':
    suffix = '.so'
elif platform.system() == 'Windows':
    suffix = '.dll'
elif platform.system() == 'Darwin':
    suffix = '.dylib'
else:
    raise Exception("Unsupported Platform")

libpath = "libwisard-cxx_3.0" + suffix
wizl = CDLL(libpath)
from PIL import Image
HEADKEY = 18446744073709551615L

""" Mapping data structure """

c_value = c_double
c_key = c_ulong

class Wentry(Structure):
    pass

class Wkeymax(Structure):
    pass

Wkeymax._fields_ = [("key", c_key),
                   ("value", c_value)]

Wentry._fields_ = [("key", c_key),
                ("value", c_value),
                ("next", POINTER(Wentry)),
                ("prev", POINTER(Wentry))]

class Discr(Structure):
    _fields_ = [("n_ram", c_int),
                ("n_bit", c_int),
                ("n_loc", c_key),
                ("size", c_int),
                ("tcounter", c_ulong),
                ("rams", POINTER(POINTER(Wentry))),
                ("maxkeys", POINTER(Wkeymax)),
                ("map", POINTER(c_int)),
                ("rmap", POINTER(c_int)),
                ("mi", POINTER(c_value)),
                ("maxmi", c_value),
                ("name", c_char_p)]

def discr_n_ram(discr):
    return discr.contents.n_ram

""" RAM interface """
_wram_create = wizl.wram_create
_wram_create.restype =  POINTER(Wentry)
_wram_set = wizl.wram_set
_wram_set.argtypes = [ POINTER(Wentry), c_key, c_value ]
_wram_set.restype =  c_value
_wram_incr = wizl.wram_incr
_wram_incr.argtypes = [ POINTER(Wentry), c_key ]
_wram_incr.restype =  c_value
_wram_decr = wizl.wram_decr
_wram_decr.argtypes = [ POINTER(Wentry), c_key ]
_wram_decr.restype =  c_value
_wram_decr_all_but_key = wizl.wram_decr_all_but_key
_wram_decr_all_but_key.argtypes = [ POINTER(Wentry), c_key ]
_wram_decr_all_but_key.restype =  c_value
_wram_del = wizl.wram_del
_wram_del.argtypes = [ POINTER(Wentry), c_key ]
_wram_del.restype =  c_value
_wram_get = wizl.wram_get
_wram_get.argtypes = [ POINTER(Wentry), c_key ]
_wram_get.restype =  c_value
_wram_len = wizl.wram_len
_wram_len.argtypes = [ POINTER(Wentry) ]
_wram_len.restype =  c_int

def wram_build(n,dict):
    ram = _wram_create()
    maxkey = 2**n
    for key,value in dict.items():
        if key > 0:
            _wram_set(ram,key,value)
    return ram

def wram_get(ram, key):
    return _wram_get(ram,c_key(key))

def wram_incr(ram, key):
    return _wram_incr(ram,c_key(key))

def wram_decr(ram, key):
    return _wram_decr(ram,c_key(key))

def wram_decr_all_but_key(ram,key):
    return _wram_decr_all_but_key(ram,c_key(key))

def wram_del(ram, key):
    return _wram_del(ram,c_key(key))

def wram_set(ram, key, value):
    return _wram_set(ram,c_key(key),c_value(value))

def showRam(ram):
    size = _wram_len(ram)
    p = ram.contents.next
    res = {}
    while p.contents.key != HEADKEY:
        res[int(p.contents.key)] = p.contents.value
        p = p.contents.next
    return res
        
""" Constructor interface """
_makeDiscr =  wizl.makeDiscr
_makeDiscr.restype =  POINTER(Discr)
_makeDiscr.argtypes = [ c_int, c_int, c_char_p, c_char_p ]

_makeTrainImgDiscr =  wizl.makeTrainImgDiscr
_makeTrainImgDiscr.argtypes = [ c_int, c_int, c_char_p, c_char_p, c_void_p, c_int, c_int, c_int, c_int, c_int ]
_makeTrainImgDiscr.restype =  POINTER(Discr)

_makeTrainImgBinDiscr =  wizl.makeTrainImgBinDiscr
_makeTrainImgBinDiscr.argtypes = [ c_int, c_int, c_char_p, c_char_p, c_void_p, c_int ]
_makeTrainImgBinDiscr.restype =  POINTER(Discr)

def makeDiscr(nbit, size, name, maptype):
    return _makeDiscr(c_int(nbit), c_int(size), c_char_p(name), c_char_p(maptype))

def makeTrainImgDiscr(nbit, size, name, maptype, img, cols, bx, by, width, tics):
    return _makeTrainImgDiscr(c_int(nbit), c_int(size), c_char_p(name), c_char_p(maptype), c_void_p(img.ctypes.data), c_int(cols), c_int(bx), c_int(by), c_int(width), c_int(tics))

def makeTrainImgBinDiscr(nbit, size, name, maptype, img, cols):
    return _makeTrainImgBinDiscr(c_int(nbit), c_int(size), c_char_p(name), c_char_p(maptype), c_void_p(img.ctypes.data), c_int(cols))

""" Train/Classify wrappers"""
_trainDiscr = wizl.trainDiscr
_trainDiscr.argtypes = [ POINTER(Discr), POINTER(c_key) ]
_trainImgDiscr = wizl.trainImgDiscr
_trainImgDiscr.argtypes = [ POINTER(Discr), c_void_p, c_int, c_int, c_int, c_int, c_int ]
_trainImgBinDiscr = wizl.trainImgBinDiscr
_trainImgBinDiscr.argtypes = [ POINTER(Discr), c_void_p, c_int ]
_trainforgetDiscr = wizl.trainforgetDiscr
_trainforgetDiscr.argtypes = [ POINTER(Discr), POINTER(c_key) , c_value, c_value]
_trainforgettopDiscr = wizl.trainforgettopDiscr
_trainforgettopDiscr.argtypes = [ POINTER(Discr), POINTER(c_key) , c_value, c_value, c_value]
_trainforgetImgDiscr = wizl.trainforgetImgDiscr
_trainforgetImgDiscr.argtypes = [ POINTER(Discr), c_void_p, c_int, c_int, c_int, c_int, c_int ]
_trainforgetImgBinDiscr = wizl.trainforgetImgBinDiscr
_trainforgetImgBinDiscr.argtypes = [ POINTER(Discr), c_void_p, c_int ]
_trainresetImgDiscr = wizl.trainforgetImgDiscr
_trainforgetImgDiscr.argtypes = [ POINTER(Discr), c_void_p, c_int, c_int, c_int, c_int, c_int ]
_classifyDiscr = wizl.classifyDiscr
_classifyDiscr.argtypes = [ POINTER(Discr), POINTER(c_key) ]
_classifyDiscr.restype = c_double
_classifyImgDiscr = wizl.classifyImgDiscr
_classifyImgDiscr.argtypes = [ POINTER(Discr), c_void_p, c_int, c_int, c_int, c_int, c_int ]
_classifyImgDiscr.restype = c_double
_classifyImgBinDiscr = wizl.classifyImgBinDiscr
_classifyImgBinDiscr.argtypes = [ POINTER(Discr), c_void_p, c_int ]
_classifyImgBinDiscr.restype = c_double
# for SVM
_trainSvmHistoDiscr = wizl.trainSvmHistoDiscr
_trainSvmHistoDiscr.argtypes = [ POINTER(Discr), POINTER(c_double), POINTER(c_double), POINTER(c_double), c_int, c_int ]
_classifySvmHistoDiscr = wizl.classifySvmHistoDiscr
_classifySvmHistoDiscr.argtypes = [ POINTER(Discr), POINTER(c_double), POINTER(c_double), POINTER(c_double), c_int, c_int ]
_classifySvmHistoDiscr.restype = c_double
_classifySvmHistoDiscrThresholded = wizl.classifySvmHistoDiscrThresholded
_classifySvmHistoDiscrThresholded.argtypes = [ POINTER(Discr), POINTER(c_double), POINTER(c_double), POINTER(c_double), c_int, c_int, c_double ]
_classifySvmHistoDiscrThresholded.restype = c_double
def trainSvmHistoDiscr(discr, data, den, off, ntics, nfeatures):
    _trainSvmHistoDiscr(discr, (c_double * nfeatures)(*data), (c_double * nfeatures)(*den),(c_double * nfeatures)(*off), ntics, nfeatures)
def classifySvmHistoDiscr(discr, data, den, off, ntics, nfeatures):
    return _classifySvmHistoDiscr(discr, (c_double * nfeatures)(*data), (c_double * nfeatures)(*den),(c_double * nfeatures)(*off), ntics, nfeatures)
def classifySvmHistoDiscrThresholded(discr, data, den, off, ntics, nfeatures, threshold):
    return _classifySvmHistoDiscrThresholded(discr, (c_double * nfeatures)(*data), (c_double * nfeatures)(*den),(c_double * nfeatures)(*off), ntics, nfeatures, threshold)


def trainDiscr(discr, intuple):
    n = len(intuple)
    if n != discr.contents.n_ram:
        raise NameError('Wrong InTuple size')
    c_tuple = (c_ulong * n)(*intuple)
    _trainDiscr(discr, c_tuple)

def trainImgDiscr(discr, img, cols, bx, by, width, tics):
    _trainImgDiscr(discr, c_void_p(img.ctypes.data), cols, bx, by, width, tics)

def trainImgBinDiscr(discr, img, cols):
    _trainImgBinDiscr(discr, c_void_p(img.ctypes.data), cols)

def trainforgetImgDiscr(discr, img, cols, bx, by, width, tics):
    _trainforgetImgDiscr(discr, c_void_p(img.ctypes.data), cols, bx, by, width, tics)

def trainforgetImgBinDiscr(discr, img, cols):
    _trainforgetImgBinDiscr(discr, c_void_p(img.ctypes.data), cols)

def trainresetImgDiscr(discr, img, cols, bx, by, width, tics):
    _trainresetImgDiscr(discr, c_void_p(img.ctypes.data), cols, bx, by, width, tics)

def trainforgetDiscr(discr, intuple, incr, decr):
    n = len(intuple)
    if n != discr.contents.n_ram:
        raise NameError('Wrong InTuple size')
    c_tuple = (c_ulong * n)(*intuple)
    _trainforgetDiscr(discr, c_tuple, incr, decr)

def trainforgettopDiscr(discr, intuple, incr, decr, top):
    n = len(intuple)
    if n != discr.contents.n_ram:
        raise NameError('Wrong InTuple size')
    c_tuple = (c_ulong * n)(*intuple)
    _trainforgettopDiscr(discr, c_tuple, incr, decr, top)

def classifyDiscr(discr, intuple):
    n = len(intuple)
    if n != discr.contents.n_ram:
        raise NameError('Wrong InTuple size')
    c_tuple = (c_ulong * n)(*intuple)
    return _classifyDiscr(discr, c_tuple)

def classifyImgDiscr(discr, img, cols, bx, by, width, tics):
    return _classifyImgDiscr(discr, c_void_p(img.ctypes.data), cols, bx, by, width, tics)

def classifyImgBinDiscr(discr, img, cols):
    return _classifyImgBinDiscr(discr, c_void_p(img.ctypes.data), cols)

""" Mental Image methods """
_mentalDiscr = wizl.mentalDiscr
_mentalDiscr.argtypes = [ POINTER(Discr) ]
_mentalDiscr.restype = c_value

def mentalDiscr(discr):
    return _mentalDiscr(discr)

""" Utilities wrapper """
_copyDiscr =  wizl.copyDiscr
_copyDiscr.restype =  POINTER(Discr)
_printDiscr =  wizl.printDiscr
_shrinkDiscr =  wizl.shrinkDiscr
_shrinkDiscr.argtypes = [ POINTER(Discr) ]
_initDiscr =  wizl.initDiscr
_initDiscr.argtypes = [ POINTER(Discr) ]

def copyDiscr(discr):
    return _copyDiscr(discr)

def shrinkDiscr(discr):
    _shrinkDiscr(discr)

def initDiscr(discr):
    _initDiscr(discr)

def printDiscr(discr):
    _printDiscr(discr)
    
def getNBitDiscr(discr):
    return discr.contents.n_bit

def getNRamDiscr(discr):
    return discr.contents.n_ram

def getNameDiscr(discr):
    if not discr.contents.name:
        return ''
    return discr.contents.name

def getTcounterDiscr(discr):
    return discr.contents.tcounter

def getMapRefDiscr(discr, index):
    return discr.contents.map[index]

def getRMapRefDiscr(discr, index):
    return discr.contents.rmap[index]

def getSizeDiscr(discr):
    return discr.contents.size

def getMaxMIDiscr(discr):
    return discr.contents.maxmi

def getMIRefDiscr(discr,index):
    return discr.contents.mi[index]

def getMapDiscr(discr):
    if not discr.contents.map:
        return []
    res = [ None for i in range(discr.contents.size)]
    for i in range(discr.contents.size):
        res[i] = discr.contents.map[i]
    return res

def setMapDiscr(discr,map):
    if discr.contents.size != len(map):
        print "Error: input map has wrong size"
    for i in range(discr.contents.size):
        discr.contents.map[i] = map[i]
        discr.contents.rmap[map[i]] = i
    return

def getRMapDiscr(discr):
    if not discr.contents.rmap:
        return []
    res = [ None for i in range(discr.contents.size)]
    for i in range(discr.contents.size):
        res[i] = discr.contents.rmap[i]
    return res

def getMaxKeysDiscr(discr):
    if not discr.contents.maxkeys:
        return []
    return [ (discr.contents.maxkeys[i].key,discr.contents.maxkeys[i].value) for i in range(discr.contents.n_ram)]

def getMIDiscr(discr):
    if not discr.contents.mi:
        return []
    res = [ None for i in range(discr.contents.size)]
    for i in range(discr.contents.size):
        res[i] = discr.contents.mi[i]
    return res

def getRamsDiscr(discr):
    res = [ None for i in range(discr.contents.n_ram)]
    for i in range(discr.contents.n_ram):
        ram = discr.contents.rams[i]
        p = ram.contents.next
        res[i] = {}
        while p.contents.key != HEADKEY:
            res[i][int(p.contents.key)] = p.contents.value    
            p = p.contents.next
    return res
                
def mentalImageDiscr(discr,w, h):
    if (w * h) == getSizeDiscr(discr):
        img = Image.new('RGB', (w, h), "white")
        pixels = img.load()
        maxval = getMaxMIDiscr(discr)
        for i in range(img.size[1]):    # for every pixel:
            for j in range(img.size[0]):
                greylevel = int(255 * (1.0 - float(getMIRefDiscr(discr, i * w + j)) / float(maxval)))
                pixels[j,i] =  (greylevel, greylevel, greylevel)
        #img.show()
        return img
    else:
        raise Exception("image dims not compatible with mental image size")

def getMIRawDiscr(discr,w, h):
    if (w * h) == getSizeDiscr(discr):
        img = Image.new('RGB', (w, h), "white")
        pixels = img.load()
        maxval = getMaxMIDiscr(discr)
        for i in range(img.size[1]):    # for every pixel:
            for j in range(img.size[0]):
                greylevel = int(255 - getMIRefDiscr(discr, i * w + j))
                pixels[j,i] =  (greylevel, greylevel, greylevel)
        #img.show()
        return img
    else:
        raise Exception("image dims not compatible with mental image size")
    
def conftable(dl,results):
    nclasses = len(dl)
    table = [[0 for i in range(nclasses+1)] for j in range(nclasses+1)]
    h = [0 for i in range(nclasses)]
    #recalls = [0.0 for i in range(nclasses)]
    w = [0 for i in range(nclasses)]
    #precisions = [0.0 for i in range(nclasses)]
    m1 = 0.0
    m2 = 0.0
    TP = FP = 0
    for res in results:
        gtclass = res[0]
        foundclass = res[1][0]
        if gtclass == foundclass:
            TP += 1
        else:
            FP += 1
        table[gtclass][foundclass] += 1
    for k in range(nclasses):
        for j in range(nclasses):
            h[j] += table[k][j]
            w[k] += table[k][j]
    for k in range(nclasses):
        # compute and store in last column recalls
        table[k][nclasses] =  table[k][k] / float(w[k]) if w[k] > 0 else 0
        # compute and store in last row precisions 
        table[nclasses][k] = table[k][k] / float(h[k]) if h[k] > 0 else 0
    for k in range(nclasses):
        m1 += table[k][nclasses]
        m2 += table[nclasses][k]
    fmeasure = ((m1 / float(nclasses)) +  (m2 / float(nclasses))) / 2.0
    table[nclasses][nclasses] = fmeasure
    # prin confusion matrix
    print
    for k in range(nclasses):
        print "|",
        for j in range(nclasses):
            print "%8d, " % (table[k][j]),
        print "|   %1.6f" % (table[k][nclasses])
    print
    print "|",
    for j in range(nclasses):
        print "%1.6f, " % (table[nclasses][j]),
    print "|   %1.6f" % (table[nclasses][nclasses])
    #return table, TP, FP, fmeasure
    return

def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f

def k(a,b):
    def _k(item):
        return (item[a],item[b])
    return _k

def prototypes(discr, path, w, h):
    if (w * h) == getSizeDiscr(discr):
        mi = getMIDiscr(discr)
        name = getNameDiscr(discr)
        size = getSizeDiscr(discr)
        maxval = int(getMaxMIDiscr(discr))
        pixels = [int(getMIRefDiscr(discr, i)) for i in range(size)]
        bv = [[1 if i < b else 0 for i in range(maxval)] for b in pixels]
        [random.shuffle(b) for b in bv]
        progress = 1
        for l in zip(*bv):
            img = Image.new('1', (w, h))
            imgpixs = img.load()
            #imgpixs = list(img.getdata())
            for i in range(img.size[0]):    # for every pixel:
                for j in range(img.size[1]):
                    greylevel = (1 - l[j * w + i]) * 255
                    imgpixs[i,j] = greylevel
                    # imgpixs[j * w + i] = greylevel
            img.save(os.path.join(path,"%s" % (name), "%s-%04d.png" % (name,progress)), format="PNG")
            progress += 1
        return maxval
    else:
        raise Exception("image dims not compatible with mental image size")
                  
#print "WiSARD library (%s) loaded!" % (os.path.basename(libpath))
