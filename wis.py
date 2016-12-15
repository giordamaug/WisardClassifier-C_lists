"""
    WiSARD Classifier in Scikit-Learn Python Package

    Created by Maurizio Giordano on 13/12/2016

"""
import sys,os
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import scipy.sparse as sps
from wisard import *
import time
import random

mypowers= [1L, 2L, 4L, 8L, 16L, 32L, 64L, 128L, 256L, 512L, 1024L, 2048L, 4096L, 8192L, 16384L, 32768L, 65536L, 131072L , 262144L, 524288L,
           1048576L, 2097152L, 4194304L, 8388608L, 16777216L, 33554432L, 67108864L, 134217728L, 268435456L, 536870912L, 1073741824L, 2147483648L,
           4294967296L, 8589934592L, 17179869184L, 34359738368L, 68719476736L, 137438953472L, 274877906944L, 549755813888L, 1099511627776L, 2199023255552L, 4398046511104L, 8796093022208L, 17592186044416L, 35184372088832L, 70368744177664L, 140737488355328L, 281474976710656L, 562949953421312L, 1125899906842624L, 2251799813685248L, 4503599627370496L, 9007199254740992L, 18014398509481984L, 36028797018963968L, 72057594037927936L, 144115188075855872L, 288230376151711744L, 576460752303423488L, 1152921504606846976L, 2305843009213693952L, 4611686018427387904L, 9223372036854775808L]

class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[0;32m'
    WHITEBLACK = '\033[1m\033[40;37m'
    BLUEBLACK = '\033[1m\033[40;94m'
    YELLOWBLACK = '\033[1m\033[40;93m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def printProgressBar(label,time,etime,basecolor, cursorcolor, linecnt,progress,size):
    barwidth = 70
    progress = linecnt / float(size);
    str = '%s |' % label
    pos = int(barwidth * progress)
    str += basecolor
    for p in range(barwidth):
        if p < pos:
            str += u'\u2588'
        elif p == pos:
            str += color.END + cursorcolor + u'\u2588' + color.END + basecolor
        else:
            str += u'\u2591'
    str += color.END + '| ' + "{:>3}".format(int(progress * 100.0)) + ' % ' + color.YELLOWBLACK + ' ' + etime + ' ' + color.WHITEBLACK + time + ' ' + color.END
    sys.stdout.write("\r%s" % str)
    sys.stdout.flush()
    return progress

def compTime(deltatime,progress):
    hours, rem = divmod(deltatime*((1.0-progress) / progress), 3600)
    hourse, reme = divmod(deltatime, 3600)
    minutes, seconds = divmod(rem, 60)
    minutese, secondse = divmod(reme, 60)
    tm = "{:0>2}:{:0>2}:{:02.0f}".format(int(hours),int(minutes),seconds)
    tme = "{:0>2}:{:0>2}:{:02.0f}".format(int(hourse),int(minutese),secondse)
    return tm,tme

class WIS(BaseEstimator, ClassifierMixin):
    """Wisard Classifier."""
    wiznet_ = {}
    ranges_ = []
    offsets_ = []
    rowcounter_ = 0
    progress_ = 0.0
    starttm_ = 0
    def __init__(self,nobits=8,notics=256,coding='histo',mapping='random',debug=False,bleaching=False,default_bleaching=1,confidence_bleaching=0.1):
        if (not isinstance(nobits, int) or nobits<1 or nobits>64):
            raise Exception('number of bits must be an integer between 1 and 64')
        if (not isinstance(notics, int) or notics<1):
            raise Exception('number of bits must be an integer greater than 1')
        if (not isinstance(bleaching, bool)):
            raise Exception('bleaching flag must be a boolean')
        if (not isinstance(debug, bool)):
            raise Exception('debug flag must be a boolean')
        if (not isinstance(default_bleaching, int) or nobits<1):
            raise Exception('bleaching downstep must be an integer greater than 1')
        if (not isinstance(mapping, str) or (not (mapping=='random' or mapping=='linear'))):
            raise Exception('mapping must either \"random\" or \"linear\"')
        if (not isinstance(coding, str) or (not (coding=='histo' or mapping=='cursor' or mapping=='binary'))):
            raise Exception('random seed must be an integer')
        if (not isinstance(confidence_bleaching, float)):
            raise Exception('bleaching confidence must be a float between 0 and 1')
        self.nobits = nobits
        self.notics = notics
        self.mapping = mapping
        self.bleaching = bleaching
        self.default_bleaching = default_bleaching
        self.confidence_bleaching = confidence_bleaching
        self.coding = coding
        self.debug = debug
        pass
    def fit(self, X, y):
        self.classes_, y = np.unique(y, return_inverse=True)
        self.size_,self.nfeatures_ = X.shape
        if sps.issparse(X):
            self.ranges_ = (X.max(axis=0)-X.min(axis=0)).toarray()[0]
            self.offsets_ = X.min(axis=0).toarray()[0]
            X = X.toarray()
        else:
            self.ranges_ = (X.max(axis=0)-X.min(axis=0))
            self.offsets_ = X.min(axis=0)
        for cl in self.classes_:
            self.wiznet_[cl] = makeDiscr(self.nobits, self.notics * self.nfeatures_, str(cl), self.mapping);
        cnt = 0
        self.progress_ = 0.01
        self.starttm_ = time.time()
        for row in X:
            data = row
            trainSvmHistoDiscr(self.wiznet_[self.classes_[y[cnt]]],data,self.ranges_,self.offsets_,self.notics,self.nfeatures_)
            cnt += 1
            tm,tme = compTime(time.time()-self.starttm_,self.progress_)
            if self.debug:
                self.progress_ = printProgressBar('train', tm, tme, color.BLUE, color.RED, cnt,self.progress_,len(X))
        if self.debug:
                sys.stdout.write('\n')
        return self
    def predict(self, X):
        if sps.issparse(X):
            X = X.toarray()
        D = self.decision_function(X)
        return self.classes_[np.argmax(D, axis=1)]
    def decision_function(self,X):
        D = np.empty(shape=[0, len(self.classes_)])
        cnt = 0
        self.starttm_ = time.time()
        self.progress_ = 0.01
        for row in X:
            data = row
            res = [classifySvmHistoDiscr(self.wiznet_[cl],data,self.ranges_,self.offsets_,self.notics,self.nfeatures_)
                for cl in self.classes_]
            D = np.append(D, [res],axis=0)
            cnt += 1
            tm,tme = compTime(time.time()-self.starttm_,self.progress_)
            if self.debug:
                self.progress_ = printProgressBar('test ',tm,tme,color.GREEN, color.RED, cnt,self.progress_,len(X))
        if self.debug:
            sys.stdout.write('\n')
        return D
    def get_params(self, deep=True):
        # suppose this estimator has parameters "alpha" and "recursive"
        return {"nobits": self.nobits, "notics": self.notics, "coding": self.coding,"mapping": self.mapping, "debug": self.debug}
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    def mksynt(self, file):    
        # alloc & init new dataset
        trainsize=0
        for cl in self.classes_:
            trainsize += getTcounterDiscr(self.wiznet_[cl])
        # alloc & init new dataset
        newdata = np.zeros(shape=(trainsize,self.nfeatures_ + 1))

        # create dataset
        ntplus = self.notics +1
        #with open(pathname, "w") as f:
        offset=0
        for cl in self.classes_:
            ntrain = getTcounterDiscr(self.wiznet_[cl])
            mentalDiscr(self.wiznet_[cl])
            pmi = np.zeros(ntplus * self.nfeatures_)
            # compute P Mental Image
            maxvalue = 0
            for x in range(self.nfeatures_):
                for y in range(1,ntplus)[::-1]:
                    pmi[x * ntplus + y] = int(getMIRefDiscr(self.wiznet_[cl],x * self.notics + (y - 1)))
                for y in range(1,ntplus)[::-1]:
                    if pmi[x * ntplus + y] > maxvalue:
                        maxvalue = pmi[x * ntplus + y]
                    if pmi[x * ntplus + y] > 0:
                        for k in range(y)[::-1]:
                            pmi[x * ntplus + k] -= pmi[x * ntplus + y]
                pmi[x * ntplus] = getTcounterDiscr(self.wiznet_[cl]) + pmi[x * ntplus]
            # generate dataset
            for x in range(self.nfeatures_):
                indices = range(ntplus)
                kindices = range(ntrain)
                random.shuffle(indices)
                random.shuffle(kindices)
                k = 0
                for kk in range(ntplus):
                    while pmi[(x * (ntplus)) + indices[kk]] > 0:
                        newdata[offset+kindices[k]][0] = cl
                        pmi[(x * (ntplus)) + indices[kk]] -=  1
                        newdata[offset+kindices[k]][x+1] = float(float(indices[kk]) * self.ranges_[x] / self.notics) + self.offsets_[x]
                        k += 1
            offset += ntrain
        # write dataset
        for sampleidx in range(trainsize):
            file.write("%d"%newdata[sampleidx][0])
            for x in range(self.nfeatures_):
                file.write(" %d:%f"%(x,newdata[sampleidx][x+1]))
            file.write("\n")

