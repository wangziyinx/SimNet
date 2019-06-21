import ctypes
import numpy as np
import tensorflow as tf
from numpy.ctypeslib import ndpointer
lib = ctypes.cdll.LoadLibrary('C:\\Users\\wangz\\PycharmProjects\\ClustDLL\\x64\\Release\\ClustDLL.dll')

class clust(object):
    def __init__(self, r, th, dim, numc):
        lib.clust_new.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int]
        lib.clust_new.restype = ctypes.c_void_p

        lib.clust_feed.argtypes = [ctypes.c_void_p,
                                   ctypes.c_int, ctypes.c_int, ctypes.c_void_p]

        lib.num_cluster.argtypes = [ctypes.c_void_p]
        lib.num_cluster.restype = ctypes.c_int

        lib.clust_get.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

        self.handle = lib.clust_new(r, th, dim, numc)
        self.dim = dim

    def feed (self, D):
        #D numpy array
        lib.clust_feed(ctypes.c_void_p(D.ctypes.data), D.shape[0], D.shape[1], self.handle)

    def get (self):
        nc = lib.num_cluster(self.handle)
        print (nc)
        C = np.zeros((nc, self.dim), dtype= np.double)
        lib.clust_get(ctypes.c_void_p(C.ctypes.data), self.handle)
        return C
#
#
# data = np.genfromtxt('C:\\Users\\wangz\\MyDatabase\\Cluster_Dataset\\S_set\\s1_Noised0.txt', delimiter = ' ')
# # data = data.astype(np.double)
# mm = np.mean(data, axis = 0)
# data[:,0] = data[:,0]- mm[0]
# data[:,1] = data[:,1]- mm[1]
# ss = np.multiply(data, data)
# ss = np.sum(ss, axis = 1)
# ss = np.sqrt(ss)
# data = (data.T*(1/ss)).T
#
# r = data.shape[0]
# c = data.shape[1]
# cc = clust(0.1, 0.97, 2)
# cc.feed(data)
# nc = lib.num_cluster(cc.handle)
# print (nc)
# c = cc.get()