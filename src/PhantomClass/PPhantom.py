# coding=UTF-8
import numpy as np
from Config.ConstantList import *
from PhantomClass.BassClass.Phantom import *
from vedo import Volume, dataurl, show
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tif
import os 
import cv2

'''
PPhantom.py: The phantom class of P shape.
'''

class PPhantomClass(PhantomClass):
    def __init__(self,Temperature=20.0,Diameter=30e-9,MagSaturation=8e5, Concentration=5e7 ):

        super().__init__(Temperature,
                         Diameter,
                         MagSaturation,
                         Concentration)

    # Return the matrix of the 'P' phantom image.
    def _get_Picture(self,Concentration,Xn,Yn,Zn):
        self._Xn = Xn
        self._Yn = Yn
        self._Zn = Zn
        C = np.zeros((self._Zn,self._Xn, self._Yn))
        for i in range(self._Zn):
            C[i,int(Xn * (14 / 121)):int(Xn * (105 / 121)), int(Yn * (29 / 121)):int(Yn * (90 / 121))] = np.ones(
                (int(Xn * (105 / 121)) - int(Xn * (14 / 121)), int(Yn * (90 / 121)) - int(Yn * (29 / 121))))
            C[i,int(Xn * (29 / 121)):int(Xn * (60 / 121)), int(Yn * (44 / 121)):int(Yn * (75 / 121))] = np.zeros(
                (int(Xn * (60 / 121)) - int(Xn * (29 / 121)), int(Yn * (75 / 121)) - int(Yn * (44 / 121))))
            C[i,int(Xn * (74 / 121)):int(Xn * (105 / 121)), int(Yn * (44 / 121)):int(Yn * (90 / 121))] = np.zeros(
                (int(Xn * (105 / 121)) - int(Xn * (74 / 121)), int(Yn * (90 / 121)) - int(Yn * (44 / 121))))

        # folder_path = '/Users/lonqi/Downloads/Stanford volume data 8bit/'
        # file_names = [f for f in os.listdir(folder_path)]
        # file_names = sorted(file_names)
        # volume = np.ndarray((121,121,121))
        # for (i,file_name) in enumerate(file_names):
        #     slice = tif.imread(folder_path + file_name)
        #     slice = cv2.resize(slice,(121,121))
        #     # for m in range(121):
        #     #     for n in range(121):
        #             # slice[m,n] = np.sqrt((m - 60.5)**2 + (n - 60.5)**2)
        #     volume[:,:,i] = slice
        # C = volume
        # fig = plt.figure()
        # plt.imshow(np.sum(C,axis=0))
        # plt.show()
        
            


        # vol = Volume(volume)


        # show(vol, __doc__, axes=1).close()

        # from vedo import Volume,show
        # vol = Volume(C)
        # show(vol)
        # C = C.transpose()
        return C * Concentration