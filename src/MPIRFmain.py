# coding=UTF-8

from DataClass.Reader.MDFReader import *
from DataClass.Reader.JReader import *
from PreprocessClass.MDFPreprocess import *
from ReconClass.MRecon import *
from matplotlib.pylab import *

from PhantomClass.PPhantom import *
from PhantomClass.EPhantom import *
from DataClass.Scanner.XScanner import *
from ReconClass.XRecon import *

from DataClass.Scanner.MScanner import *

from ReconClass.BaseClass.Imager import *

from PostprocessClass.Postprocess import *

import os

'''
MPIRFmain.py: Call the MPIRF module functions.
'''

class JsonDefaultEnconding(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, complex):
            return "{real}+{image}i".format(real=o.real, image=o.real)
        if isinstance(o, np.ndarray):
            return o.tolist()

'''
Access measurement data from file and reconstrution
#Arguments
*FileType:     the file type(JSON OR MDF).
*MsPath:       the file path of system matrix data.(only be used by when FileType is MDF)
*MmPath:       the file path of measurement data.
*MethodType:   the reconstruction method type.(system matrix or x-space)
*itera:        the number of iterations.(only be used by when FileType is MDF)
*theta:        the regularization parameter.(only be used by when FileType is MDF)
*ThresholdSeg: the image threshold segmentation parameter.
'''
def FileMain(FileType,MsPath,MmPath,MethodType,itera,theta,ThresholdSeg):

    if FileType==0:
        try:
            # Call MDFReader Component to access the MDF file.
            phan = MDFReaderClass(MsPath, MmPath)
        except:
            return 1,None,None,None,None,None,None,None,None,None

        try:
            # Call Pre-processing module to realize the pre-processing and verification of message dictionary data
            MDFPreprocessClass(phan.Message)
        except:
            return 2,None,None,None,None,None,None,None,None,None

    elif FileType==1:

        try:
            # Call MDFReader Component to access the JSON file.
            phan = JReaderClass(MmPath)
        except:
            return 1, None, None, None, None, None, None, None, None, None

        try:
            # Call Pre-processing module to realize the pre-processing and verification of message dictionary data
            MDFPreprocessClass(phan.Message)
        except:
            return 2, None, None, None, None, None, None, None, None, None

    try:
        if MethodType==0:
            # Call System Matrix component to realize image reconstruction
            ImgData = MReconClass(phan.Message, itera,np.linalg.norm(phan.Message[MEASUREMENT][AUXSIGNAL], ord='fro') * theta)
        elif MethodType==1:
            # Call X-Space component to realize image reconstruction
            ImgData= XReconClass(phan.Message)
        else:
            raise Exception
    except:
        return 3, None, None, None, None, None, None, None, None, None

    try:
        # Call Post-processing module to realize image gray-level analysis and image threshold segmentation
        return 0,\
           ImgData.get_ImagSiganl()[1][0], \
           ImgData.get_ImagSiganl()[1][1], \
           ImgData.get_ImagSiganl()[1][2], \
           PostprocessClass(ImgData.get_ImagSiganl()[1][0]).get_GrayLevel(), \
           PostprocessClass(ImgData.get_ImagSiganl()[1][1]).get_GrayLevel(), \
           PostprocessClass(ImgData.get_ImagSiganl()[1][2]).get_GrayLevel(), \
           PostprocessClass(ImgData.get_ImagSiganl()[1][0]).get_ThresholdSeg(ThresholdSeg), \
           PostprocessClass(ImgData.get_ImagSiganl()[1][1]).get_ThresholdSeg(ThresholdSeg), \
           PostprocessClass(ImgData.get_ImagSiganl()[1][2]).get_ThresholdSeg(ThresholdSeg)
    except:
        return 4, None, None, None, None, None, None, None, None, None

'''
Simulate measurement data from file and reconstrution
#Arguments
*MethodType:       the reconstruction method type.(system matrix or x-space)
*PhanType:         the phantom type.(P shape phantom or E shape phantom)
*Temperature:      the particle temperature.
*Diameter:         the diameter of the particle core.
*MagSaturation:    the saturation magnetization of the particle.
*Concentration:    the particle concentration of phantom.
*SelectGradietX:   the gradient strength of the selection field in the x-direction.
*SelectGradietY:   the gradient strength of the selection field in the y-direction.
*DriveFrequencyX:  the frequency of the drive field strength in the x-direction.
*DriveFrequencyY:  the frequency of the drive field strength in the y-direction.
*DriveAmplitudeX:  the amplitude of the drive field strength in the x-direction.
*DriveAmplitudeY:  the amplitude of the drive field strength in the y-direction.
*RepetitionTime:   the repeat time of trajectory one period.
*SampleFrequency:  the frequency of sampling.
*itera:            the number of iterations.(only be used by when FileType is MDF)
*theta:            the regularization parameter.(only be used by when FileType is MDF)
*ThresholdSeg:     the image threshold segmentation parameter.
'''
def SimulationMain(MethodType,
                   PhanType,Temperature,Diameter,MagSaturation,Concentration,
                   SelectGradietX,SelectGradietY,SelectGradietZ,
                   DriveFrequencyX,DriveFrequencyY,DriveFrequencyZ,DriveAmplitudeX,DriveAmplitudeY,DriveAmplitudeZ,
                   RepetitionTime,SampleFrequency,
                   Itera,Theta,Delta,ThresholdSeg):

    try:
        if PhanType==0:
            # Create a P shape phantom.
            phan = PPhantomClass(Temperature,Diameter,MagSaturation,Concentration)
        elif PhanType==1:
            # Create a E shape phantom. 
            # Not modified yet
            phan = EPhantomClass(Temperature, Diameter, MagSaturation, Concentration)
    except:
        return 1, None, None, None, None, None, None

    if MethodType==0:
        try:
            # Call System-matrix base scanner Component to simulate system matrix scanner.
            scanner = MScannerClass(phan,
                                SelectGradietX, SelectGradietY,SelectGradietZ,
                                DriveFrequencyX, DriveFrequencyY, DriveFrequencyZ,DriveAmplitudeX, DriveAmplitudeY,DriveAmplitudeZ,
                                RepetitionTime, SampleFrequency,Delta)
        except:
            return 1, None, None, None, None, None, None

        try:
            # Call Pre-processing module to realize the pre-processing and verification of message dictionary data
            PreprocessClass(scanner.Message)
        except:
            return 2, None, None, None, None, None, None

        try:
            # Call System Matrix component to realize image reconstruction
            ImgData = MReconClass(scanner.Message,Itera,Theta)
            ResultImage = ImgData.get_ImagSiganl()[1][0]
        except:
            return 3, None, None, None, None, None, None

    elif MethodType==1:
        try:
            # Call X-Space base scanner Component to simulate system matrix scanner.
            scanner = XScannerClass(phan,
                              SelectGradietX,SelectGradietY,
                              DriveFrequencyX,DriveFrequencyY,DriveAmplitudeX,DriveAmplitudeY,
                              RepetitionTime,SampleFrequency)
        except:
            return 1, None, None, None, None, None, None

        try:
            # Call Pre-processing module to realize the pre-processing and verification of message dictionary data
            PreprocessClass(scanner.Message)
        except:
            return 2, None, None, None, None, None, None

        try:
            # Call X-Space component to realize image reconstruction
            ImgData = XReconClass(scanner.Message)
            ResultImage = ImgData.get_ImagSiganl()[1][0]
        except:
            return 3, None, None, None, None, None, None



    try:
        # Call Post-processing module to realize image gray-level analysis and image threshold segmentation
        return 0,\
           phan.get_PhantomMatrix(), \
           ResultImage, \
           PostprocessClass(ResultImage).get_GrayLevel(), \
           scanner.Message[MEASUREMENT][MEANUMBER][0], \
           scanner.Message[MEASUREMENT][MEANUMBER][1], \
           PostprocessClass(ResultImage).get_ThresholdSeg(ThresholdSeg)
    except:
        return 4, None, None, None, None, None, None

def DelFiles(path_file):
    ls = os.listdir(path_file)
    for i in ls:
        f_path = os.path.join(path_file, i)
        if os.path.isdir(f_path):
            DelFiles(f_path)
        else:
            os.remove(f_path)

if __name__ == "__main__":
    print("*********************************************************")
    print("1: SystemMatrix Reconstruction based on MDF File")
    print("2: SystemMatrix Reconstruction based on Json File")
    print("3: X-Space Reconstruction based on Json File")
    print("4: Simulation of MPI scan based on SystemMatrix")
    print("5: Simulation of MPI scan based on X-Space")
    print("Q: Quit")
    print("*********************************************************")
    instr=''
    instr = input("Please enter the Number: ")
    flag = 0
    while not instr=='Q':
        if instr=='1':
            MsPath = input("Please enter the SystemMatrix File Path: ")
            MmPath = input("Please enter the Measurement File Path: ")
            print("Reconstructing...")
            flag,x,y,z,x1,y1,z1,x2,y2,z2 = FileMain(0, MsPath, MmPath, 0, 1, 1, 150)
            if flag!=0:
                break
            DelFiles("./TempImage/")
            ImagerClass.WriteImage(x, "./TempImage/", "XY.jpg")
            ImagerClass.WriteImage(y, "./TempImage/", "XZ.jpg")
            ImagerClass.WriteImage(z, "./TempImage/", "YZ.jpg")
            ImagerClass.WriteImage(x1, "./TempImage/", "XY GrayLevel.jpg")
            ImagerClass.WriteImage(y1, "./TempImage/", "XZ GrayLevel.jpg")
            ImagerClass.WriteImage(z1, "./TempImage/", "YZ GrayLevel.jpg")
            ImagerClass.WriteImage(x2, "./TempImage/", "XY Segmetion.jpg")
            ImagerClass.WriteImage(y2, "./TempImage/", "XZ Segmetion.jpg")
            ImagerClass.WriteImage(z2, "./TempImage/", "YZ Segmetion.jpg")
            print("The result is written into the path:" + os.getcwd() + "/TempImage/")
            break
        elif instr=='2':
            MsPath = None
            MmPath = input("Please enter the JSON File Path: ")
            print("Reconstructing...")
            flag,x,y,z,x1,y1,z1,x2,y2,z2 = FileMain(1, MsPath, MmPath, 0, 1, 1, 150)
            if flag!=0:
                break
            DelFiles("./TempImage/")
            ImagerClass.WriteImage(x, "./TempImage/", "XY.jpg")
            ImagerClass.WriteImage(y, "./TempImage/", "XZ.jpg")
            ImagerClass.WriteImage(z, "./TempImage/", "YZ.jpg")
            ImagerClass.WriteImage(x1, "./TempImage/", "XY GrayLevel.jpg")
            ImagerClass.WriteImage(y1, "./TempImage/", "XZ GrayLevel.jpg")
            ImagerClass.WriteImage(z1, "./TempImage/", "YZ GrayLevel.jpg")
            ImagerClass.WriteImage(x2, "./TempImage/", "XY Segmetion.jpg")
            ImagerClass.WriteImage(y2, "./TempImage/", "XZ Segmetion.jpg")
            ImagerClass.WriteImage(z2, "./TempImage/", "YZ Segmetion.jpg")
            print("The result is written into the path:" + os.getcwd() + "/TempImage/")
            break
        elif instr=='3':
            MsPath = None
            MmPath = input("Please enter the JSON File Path: ")
            print("Reconstructing...")
            flag,x, y, z, x1, y1, z1, x2, y2, z2 = FileMain(1, MsPath, MmPath, 1, 1, 1, 150)
            if flag!=0:
                break
            DelFiles("./TempImage/")
            ImagerClass.WriteImage(x, "./TempImage/", "XY.jpg")
            ImagerClass.WriteImage(y, "./TempImage/", "XZ.jpg")
            ImagerClass.WriteImage(z, "./TempImage/", "YZ.jpg")
            ImagerClass.WriteImage(x1, "./TempImage/", "XY GrayLevel.jpg")
            ImagerClass.WriteImage(y1, "./TempImage/", "XZ GrayLevel.jpg")
            ImagerClass.WriteImage(z1, "./TempImage/", "YZ GrayLevel.jpg")
            ImagerClass.WriteImage(x2, "./TempImage/", "XY Segmetion.jpg")
            ImagerClass.WriteImage(y2, "./TempImage/", "XZ Segmetion.jpg")
            ImagerClass.WriteImage(z2, "./TempImage/", "YZ Segmetion.jpg")
            print("The result is written into the path:" + os.getcwd() + "/TempImage/")
            break
        elif instr=='4':
            print("Use default parameters,Simulating...")
            flag,P,I,G,X,Y,T=SimulationMain(0,0, 20, 30e-9, 8e5, 5e7,
                           2.0, 2.0,
                           2500000.0 / 102.0, 2500000.0 / 96.0, 12e-3, 12e-3,
                           6.528e-4, 2.5e6,
                           1, 1, 5e7, 150)
            if flag!=0:
                break
            DelFiles("./TempImage/")
            ImagerClass.WriteImage(P, "./TempImage/", "Phantom.jpg")
            ImagerClass.WriteImage(I, "./TempImage/", "Reconstruction Result.jpg")
            ImagerClass.WriteImage(G, "./TempImage/", "GrayLevel histogram.jpg")
            ImagerClass.WriteImage(T, "./TempImage/", "Threshold Segmetion.jpg")
            print("The result is written into the path:" + os.getcwd() + "/TempImage/")
            break
        elif instr=='5':
            print("Use default parameters,Simulating...")
            flag,P,I,G,X,Y,T=SimulationMain(1, 0, 20, 30e-9, 8e5, 5e7,
                                       2.0, 2.0,
                                       2500000.0 / 102.0, 2500000.0 / 96.0, 12e-3, 12e-3,
                                       6.528e-4, 2.5e6,
                                       1, 1, 5e7, 150)

            if flag!=0:
                break
            DelFiles("./TempImage/")
            ImagerClass.WriteImage(P, "./TempImage/", "Phantom.jpg")
            ImagerClass.WriteImage(I, "./TempImage/", "Reconstruction Result.jpg")
            ImagerClass.WriteImage(G, "./TempImage/", "GrayLevel histogram.jpg")
            ImagerClass.WriteImage(T, "./TempImage/", "Threshold Segmetion.jpg")
            print("The result is written into the path:" + os.getcwd() + "/TempImage/")
            break
        else:
            instr = input("Wrong input, Please re-enter:")

    if flag == 1:
        print("DATA ACCESS module is abnormal.")
    elif flag == 2:
        print("DATA PREPROCESSING module is abnormal.")
    elif flag == 3:
        print("IMAGE RECONSTRUCTION module is abnormal.")
    elif flag == 4:
        print("IMAGE POSTPROCESSING module is abnormal.")