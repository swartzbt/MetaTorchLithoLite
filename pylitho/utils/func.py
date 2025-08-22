import torch
from torch import fft
import pickle
import os
from pathlib import Path
import hashlib

workspace = ".OpenLitho"

def getMaskFFT(mask):
    return fft.fftshift(fft.fft2(mask))

def getK(wavelength):
    return 2 * torch.pi / wavelength

def getHomePath():
    return str(Path.home())

def buildTheWorkSpace():
    path = getHomePath() + "/" + workspace
    if not os.path.exists(path):
        os.makedirs(path)
        
def getWorkspacePath():
    path =  getHomePath() + "/" + workspace
    if not os.path.exists(path):
        buildTheWorkSpace()
    return path

def nameHashing(layout : str, layer : int, pixel : int):
    md5 = hashlib.md5()
    md5.update(bytes(f"{layout}_{layer}_{pixel}", 'utf-8'))
    return md5.hexdigest()

def buildFolderInWorkspace(name : str):
    path = getWorkspacePath() + "/" + name
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def getFolderInWorkspace(name : str):
    path = getWorkspacePath() + "/" + name
    if not os.path.exists(path):
        buildFolderInWorkspace(name)
    return path