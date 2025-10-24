import pickle
from typing import Union, List

import cv2
import numpy as np
from sklearn.utils.extmath import randomized_svd

SINMAX = 0.9375  # Don't know what this is
MAX_TCC_SIZE = 64


def getFreqs(pixel, canvas):
    """
    Get the frequency space sample points that correspond to given spatial sampling pitch and extent.
    :param pixel: size of 1 pixel in spatial domain
    :param canvas: size of entire simulation region
    :return: (np.array, np.array), frequency-space sample point frequencies.
    """
    size = int(round(canvas / pixel))
    basic = np.fft.fftshift(np.fft.fftfreq(size, d=pixel))
    freqX = basic.reshape((1, -1)).repeat(size, axis=0)
    freqY = basic.reshape((-1, 1)).repeat(size, axis=1)
    assert freqX.shape[0] == size and freqX.shape[1] == size
    assert freqY.shape[0] == size and freqY.shape[1] == size
    return freqX, freqY


def srcPoint(pixel, canvas):
    freqX, freqY = getFreqs(pixel, canvas)
    result = (freqX == 0) * (freqY == 0)
    return result.astype(np.double)


def funcPupil(pixel, canvas, na, lam, defocus=None, refract=None):
    """
    Transfer function of an imaging system with specified numerical aperture and defocus
    :param pixel: size of 1 pixel in spatial domain
    :param canvas: size of entire simulation region
    :param na: Numerical aperture
    :param lam: Wavelength
    :param defocus: defocus distance in same units as wavelength
    :param refract: Don't know what this is. If this is 1, then result is a normal transfer function with defocus
    :return:
    """
    limit = na / lam  # Maximum supported spatial frequency
    freqX, freqY = getFreqs(pixel, canvas)  # Frequency sample points
    result = np.sqrt(freqX**2 + freqY**2) < limit
    result = result.astype(np.double)
    if not defocus is None: 
        assert not refract is None
        print(f"{np.max(freqX), np.max(freqY)}")
        mask = result > 0
        opd = defocus * (refract - np.sqrt(refract**2 - lam**2 * ((freqX*mask)**2 + (freqY*mask)**2)))
        shift = np.exp(1j * (2 * np.pi / lam) * opd)
        result = result * shift
    return result


def TCC(src, pupil, pixel, canvas, thresh=1.0e-6):
    """

    :param src: source matrix; intensity of the source in frequency space
    :param pupil: transfer function of the imaging system (frequency space)
    :param pixel: size of 1 pixel in spatial domain
    :param canvas: size of entire simulation region
    :param thresh: minimum singular value that will be retained
    :return:
    """
    size = round(canvas / pixel)
    pupilFFT = np.fft.fftshift(np.fft.fft2(pupil)) # h
    pupilStar = pupilFFT.conj() # h*
    srcFFT = np.fft.fftshift(np.fft.fft2(src/np.sum(src))) # J
    print(f"Creating big matrix: {pupilStar.shape + pupilStar.shape}")
    w = np.zeros(pupilStar.shape + pupilStar.shape, dtype=np.complex64)
    for idx in range(pupilStar.shape[0]):
        for jdx in range(pupilStar.shape[1]):
            srcShifted = np.roll(srcFFT, shift=(idx, jdx), axis=(0, 1))
            srcShifted = np.flip(srcShifted, axis=(0, 1))
            w[idx, jdx] = srcShifted * pupilFFT[idx, jdx] * pupilStar / (np.prod(pupil.shape) * np.prod(src.shape))
    sizeAll = np.prod(pupilStar.shape)
    w = w.reshape(sizeAll, sizeAll)
    print(f"Running SVD for matrix {w.shape}")
    matU, matS, matVT = randomized_svd(w, n_components=64, n_iter=32, n_oversamples=16)
    print(f"SVD results: {matU.shape, matS.shape, matVT.shape}, weights = {matS}")
    phis = []
    weights = []
    for idx, weight in enumerate(matS):
        if not thresh is None and weight >= thresh:
            phis.append(matU[:, idx].reshape(size, size) * (size*size))
            weights.append(matS[idx])
    return phis, weights


def genTCC(pixel: int,
           canvas: int,
           na: float,
           wavelength: int,
           defocus: Union[None, List[int]] = None,
           thresh: float = 1.0e-6):
    refract = na / SINMAX
    size = canvas//pixel
    if defocus is not None:
        phis, weights = [], []
        for d in defocus:
            if size <= MAX_TCC_SIZE: 
                pupil = funcPupil(pixel, canvas, na, wavelength, defocus=d, refract=refract)
                circ = srcPoint(pixel, canvas)
                _phis, _weights = TCC(circ, pupil, pixel, canvas, thresh=thresh)
            else: 
                tccpixel = pixel
                tcccanvas = canvas
                resize = 1
                padding = 1
                while tcccanvas//tccpixel > MAX_TCC_SIZE: 
                    if tcccanvas > 2048: 
                        tcccanvas //= 2
                        resize *= 2
                    else: 
                        tccpixel *= 2
                        padding *= 2
                pupil = funcPupil(tccpixel, tcccanvas, na, wavelength, refract=refract)
                circ = srcPoint(tccpixel, tcccanvas)
                _phis, _weights = TCC(circ, pupil, tccpixel, tcccanvas, thresh=thresh)
                tccsize = tcccanvas//tccpixel
                for idx in range(len(_phis)): 
                    padded = tccsize*padding
                    ffted = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(_phis[idx])))
                    realpart = np.zeros((padded, padded))
                    imagpart = np.zeros((padded, padded))
                    begin = (padded - tccsize) // 2
                    end = begin + tccsize
                    realpart[begin:end, begin:end] = ffted.real
                    imagpart[begin:end, begin:end] = ffted.imag
                    ffted = realpart + 1j * imagpart
                    realfft = cv2.resize(ffted.real, (size, size), interpolation=cv2.INTER_LINEAR)
                    imagfft = cv2.resize(ffted.imag, (size, size), interpolation=cv2.INTER_LINEAR)
                    ffted = realfft + 1j * imagfft
                    _phis[idx] = padding**2 * resize**2 * np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(ffted)))
            phis.append(_phis)
            weights.append(_weights)
        return phis, weights
    else:
        if size <= MAX_TCC_SIZE: 
            pupil = funcPupil(pixel, canvas, na, wavelength, refract=refract)
            circ = srcPoint(pixel, canvas)
            phis, weights = TCC(circ, pupil, pixel, canvas, thresh=thresh)
        else: 
            tccpixel = pixel
            tcccanvas = canvas
            resize = 1
            padding = 1
            while tcccanvas//tccpixel > MAX_TCC_SIZE: 
                if tcccanvas > 2048: 
                    tcccanvas //= 2
                    resize *= 2
                else: 
                    tccpixel *= 2
                    padding *= 2
            pupil = funcPupil(tccpixel, tcccanvas, na, wavelength, refract=refract)
            circ = srcPoint(tccpixel, tcccanvas)
            phis, weights = TCC(circ, pupil, tccpixel, tcccanvas, thresh=thresh)
            tccsize = tcccanvas//tccpixel
            for idx in range(len(phis)): 
                padded = tccsize*padding
                ffted = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(phis[idx])))
                realpart = np.zeros((padded, padded))
                imagpart = np.zeros((padded, padded))
                begin = (padded - tccsize) // 2
                end = begin + tccsize
                realpart[begin:end, begin:end] = ffted.real
                imagpart[begin:end, begin:end] = ffted.imag
                ffted = realpart + 1j * imagpart
                realfft = cv2.resize(ffted.real, (size, size), interpolation=cv2.INTER_LINEAR)
                imagfft = cv2.resize(ffted.imag, (size, size), interpolation=cv2.INTER_LINEAR)
                ffted = realfft + 1j * imagfft
                phis[idx] = padding**2 * resize**2 * np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(ffted)))
        return phis, weights


def readTccParaFromDisc(path: str):
    with open(path, "rb") as fin:
        phis, weights = pickle.load(fin)
    return phis, weights


def writeTccParaToDisc(phis, weights, path: str):
    with open(path, "wb") as fout:
        pickle.dump((phis, weights), fout)


if __name__ == "__main__": 
    genTCC(4, 512, 1.35, 193)
