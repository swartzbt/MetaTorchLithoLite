from abc import ABC, abstractmethod
import torch
from .sim.abbe.func import AbbeFunc
from .sim.hopkins.func import HopkinsFunc
from .sim.hopkins.tcc import genTCC
from typing import Union, List
from .utils.interpolate import getInterpolateAerialImageNCHW
from .utils.interpolate import getInterpolateAerialImageCHW
from .utils.interpolate import getInterpolateAerialImage

class Simulator(ABC):
    def __init__(self, 
                canvas : int,
                pixel : int,
                sigma : float =0.05, 
                NA : float =1.35, 
                wavelength : int =193,
                defocus : Union[None, List[int], str] = None,  
                batch : bool =False,
                **kwargs) -> None:
        super().__init__()
        assert canvas % pixel == 0, "Canvas size must be divisible by pixel size"
        self.canvas = canvas
        self.pixel = pixel
        self.sigma = sigma
        self.NA = NA
        self.wavelength = wavelength
        self.defocus = defocus
        self.batch = batch
        if "parallel" in kwargs: 
            self.parallel = kwargs["parallel"]
        else:
            self.parallel = False
            
        self.defocus = [0, 30, 60] if defocus == "default" else defocus
        
    def __interpolate(self, mask : torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def __call__(self, mask : torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Simulator subclass must implement __call__ method")
        

class Abbe(Simulator):
    def __init__(self, 
                 canvas : int,
                 pixel : int, 
                 sigma : float = 0.05, 
                 NA : float = 1.35, 
                 wavelength : int = 193, 
                 defocus : Union[None, List[int], str] = None, 
                 batch : bool = False, 
                 **kwargs):
        super().__init__(
                        canvas,
                        pixel, 
                        sigma, 
                        NA, 
                        wavelength, 
                        defocus, 
                        batch, 
                        **kwargs)
        
    def __call__(self, mask : torch.Tensor):
        # using AbbeFunc from OpenLitho/pylitho/sim/abbe/func.py
        
        image = AbbeFunc.apply(mask, 
                              self.pixel, 
                              self.sigma, 
                              self.NA, 
                              self.wavelength, 
                              self.defocus, 
                              self.parallel,
                              self.batch)
        if image.dim() == 2:
            return getInterpolateAerialImage(image, self.pixel)
        elif image.dim() == 3:
            return getInterpolateAerialImageCHW(image, self.pixel)
        return getInterpolateAerialImageNCHW(image, self.pixel)
        
class Hopkins(Simulator):
    def __init__(self, 
                canvas : int,
                pixel : int, 
                sigma : float = 0.05, 
                NA : float = 1.35, 
                wavelength : int = 193, 
                defocus : Union[None, List[int], str] = None, 
                batch : bool = False, 
                **kwargs):
        super().__init__(
                        canvas,
                        pixel, 
                        sigma, 
                        NA, 
                        wavelength, 
                        defocus, 
                        batch, 
                        **kwargs)
        
        self.tcc = genTCC(self.pixel,
                            self.canvas,
                            self.NA,
                            self.wavelength,
                            self.defocus)
        
    def __call__(self, mask : torch.Tensor):
        image = HopkinsFunc(
            self.tcc,
            defocus=self.defocus is not None,
            device=mask.device,
        )(mask)
        
        if image.dim() == 2:
            return getInterpolateAerialImage(image, self.pixel)
        elif image.dim() == 3:
            return getInterpolateAerialImageCHW(image, self.pixel)
        return getInterpolateAerialImageNCHW(image, self.pixel)
