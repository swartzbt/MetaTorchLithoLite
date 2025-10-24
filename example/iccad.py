import torch
import numpy as np
from matplotlib import pyplot as plt


from ..pylitho import Hopkins, Abbe
from ..pylitho import Design

if __name__ == "__main__":

    pixelsize = 1
    canvas = 2048
    size = round(canvas / pixelsize)

    device=torch.device("cpu")
    
    sim = Abbe(
        canvas=canvas,
        pixel=pixelsize
    )

    design=Design("benchmark/ICCAD2013/M1_test1.glp")
    img=design.mat().reshape(1, 1, 2048, 2048)

    # Abbe simulation
    tensor=torch.tensor(img, device=device, requires_grad=True)
    printed = sim(tensor)

    printed.backward(torch.ones_like(tensor))


    plt.subplot(1, 3, 1)
    plt.imshow(tensor[0][0].detach().numpy())
    plt.subplot(1, 3, 2)
    plt.imshow(printed[0][0].detach().numpy())
    plt.subplot(1, 3, 3)
    plt.imshow(tensor.grad[0][0].numpy())
    plt.savefig("example.png")
