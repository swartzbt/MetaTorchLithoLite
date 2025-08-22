import torch
import numpy as np
from matplotlib import pyplot as plt


from pylitho import Hopkins, Abbe

if __name__ == "__main__":

    pixelsize = 8
    canvas = 512
    size = round(canvas / pixelsize)

    device=torch.device("cpu")

    sim1 = Hopkins(
        canvas=canvas,
        pixel=pixelsize
    )
    sim2 = Abbe(
        canvas=canvas,
        pixel=pixelsize
    )

    nb=np.zeros([1, 1, size, size])
    nb[0, 0, 16:48, 16:48]=1

    # Hopkins simulation
    b=torch.tensor(nb, device=device, requires_grad=True)
    image1 = sim1(b)
    image1 = image1
    
    # Abbe simulation
    b1=torch.tensor(nb, device=device, requires_grad=True)
    image2 = sim2(b1)

    image1.backward(torch.ones_like(image1))
    image2.backward(torch.ones_like(image2))

    print(image1.shape)
    print(image2.shape)


    plt.subplot(2, 2, 1)
    plt.imshow(image1[0][0].detach().numpy())
    plt.subplot(2, 2, 2)
    plt.imshow(image2[0][0].detach().numpy())
    plt.subplot(2, 2, 3)
    plt.imshow(b.grad[0][0].numpy())
    plt.subplot(2, 2, 4)
    plt.imshow(b1.grad[0][0].numpy())
    plt.savefig("example.png")

