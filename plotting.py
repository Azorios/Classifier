import matplotlib.pyplot as plt
import numpy as np


def imshow(img, title):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.show()


def loss_plot(epoch_losses, epochs):
    epochs = range(1, epochs+1)

    plt.plot(epochs, epoch_losses, 'g', label='Training loss')
    plt.title('Trainingloss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


#def plot_saliency():
