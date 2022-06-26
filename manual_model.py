import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

from torchcam.methods import GradCAM
from matplotlib import image as mpimg, pyplot as plt
from torchcam.utils import overlay_mask
from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, to_pil_image

from plotting import loss_plot
from neural_network import create_model, load_model, save_model
from training import train_model, show_training_imgs
from predicting import predict_acc, predict_class_acc, test_prediction


#if __name__ == '__main__':

    # use GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device, 'will be used.')

    num_epochs = 2
    batch_size = 30
    learning_rate = 0.001

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # transform input data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # load CIFAR10 dataset
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # load datasets into Python parameters
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # test: show some random training images
    for i in range(2): show_training_imgs(train_loader, classes, batch_size)

    # define network
    model = create_model(device)
    load_model(model)

    # choosing loss function
    criterion = nn.CrossEntropyLoss()

    # stochastic gradient descent: to perform parameter update for each training sample
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # training phase
    model.train()

    # train the model
    epoch_losses = train_model(num_epochs, train_loader, device, optimizer, model, criterion)

    # plot the training loss; if curve starts increasing back, model is overfitting => adjust number of epochs
    loss_plot(epoch_losses, num_epochs)

    # save model
    save_model(model)

    # testing phase
    model.eval()

    # test: predict some images
    for i in range(5): test_prediction(test_loader, classes, batch_size, model)

    # test model for accuracy
    predict_acc(test_loader, device, model)
    predict_class_acc(classes, test_loader, model)

    # create saliency map

    print('GRAD-CAM')
    # get images and labels
    dataiter = iter(train_loader)
    images, labels = next(dataiter)

    # get predictions
    output = model(images)
    _, predicted = torch.max(output, 1)

    # construct GradCAM object once and re-use on multiple images
    cam_extractor = GradCAM(model, 'conv3')

    for i in range(batch_size):
        # save image as png file
        im_path = f'{i + 1}.png'
        unnorm_im = images[i] / 2 + 0.5  # unnormalize
        torchvision.utils.save_image(unnorm_im, im_path)
        open_im = mpimg.imread(im_path)

        # print actual and predicted class for every image
        print(f'{i + 1}. class: {classes[labels[i]]}, predicted: {classes[predicted[i]]}')

        # get input image
        img = read_image(im_path)

        # preprocess for chosen model
        input_tensor = normalize(img/255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # preprocess data and feed it to the model
        out = model(input_tensor.unsqueeze(0))
        # retrieve GradCAM by passing the class index and the model output
        activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)

        # subplot image
        plt.subplot(1, 3, 1)
        plt.title(f'class: {classes[labels[i]]}\n predicted: {classes[predicted[i]]}')
        plt.imshow(open_im)
        plt.axis('off')

        # subplot raw CAM
        plt.subplot(1, 3, 2)
        plt.imshow(activation_map[0].squeeze(0).numpy())
        plt.title('raw CAM')
        plt.axis('off')

        # subplot image overlayed with CAM
        result = overlay_mask(to_pil_image(img), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
        plt.subplot(1, 3, 3)
        plt.imshow(result)
        plt.title('result')
        plt.axis('off')

        # all in one plot
        plt.tight_layout()
        plt.show()
