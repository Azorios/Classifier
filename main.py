import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

from plotting import loss_plot
from neural_network import create_model, load_model, save_model
from training import train_model, show_training_imgs
from testing import predict_acc, predict_class_acc, test_prediction


if __name__ == '__main__':

    # use GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device, 'will be used.')

    num_epochs = 2
    batch_size = 4
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
    for i in range (2): show_training_imgs(train_loader, classes, batch_size)

    # define network
    model = create_model(device)
    load_model(model)

    # choosing loss function
    criterion = nn.CrossEntropyLoss()

    # stochastic gradient descent: to perform parameter update for each training sample
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    #training phase
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
