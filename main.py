import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from torchcam.methods import GradCAM
from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchcam.utils import overlay_mask

from predicting import predict_acc, predict_class_acc, test_prediction


if __name__ == '__main__':
    # use GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device, 'will be used.')

    batch_size = 30

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

    # use pretrained model
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_repvgg_a1", pretrained=True)

    # testing phase
    model.eval()

    # test: predict some images
    for i in range(2): test_prediction(test_loader, classes, batch_size, model)

    # test model for accuracy
    #predict_acc(test_loader, device, model)
    #predict_class_acc(classes, test_loader, model)

    print('GRAD-CAM')

    # get images and labels
    dataiter = iter(test_loader)
    images, labels = next(dataiter)

    # get predictions
    output = model(images)
    _, predicted = torch.max(output, 1)

    # construct GradCAM object once and re-use on multiple images
    cam_extractor = GradCAM(model)

    for i in range(batch_size):
        # save image as png file
        im_path = f'./images/{i+1}.png'
        unnorm_im = images[i] / 2 + 0.5  # unnormalize
        torchvision.utils.save_image(unnorm_im, im_path)
        open_im = mpimg.imread(im_path)

        # print actual and predicted class for every image
        print(f'{i+1}. class: {classes[labels[i]]}, predicted: {classes[predicted[i]]}')

        # get input image
        img = read_image(im_path)

        # preprocess for chosen model
        input_tensor = normalize(resize(img, [224, 224]) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

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
