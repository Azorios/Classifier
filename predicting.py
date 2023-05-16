import torch
import torchvision

from plotting import imshow


def predict_acc(test_loader, device, model):
    # prediction counter
    correct = 0
    total = 0

    with torch.no_grad():

        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            #print(total)

    print(f'Accuracy of the network on the {total} test images: {100 * correct // total} %')


def predict_class_acc(classes, test_loader, model):
    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')


def test_prediction(test_loader, classes, batch_size, model):
    # get some random testing images
    dataiter = iter(test_loader)
    images, labels = next(dataiter)

    # show images
    imshow(torchvision.utils.make_grid(images), 'testing images')
    # print labels as one string
    print('GroundTruth: ', ' '.join(f'{classes[labels[j]]}' for j in range(batch_size)))

    # output prediction
    output = model(images)
    _, predicted = torch.max(output, 1)

    print('Predicted: ', ' '.join(f'{classes[predicted[j]]}' for j in range(batch_size)))
