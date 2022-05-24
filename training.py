import torchvision.utils

from plotting import imshow


def train_model(num_epochs, train_loader, device, optimizer, model, criterion):
    # record training loss for plot
    epoch_losses = []

    # loop over the dataset multiple times
    for epoch in range(num_epochs):
        running_loss = 0.0
        saved_loss = 0.0

        for i, data in enumerate(train_loader, 0):
            # get inputs and labels; data is a list of [inputs, labels]
            inputs, labels = data

            # convert to appropriate device
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                saved_loss = running_loss
                running_loss = 0.0

        epoch_losses.append(saved_loss/10000)

    print('Finished Training')
    return epoch_losses


def show_training_imgs(train_loader, classes, batch_size):

    # get images and labels
    dataiter = iter(train_loader)
    images, labels = next(dataiter)

    # show images
    imshow(torchvision.utils.make_grid(images), 'training examples')

    # print labels as one string
    print(' '.join(f'{classes[labels[j]]}' for j in range(batch_size)))
