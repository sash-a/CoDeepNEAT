def train(model, device, train_loader, epoch):
    """
    Run a single train epoch
    :param model: the network of type torch.nn.Module
    :param device: Device to train on (cuda or cpu)
    :param train_loader: the training dataset
    :param epoch: the current epoch
    """
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # TODO add all to GPU in parallel and keep the data there
        data, target = data.to(device), target.to(device)
        model.optimizer.zero_grad()
        # compute loss without variables to avoid copying from gpu to cpu
        model.loss_fn(model(data), target).backward()
        model.optimizer.step()

    if epoch % 10 == 0:
        print('some useful info')


def test(model, device, train_loader, epoch):
    """
    Run through a test dataset and return the accuracy
    :param model: the network of type torch.nn.Module
    :param device: Device to train on (cuda or cpu)
    :param train_loader: the training dataset
    :param epoch: the current epoch
    :return: accuracy
    """
    model.eval()
    # TODO
