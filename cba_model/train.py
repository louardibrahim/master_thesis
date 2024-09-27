import torch
from tqdm import tqdm

def train(model, trainloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    running_correct = 0
    counter = 0

    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        x_batch, y_batch = data
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)

        running_loss += loss.item()
        _, preds = torch.max(outputs.data, 1)
        running_correct += (preds == y_batch).sum().item()

        loss.backward()
        optimizer.step()

    epoch_loss = running_loss / counter
    epoch_acc = 100. * (running_correct / len(trainloader.dataset))
    return epoch_loss, epoch_acc

def validate(model, testloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    counter = 0

    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            x_batch, y_batch = data
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)

            running_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            running_correct += (preds == y_batch).sum().item()

    epoch_loss = running_loss / counter
    epoch_acc = 100. * (running_correct / len(testloader.dataset))
    return epoch_loss, epoch_acc
