import torch
import matplotlib.pyplot as plt

def train(model, device, trainloader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    batch_losses = []
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        # one_hot_vectors = torch.nn.functional.one_hot(targets, num_classes=10).float().to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        batch_losses.append(loss.item())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % 100 == 99:
            # print('Train inputs: ',inputs[50])
            # print('Train outputs: ',outputs[50])
            print(f'Epoch {epoch}, Batch {batch_idx + 1}, Average_Loss: {running_loss / 100:.3f}')
            running_loss = 0.0
     # matplotlib
    plt.figure(figsize=(10, 5))
    plt.plot(batch_losses, label='Batch Loss')
    plt.xlabel('Batch Number')
    plt.ylabel('Loss')
    plt.title('Training Loss per Batch in one epoch')
    plt.legend()
    plt.show()

def test(model, device, testloader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            # one_hot_vectors = torch.nn.functional.one_hot(targets, num_classes=10).float().to(device)
            outputs = model(inputs)
            # if batch_idx % 50 == 49:
            #     print('Test input: ',inputs[50])
            #     print('Test output: ',outputs[50])
            test_loss += criterion(outputs, targets).item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

    test_loss /= len(testloader.dataset)
    accuracy = 100. * correct / len(testloader.dataset)
    print(f'Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')
