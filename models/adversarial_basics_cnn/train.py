import torch
import xlab
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, ReLU, Dropout #All of the necessary layers for this model
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = Conv2d(3, 16, kernel_size = 3, padding = 1)
        self.pooling = MaxPool2d(2,2)
        self.dropout = Dropout(p=0.3)
        self.conv2 = Conv2d(16, 32, kernel_size = 3, padding = 1)
        self.relu = ReLU()
        self.flatten = Flatten()
        self.linear1 = Linear(2048, 128)
        self.linear2 = Linear(128, 10)

    def forward(self, x):
        out = self.conv1(x)        # 1. conv1
        out = self.pooling(out)    # 2. pooling
        out = self.dropout(out)    # 3. dropout
        out = self.conv2(out)      # 4. conv2
        out = self.pooling(out)    # 5. pooling
        out = self.flatten(out)    # 6. flatten
        out = self.linear1(out)    # 7. linear1
        out = self.relu(out)       # 8. relu
        out = self.dropout(out)    # 9. dropout
        out = self.linear2(out)    # 10. linear2
        return out
    
def get_cifar10_train_and_test_loaders(batch_size=128, num_workers=4):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader

def train(model, train_loader, test_loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

        if (batch_idx + 1) % 100 == 0:
            # print every 100 batches
            train_loss = total_loss / (batch_idx + 1)
            train_acc = 100. * correct / total
            test_loss, test_acc = estimate_test_accuracy(model, test_loader, criterion, device)
            # switch back to train mode
            model.train()
            print(f'Epoch: {epoch} | Batch: {batch_idx+1}/{len(train_loader)} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.3f}% | Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.3f}%')


# NOTE: in general it is best practice to use a separate test set to evaluate the model
# once training is complete. That means that if you are using the test accuracy to
# tune hyperparameters, you should use a separate test set to evaluate the model
# once training is complete. We aren't doing that here, but it is good to know.
def estimate_test_accuracy(model, test_loader, criterion, device, num_batches=5):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0
    num_actual_batches = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            if num_batches is not None and batch_idx >= num_batches:
                break
            
            num_actual_batches += 1
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    avg_loss = test_loss / num_actual_batches if num_actual_batches > 0 else 0
    accuracy = 100 * correct / total if total > 0 else 0
    return avg_loss, accuracy

def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")

    model = CNN().to(device)
    train_loader, test_loader = get_cifar10_train_and_test_loaders()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 10
    for epoch in range(1, epochs + 1):
        train(model, train_loader, test_loader, optimizer, criterion, device, epoch)

    print("Finished Training")
    torch.save(model.state_dict(), "adversarial_basics_cnn.pth")
    print("Saved model weights to adversarial_basics_cnn.pth")

if __name__ == '__main__':
    main() 