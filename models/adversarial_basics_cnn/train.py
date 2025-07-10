import torch
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, ReLU, Dropout #All of the necessary layers for this model
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import numpy as np

class BasicBlock(nn.Module):
    """Basic residual block for compact WideResNet"""
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.3):
        super(BasicBlock, self).__init__()
        
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                                    stride=stride, bias=False)
    
    def forward(self, x):
        out = F.relu(self.bn1(x))
        out = self.conv1(out)
        out = self.dropout(out)
        out = F.relu(self.bn2(out))
        out = self.conv2(out)
        
        # Skip connection
        shortcut = self.shortcut(x)
        out += shortcut
        
        return out

class MiniWideResNet(nn.Module):
    """Ultra-compact WideResNet for CIFAR with minimal parameters"""
    def __init__(self, num_classes=10, width_multiplier=2, dropout_rate=0.3):
        super(MiniWideResNet, self).__init__()
        
        base_width = 16
        widths = [base_width, base_width * width_multiplier, 
                 base_width * width_multiplier * 2]
        
        self.conv1 = nn.Conv2d(3, widths[0], kernel_size=3, padding=1, bias=False)
        
        # Only two groups for minimal size
        self.group1 = self._make_group(widths[0], widths[1], 2, stride=1, dropout_rate=dropout_rate)
        self.group2 = self._make_group(widths[1], widths[2], 2, stride=2, dropout_rate=dropout_rate)
        
        self.bn_final = nn.BatchNorm2d(widths[2])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(widths[2], num_classes)
        
        self._initialize_weights()
    
    def _make_group(self, in_channels, out_channels, num_blocks, stride, dropout_rate):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride, dropout_rate))
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, 1, dropout_rate))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.group1(out)  # 32x32 -> 32x32
        out = self.group2(out)  # 32x32 -> 16x16
        out = F.relu(self.bn_final(out))
        out = self.avgpool(out)  # 16x16 -> 1x1
        out = out.view(out.size(0), -1)
        out = self.fc(out)
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
            test_loss, test_acc = evaluate_model(model, test_loader, criterion, device, num_batches=5)
            # switch back to train mode
            model.train()
            print(f'Epoch: {epoch} | Batch: {batch_idx+1}/{len(train_loader)} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.3f}% | Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.3f}%')
    
    final_train_loss = total_loss / len(train_loader)
    final_train_acc = 100. * correct / total
    return final_train_loss, final_train_acc


# NOTE: in general it is best practice to use a separate test set to evaluate the model
# once training is complete. That means that if you are using the test accuracy to
# tune hyperparameters, you should use a separate test set to evaluate the model
# once training is complete. We aren't doing that here, but it is good to know.
def evaluate_model(model, loader, criterion, device, num_batches=None):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    num_actual_batches = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            if num_batches is not None and batch_idx >= num_batches:
                break
            
            num_actual_batches += 1
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    avg_loss = total_loss / num_actual_batches if num_actual_batches > 0 else 0
    accuracy = 100 * correct / total if total > 0 else 0
    return avg_loss, accuracy

def plot_training_history(history, save_path="training_performance.png"):
    """Plots and saves the training history."""
    fig, ax1 = plt.subplots(figsize=(12, 7))

    epochs = range(1, len(history['train_loss']) + 1)

    # Plotting Loss on the primary y-axis (ax1)
    color = '#4871cf'
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', color=color, fontsize=12)
    ax1.plot(epochs, history['train_loss'], color=color, linestyle='-', marker='o', markersize=4, label='Train Loss')
    ax1.plot(epochs, history['test_loss'], color=color, linestyle='--', marker='x', markersize=4, label='Test Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Creating a secondary y-axis for Accuracy
    ax2 = ax1.twinx()
    color = 'maroon'
    ax2.set_ylabel('Accuracy (%)', color=color, fontsize=12)
    ax2.plot(epochs, history['train_acc'], color=color, linestyle='-', marker='o', markersize=4, label='Train Accuracy')
    ax2.plot(epochs, history['test_acc'], color=color, linestyle='--', marker='x', markersize=4, label='Test Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)

    # Adding title
    plt.title('Model Training History', fontsize=14)
    
    # Unified legend positioned outside the plot area
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout to make room for the legend
    plt.tight_layout()
    
    # Save the figure with bbox_inches='tight' to include the legend
    fig.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Training history plot saved to {save_path}")
    plt.close(fig)

def log_final_model_stats(model, train_loader, test_loader, criterion, device):
    print("\n--- Final Model Statistics ---")
    
    # Final Training Statistics
    train_loss, train_acc = evaluate_model(model, train_loader, criterion, device)
    print(f"Final Train Loss: {train_loss:.4f} | Final Train Accuracy: {train_acc:.2f}%")
    
    # Final Test Statistics
    test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
    print(f"Final Test Loss:  {test_loss:.4f} | Final Test Accuracy:  {test_acc:.2f}%")
    print("---------------------------------\n")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")

    model = MiniWideResNet().to(device)
    print(f"Mini WideResNet parameters: {count_parameters(model):,}")
    
    train_loader, test_loader = get_cifar10_train_and_test_loaders()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    epochs = 75
    
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train(model, train_loader, test_loader, optimizer, criterion, device, epoch)
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        print(f"Epoch {epoch}/{epochs} Summary -> Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.3f}% | Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.3f}%")


    print("Finished Training")
    log_final_model_stats(model, train_loader, test_loader, criterion, device)
    
    plot_training_history(history)

    torch.save(model.state_dict(), "adversarial_basics_cnn.pth")
    print("Saved model weights to adversarial_basics_cnn.pth")

if __name__ == '__main__':
    main() 