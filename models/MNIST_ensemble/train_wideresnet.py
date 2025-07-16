import torch
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, ReLU, Dropout #All of the necessary layers for this model
from torch import nn
from torch.nn import functional as F
import numpy as np
import time
from train_mnist_utils import get_mnist_train_and_test_loaders, train, evaluate_model, plot_training_history, plot_batch_training_history, log_final_model_stats, count_parameters

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
    """Ultra-compact WideResNet for MNIST with minimal parameters"""
    def __init__(self, num_classes=10, width_multiplier=2, dropout_rate=0.3):
        super(MiniWideResNet, self).__init__()
        
        base_width = 16
        widths = [base_width, base_width * width_multiplier, 
                 base_width * width_multiplier * 2]
        
        self.conv1 = nn.Conv2d(1, widths[0], kernel_size=3, padding=1, bias=False)
        
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
        out = self.group1(out)  # 28x28 -> 28x28
        out = self.group2(out)  # 28x28 -> 14x14
        out = F.relu(self.bn_final(out))
        out = self.avgpool(out)  # 14x14 -> 1x1
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

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
    
    train_loader, test_loader = get_mnist_train_and_test_loaders()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    epochs = 2
    
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    batch_history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

    # Start timing
    start_time = time.time()
    print("Starting training...")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc, batch_metrics = train(model, train_loader, test_loader, optimizer, criterion, device, epoch)
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
        
        # Collect epoch-level metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        # Collect batch-level metrics
        for batch_metric in batch_metrics:
            batch_history['train_loss'].append(batch_metric['train_loss'])
            batch_history['train_acc'].append(batch_metric['train_acc'])
            batch_history['test_loss'].append(batch_metric['test_loss'])
            batch_history['test_acc'].append(batch_metric['test_acc'])

        print(f"Epoch {epoch}/{epochs} Summary -> Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.3f}% | Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.3f}%")

    # End timing and calculate total time
    end_time = time.time()
    total_time = end_time - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)

    print("Finished Training")
    print(f"Total training time: {hours:02d}:{minutes:02d}:{seconds:02d} (hours:minutes:seconds)")
    log_final_model_stats(model, train_loader, test_loader, criterion, device)
    
    plot_batch_training_history(batch_history)

    torch.save(model, "mnist_wideresnet.pth")
    print("Saved model weights to mnist_wideresnet.pth")

if __name__ == '__main__':
    main()
