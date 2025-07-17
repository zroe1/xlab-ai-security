import torch
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, ReLU, Dropout, Tanh #All of the necessary layers for this model
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import nn
from torch.nn import functional as F

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
    
class BlackBox(nn.Module):
    """Black box MLP for MNIST classification with specified architecture"""
    def __init__(self, num_classes=10):
        super(BlackBox, self).__init__()
        
        # Input size for MNIST: 28 * 28 = 784
        input_size = 28 * 28
        
        # Fully connected layers with specified architecture
        self.fc1 = Linear(input_size, 256)  # Layer 1: 256 outputs
        self.fc2 = Linear(256, 32)          # Layer 2: 32 outputs
        self.fc3 = Linear(32, num_classes)  # Final layer: num_classes outputs
        
        # Flatten layer
        self.flatten = Flatten()
        
        # Activation functions
        self.tanh = Tanh()
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.tanh(self.fc1(x))  # Layer 1 with tanh activation
        x = self.tanh(self.fc2(x))  # Layer 2 with tanh activation
        x = self.fc3(x)             # Final layer (no activation, will use CrossEntropyLoss)
        
        return x
    
class ConvolutionalMNIST(nn.Module):
    """Simple CNN for MNIST classification"""
    def __init__(self, num_classes=10, dropout_rate=0.3):
        super(ConvolutionalMNIST, self).__init__()
        
        # Convolutional layers
        self.conv1 = Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = MaxPool2d(2, 2)
        
        # Dropout layers
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
        
        # Fully connected layers
        # After 3 conv+pool layers: 28x28 -> 14x14 -> 7x7 -> 3x3 (with padding)
        # Actually: 28x28 -> 14x14 -> 7x7 -> 3x3, so 64 * 3 * 3 = 576
        self.fc1 = Linear(64 * 3 * 3, 128)
        self.fc2 = Linear(128, 64)
        self.fc3 = Linear(64, num_classes)
        
        # Flatten layer
        self.flatten = Flatten()
        
        # Activation functions
        self.relu = ReLU()
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # First conv block: 28x28 -> 14x14
        x = self.pool(self.relu(self.conv1(x)))
        
        # Second conv block: 14x14 -> 7x7
        x = self.pool(self.relu(self.conv2(x)))
        
        # Third conv block: 7x7 -> 3x3
        x = self.pool(self.relu(self.conv3(x)))
        
        # Flatten and apply dropout
        x = self.flatten(x)
        x = self.dropout1(x)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

class BasicBlockMNIST(nn.Module):
    """Basic residual block for compact WideResNet"""
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.3):
        super(BasicBlockMNIST, self).__init__()
        
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

class ResNetMNIST(nn.Module):
    """Ultra-compact WideResNet for MNIST with minimal parameters"""
    def __init__(self, num_classes=10, width_multiplier=2, dropout_rate=0.3):
        super(ResNetMNIST, self).__init__()
        
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
        layers.append(BasicBlockMNIST(in_channels, out_channels, stride, dropout_rate))
        for _ in range(1, num_blocks):
            layers.append(BasicBlockMNIST(out_channels, out_channels, 1, dropout_rate))
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

class FeedforwardMNIST(nn.Module):
    """Simple 4-layer MLP for MNIST classification"""
    def __init__(self, num_classes=10, hidden_size=48):
        super(FeedforwardMNIST, self).__init__()
        
        # Input size for MNIST: 28 * 28 = 784
        input_size = 28 * 28
        
        # Fully connected layers
        self.fc1 = Linear(input_size, hidden_size)
        self.fc2 = Linear(hidden_size, hidden_size)
        self.fc3 = Linear(hidden_size, num_classes)
        
        # Flatten layer
        self.flatten = Flatten()
        
        # Activation functions
        self.relu = ReLU()
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x