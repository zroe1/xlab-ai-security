import torch
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, ReLU, Dropout #All of the necessary layers for this model
from torch import nn
from torch.nn import functional as F
import numpy as np
import time
from train_mnist_utils import get_mnist_train_and_test_loaders, train, evaluate_model, plot_training_history, plot_batch_training_history, log_final_model_stats, count_parameters

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

def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")

    model = ConvolutionalMNIST().to(device)
    print(f"Simple CNN parameters: {count_parameters(model):,}")
    
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
    
    plot_batch_training_history(batch_history, save_path="simple_cnn_training_performance.png")

    torch.save(model, "mnist_simple_cnn.pth")
    print("Saved model weights to mnist_simple_cnn.pth")

if __name__ == '__main__':
    main() 