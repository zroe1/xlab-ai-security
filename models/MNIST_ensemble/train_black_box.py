import torch
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, ReLU, Dropout, Tanh #All of the necessary layers for this model
from torch import nn
from torch.nn import functional as F
import numpy as np
from train_mnist_utils import get_mnist_train_and_test_loaders, train, evaluate_model, plot_training_history, plot_batch_training_history, log_final_model_stats, count_parameters

class BlackBoxMLP(nn.Module):
    """Black box MLP for MNIST classification with specified architecture"""
    def __init__(self, num_classes=10):
        super(BlackBoxMLP, self).__init__()
        
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

def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")

    model = BlackBoxMLP().to(device)
    print(f"Black Box MLP parameters: {count_parameters(model):,}")
    
    train_loader, test_loader = get_mnist_train_and_test_loaders()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

    epochs = 2
    
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    batch_history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

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


    print("Finished Training")
    log_final_model_stats(model, train_loader, test_loader, criterion, device)
    
    plot_batch_training_history(batch_history, save_path="black_box_mlp_training_performance.png")

    torch.save(model, "mnist_black_box_mlp.pth")
    print("Saved model weights to mnist_black_box_mlp.pth")

if __name__ == '__main__':
    main() 