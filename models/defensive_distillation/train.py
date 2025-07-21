import torch
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, ReLU, Dropout #All of the necessary layers for this model
from torch import nn
from torch.nn import functional as F
import numpy as np
import time
from train_mnist_utils import get_mnist_train_and_test_loaders, train, evaluate_model, plot_training_history, plot_batch_training_history, log_final_model_stats, count_parameters

class TemperatureScaledNLLLoss(nn.Module):
    """Negative log-likelihood with temperature-scaled softmax.

    The logits are divided by temperature **T** before the log-softmax is taken.
    Setting T=1 reduces to the standard NLL loss.
    """
    def __init__(self, T: float = 1.0):
        super().__init__()
        self.T = T
        self.nll_loss = nn.NLLLoss()

    def forward(self, logits, target):
        # Scale logits by temperature and compute log-probabilities
        log_probs = F.log_softmax(logits / self.T, dim=1)
        return self.nll_loss(log_probs, target)

class FeedforwardMNIST(nn.Module):
    """Simple 4-layer MLP for MNIST classification"""
    def __init__(self, num_classes=10):
        super(FeedforwardMNIST, self).__init__()
        
        input_size = 28 * 28
        self.fc1 = Linear(input_size, 256)
        self.fc2 = Linear(256, 64)
        self.fc3 = Linear(64, num_classes)
        
        self.flatten = Flatten()
        self.relu = ReLU()
        
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
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

    model = FeedforwardMNIST().to(device)
    print(f"FeedforwardMNIST parameters: {count_parameters(model):,}")
    
    train_loader, test_loader = get_mnist_train_and_test_loaders()
    # Temperature to use for softmax scaling
    temperature = 20
    print(f"Training with T={temperature}")
    criterion = TemperatureScaledNLLLoss(temperature)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

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
    
    plot_batch_training_history(batch_history, save_path="mlp_training_performance.png")

    torch.save(model, "mnist_mlp.pth")
    print("Saved model weights to mnist_mlp.pth")

if __name__ == '__main__':
    main() 