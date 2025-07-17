import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

def get_mnist_train_and_test_loaders(batch_size=128, num_workers=4):
    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader

def train(model, train_loader, test_loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    batch_metrics = []
    
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

        if (batch_idx + 1) % 50 == 0:
            # track metrics every 50 batches
            train_loss = total_loss / (batch_idx + 1)
            train_acc = 100. * correct / total
            test_loss, test_acc = evaluate_model(model, test_loader, criterion, device, num_batches=5)
            # switch back to train mode
            model.train()
            
            batch_metrics.append({
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_loss': test_loss,
                'test_acc': test_acc
            })
            
            print(f'Epoch: {epoch} | Batch: {batch_idx+1}/{len(train_loader)} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.3f}% | Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.3f}%')
    
    final_train_loss = total_loss / len(train_loader)
    final_train_acc = 100. * correct / total
    return final_train_loss, final_train_acc, batch_metrics


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

def plot_batch_training_history(batch_history, save_path="training_performance.png"):
    """Plots and saves the batch-level training history."""
    fig, ax1 = plt.subplots(figsize=(12, 7))

    batch_steps = range(1, len(batch_history['train_loss']) + 1)

    # Plotting Loss on the primary y-axis (ax1)
    color = '#4871cf'
    ax1.set_xlabel('Batch Steps (every 50 batches)', fontsize=12)
    ax1.set_ylabel('Loss', color=color, fontsize=12)
    ax1.plot(batch_steps, batch_history['train_loss'], color=color, linestyle='-', marker='o', markersize=2, label='Train Loss')
    ax1.plot(batch_steps, batch_history['test_loss'], color=color, linestyle='--', marker='x', markersize=2, label='Test Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Creating a secondary y-axis for Accuracy
    ax2 = ax1.twinx()
    color = 'maroon'
    ax2.set_ylabel('Accuracy (%)', color=color, fontsize=12)
    ax2.plot(batch_steps, batch_history['train_acc'], color=color, linestyle='-', marker='o', markersize=2, label='Train Accuracy')
    ax2.plot(batch_steps, batch_history['test_acc'], color=color, linestyle='--', marker='x', markersize=2, label='Test Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)

    # Adding title
    plt.title('Model Training History (Batch Level)', fontsize=14)
    
    # Unified legend positioned outside the plot area
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout to make room for the legend
    plt.tight_layout()
    
    # Save the figure with bbox_inches='tight' to include the legend
    fig.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Batch training history plot saved to {save_path}")
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
