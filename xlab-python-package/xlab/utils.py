import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import math
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, ReLU, Dropout
from torchvision.io import read_image
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image


def add_noise(img, stdev=0.001, mean=0):
    """
    Helper function for PGD_generator
    
    Parameters:
    -----------
    img : Tensor
        image Tensor to be predicted

    Returns:
    --------
    noisy_img: Tensor
        Added noise to input
    """
    noise = torch.randn_like(img) * stdev + mean
    return img + noise


def PGD_generator(model, loss_fn, path, y, epsilon=1/1000, alpha=0.0005, num_iters=6):
    """
    Create adversarial image using PGD
    
    Parameters:
    -----------
    model : PyTorch model used for classification
    loss_fn : Loss function
    path: Image filepath
    y: Image label
    epsilon: Perturbation variable
    Alpha: Perturbation variable
    num_iters: Number of iterations

    Returns:
    --------
    adv_img: Adversarially perturbed image tensor
    """
    model.eval()
    x = process_image(path)
    x = add_noise(x)
    x = torch.clamp(x, -1, 1)
    for i in range(num_iters):
        x.requires_grad = True
        output = model(x)
        loss = loss_fn(output, y)
        model.zero_grad()
        loss.backward()
        loss_gradient = x.grad.data
        x = x.detach()
        x = x + alpha * torch.sign(loss_gradient)
        x = clip(x, epsilon)
    return x


def prediction(model, img):
    """
    Return prediction tuple:
    
    Parameters:
    -----------
    model : PyTorch model used for classification
    img : Tensor
        image Tensor to be predicted

    Returns:
    --------
    pred_class : torch.Tensor
        The predicted class label
    prob : torch.Tensor
        The confidence of the model 
    
    """
    with torch.no_grad(): #Stops calculating gradients
        prediction = model(img)
        _, pred_class = torch.max(prediction, 1)
    probs = prediction.softmax(dim=-1)
    return pred_class, probs[0][pred_class]


def show_image(img):
    """
    Display image tensor using plt
    
    Parameters:
    -----------
    img : Tensor
        image Tensor to be displayed
    """
    img = img.squeeze(0)
    plt.imshow(img.permute(1, 2, 0).detach().numpy())

def process_image(path):
    """
    Convert file path to scaled torch tensor
    
    Parameters:
    -----------
    path : str
        Filepath for image

    Returns:
    --------
    processedImg: Scaled and transformed image tensor
    
    """
    img = Image.open(path)
    transform = Compose([Resize((32,32)), ToTensor()])
    processedImg = transform(img)
    processedImg = processedImg.unsqueeze(0)
    return processedImg

#Basic CNN for adversarial image generation
class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = Conv2d(3, 16, kernel_size = 3, padding = 1)
        self.dropout = Dropout(p=0.3)
        self.conv2 = Conv2d(16, 32, kernel_size = 3, padding = 1)
        self.pooling = MaxPool2d(2,2)
        self.relu = ReLU()
        self.flatten = Flatten()
        self.linear1 = Linear(2048, 128)
        self.linear2 = Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x) #Convolution layer
        x = self.pooling(x) #Max Pooling Layer
        x = self.dropout(x) #Dropout Layer
        x = self.conv2(x) #Second Convolution Layer
        x = self.pooling(x) #Second Pooling Layer
        x = self.flatten(x) #Flatten Layer
        x = self.relu(self.linear1(x)) #Regular Layer
        x = self.dropout(x) #Second Dropout Layer
        x = self.linear2(x) #Output Layer
        return x

# CIFAR-10 classes
class CIFAR10:
    classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    itos = {i: s for i, s in enumerate(classes)}
    stoi = {s: i for i, s in itos.items()}

def plot_tensors(tensors, ncols=3, colorbar=True, log_scale=False, titles=None, figsize=None, **kwargs):
    """
    Custom imshow for multiple tensors: negative=navy, zero=white, positive=maroon
    
    Parameters:
    -----------
    tensors : list of array-like
        List of 2D tensors to display
    ncols : int, default=3
        Number of columns in the grid
    colorbar : bool, default=True
        Whether to show colorbars
    log_scale : bool, default=False
        Whether to use symmetric log scale for colorbar
    titles : list of str, optional
        Custom titles for each tensor. If None, uses "Tensor 1", "Tensor 2", etc.
    figsize : tuple, optional
        Figure size (width, height)
    **kwargs : dict
        Additional arguments passed to imshow
    """
    
    # Convert to list if single tensor passed
    if not isinstance(tensors, (list, tuple)):
        tensors = [tensors]
    
    # Convert all to numpy arrays
    tensors = [np.asarray(t) for t in tensors]
    
    # Assert all tensors have the same shape
    shapes = [t.shape for t in tensors]
    assert all(shape == shapes[0] for shape in shapes), \
        f"All tensors must have the same shape. Got shapes: {shapes}"
    
    # Assert titles length matches number of tensors if provided
    if titles is not None:
        assert len(titles) == len(tensors), \
            f"Number of titles ({len(titles)}) must match number of tensors ({len(tensors)})"
    
    # Calculate grid dimensions
    ntensors = len(tensors)
    nrows = math.ceil(ntensors / ncols)
    
    # Calculate global vmin/vmax for consistent scaling
    all_values = np.concatenate([t.flatten() for t in tensors])
    vmax = np.max(np.abs(all_values))
    
    # Create colormap
    cmap = mcolors.LinearSegmentedColormap.from_list('custom', ['navy', 'white', 'maroon'])
    
    # Set up plot defaults
    defaults = {'cmap': cmap}
    
    # Add normalization
    if log_scale:
        # Use symmetric log norm to handle negative, zero, and positive values
        # linthresh determines the linear threshold around zero
        linthresh = vmax / 100  # Linear region is 1% of max value
        defaults['norm'] = mcolors.SymLogNorm(linthresh=linthresh, vmin=-vmax, vmax=vmax)
    else:
        # Use regular linear normalization
        defaults.update({'vmin': -vmax, 'vmax': vmax})
    
    defaults.update(kwargs)
    
    # Create figure and subplots
    if figsize is None:
        figsize = (4 * ncols, 4 * nrows)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    
    # Handle case where we have only one subplot
    if ntensors == 1:
        axes = [axes]
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)
    
    # Flatten axes for easy iteration
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else axes
    
    # Plot each tensor
    images = []
    for i, tensor in enumerate(tensors):
        ax = axes_flat[i]
        im = ax.imshow(tensor, **defaults)
        
        # Use custom title if provided, otherwise default
        title = titles[i] if titles is not None else f'Tensor {i+1}'
        ax.set_title(title)
        
        images.append(im)
        
        if colorbar:
            plt.colorbar(im, ax=ax, shrink=0.6, aspect=20)
    
    # Hide unused subplots
    for i in range(ntensors, len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    plt.tight_layout()
    return fig, images


def plot_2d(x, y, x_range=None, y_range=None, title=None, figsize=(8, 6), **kwargs):
    """
    Plot 2D data with maroon color scheme.
    
    Parameters:
    -----------
    x : array-like
        X coordinates/values
    y : array-like
        Y coordinates/values
    x_range : tuple, optional
        X-axis range as (min, max). If None, uses data range.
    y_range : tuple, optional
        Y-axis range as (min, max). If None, uses data range.
    title : str, optional
        Plot title. If None, no title is set.
    figsize : tuple, default=(8, 6)
        Figure size (width, height)
    **kwargs : dict
        Additional arguments passed to plot function
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    ax : matplotlib.axes.Axes
        The axes object
    """
    
    # Convert to numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Assert x and y have the same length
    assert len(x) == len(y), f"x and y must have the same length. Got x: {len(x)}, y: {len(y)}"
    
    # Set up plot defaults with maroon color
    defaults = {'color': 'maroon', 'linewidth': 2}
    defaults.update(kwargs)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the data
    line = ax.plot(x, y, **defaults)
    
    # Set ranges if provided
    if x_range is not None:
        ax.set_xlim(x_range)
    if y_range is not None:
        ax.set_ylim(y_range)
    
    # Set title if provided
    if title is not None:
        ax.set_title(title)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3)
    
    # Labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    plt.tight_layout()
    return fig, ax

def f_6(logits, target, k=0.1):
    i_neq_t = torch.argmax(logits)
    if i_neq_t == target:
        i_neq_t = torch.argmax(torch.cat([logits[:target], logits[target+1:]]))
    return torch.max(logits[i_neq_t] - logits[target], -torch.tensor(k))          

def CW_targeted_l2(img, model, c, target, k=0.1, l2_limit=0.5, num_iters=100):

    device = next(model.parameters()).device
    print(f"Using device: {device} for testing...")
    
    cw_weights = torch.randn_like(img).to(device) * 0.001
    cw_weights.requires_grad = True
    optimizer = optim.Adam([cw_weights], lr=5e-2)

    delta = 0.5 * (F.tanh(cw_weights) + 1) - img

    for _ in range(num_iters):
        
        logits = model(img + delta)

        if torch.argmax(logits[0]) == target and torch.sum((delta)**2).item() <= l2_limit:
            # print(torch.sum((delta)**2).item())
            # print(l2_limit)
            return img + delta

        if f_6(logits[0], target, k) < -k:
            print(logits[0])
            print(target)
            print(k)
            print(f_6(logits[0], target, k))
    
        success_loss = c * f_6(logits[0], target, k)
        l2_reg = torch.sum((delta)**2)

        loss = success_loss + l2_reg
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        delta = 0.5 * (F.tanh(cw_weights) + 1) - img

    # print(torch.sum((delta)**2).item())
    # print(l2_limit)
    print("warning! targeted attack was not successful")
    return img + delta
