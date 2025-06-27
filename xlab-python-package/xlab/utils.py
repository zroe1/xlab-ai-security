import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import math
import torch
import torch.nn.functional as F
import torch.optim as optim

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