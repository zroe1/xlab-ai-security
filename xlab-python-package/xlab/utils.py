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
from torchvision import datasets, transforms
from PIL import Image
import os
import pkg_resources
from xlab.models import BlackBox

def load_cifar10_test_samples(n, download=True, transform=None, data_dir='./data'):
    """
    Load the first n test set examples from CIFAR-10.
    
    This function provides a convenient way to load a subset of CIFAR-10 test data
    for experimentation and educational purposes, particularly useful for adversarial
    attacks and security research.
    
    Parameters:
    -----------
    n : int
        Number of test samples to load. If n exceeds the test set size (10,000),
        all available samples will be returned.
    download : bool, default=True
        Whether to download CIFAR-10 if not already present.
    transform : torchvision.transforms, optional
        Optional transform to apply to the images. If None, applies standard
        transforms (ToTensor and Normalize) suitable for most models.
    data_dir : str, default='./data'
        Directory to store/load CIFAR-10 data.
    
    Returns:
    --------
    images : torch.Tensor
        Tensor of shape (n, 3, 32, 32) containing the image data.
        Values are normalized to [0, 1] if using default transform.
    labels : torch.Tensor
        Tensor of shape (n,) containing the integer labels (0-9).
    
    Examples:
    --------
    >>> # Load first 100 test samples with default transforms
    >>> images, labels = load_cifar10_test_samples(100)
    >>> print(f"Images shape: {images.shape}")  # torch.Size([100, 3, 32, 32])
    >>> print(f"Labels shape: {labels.shape}")  # torch.Size([100])
    
    >>> # Load with custom transforms
    >>> from torchvision import transforms
    >>> custom_transform = transforms.Compose([
    ...     transforms.ToTensor(),
    ...     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # [-1, 1] range
    ... ])
    >>> images, labels = load_cifar10_test_samples(50, transform=custom_transform)
    
    Notes:
    ------
    - CIFAR-10 test set contains 10,000 samples total
    - Default transform normalizes to [0, 1] range using ToTensor()
    - For adversarial attacks, you may want to use [-1, 1] normalization
    - Use CIFAR10.itos to convert integer labels to class names
    """
    # Set default transform if none provided
    if transform is None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            # Note: ToTensor() automatically normalizes PIL Images to [0, 1]
        ])
    
    # Load CIFAR-10 test dataset
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,  # Use test set
        download=download,
        transform=transform
    )
    
    # Ensure n doesn't exceed dataset size
    n = min(n, len(test_dataset))
    
    if n <= 0:
        raise ValueError("n must be a positive integer")
    
    # Extract first n samples
    images = []
    labels = []
    
    for i in range(n):
        image, label = test_dataset[i]
        images.append(image)
        labels.append(label)
    
    # Stack into tensors
    images = torch.stack(images)
    labels = torch.tensor(labels, dtype=torch.long)
    
    return images, labels


def load_mnist_test_samples(n, download=True, transform=None, data_dir='./data'):
    """
    Load the first n test set examples from MNIST.
    
    This function provides a convenient way to load a subset of MNIST test data
    for experimentation and educational purposes, particularly useful for adversarial
    attacks and security research.
    
    Parameters:
    -----------
    n : int
        Number of test samples to load. If n exceeds the test set size (10,000),
        all available samples will be returned.
    download : bool, default=True
        Whether to download MNIST if not already present.
    transform : torchvision.transforms, optional
        Optional transform to apply to the images. If None, applies standard
        transforms (ToTensor) suitable for most models.
    data_dir : str, default='./data'
        Directory to store/load MNIST data.
    
    Returns:
    --------
    images : torch.Tensor
        Tensor of shape (n, 1, 28, 28) containing the image data.
        Values are normalized to [0, 1] if using default transform.
    labels : torch.Tensor
        Tensor of shape (n,) containing the integer labels (0-9).
    
    Examples:
    --------
    >>> # Load first 100 test samples with default transforms
    >>> images, labels = load_mnist_test_samples(100)
    >>> print(f"Images shape: {images.shape}")  # torch.Size([100, 1, 28, 28])
    >>> print(f"Labels shape: {labels.shape}")  # torch.Size([100])
    
    >>> # Load with custom transforms
    >>> from torchvision import transforms
    >>> custom_transform = transforms.Compose([
    ...     transforms.ToTensor(),
    ...     transforms.Normalize((0.5,), (0.5,))  # [-1, 1] range for grayscale
    ... ])
    >>> images, labels = load_mnist_test_samples(50, transform=custom_transform)
    
    Notes:
    ------
    - MNIST test set contains 10,000 samples total
    - Images are 28x28 grayscale (1 channel)
    - Default transform normalizes to [0, 1] range using ToTensor()
    - For adversarial attacks, you may want to use [-1, 1] normalization
    - Labels are digits 0-9
    """
    # Set default transform if none provided
    if transform is None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            # Note: ToTensor() automatically normalizes PIL Images to [0, 1]
        ])
    
    # Load MNIST test dataset
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,  # Use test set
        download=download,
        transform=transform
    )
    
    # Ensure n doesn't exceed dataset size
    n = min(n, len(test_dataset))
    
    if n <= 0:
        raise ValueError("n must be a positive integer")
    
    # Extract first n samples
    images = []
    labels = []
    
    for i in range(n):
        image, label = test_dataset[i]
        images.append(image)
        labels.append(label)
    
    # Stack into tensors
    images = torch.stack(images)
    labels = torch.tensor(labels, dtype=torch.long)
    
    return images, labels


def load_mnist_train_samples(n, download=True, transform=None, data_dir='./data'):
    """
    Load the first n training set examples from MNIST.

    This function mirrors `load_mnist_test_samples` but pulls data from the
    training split of the dataset (60,000 samples) instead of the test split.

    Parameters:
    -----------
    n : int
        Number of training samples to load. If n exceeds the train set size
        (60,000), all available samples will be returned.
    download : bool, default=True
        Whether to download MNIST if not already present.
    transform : torchvision.transforms, optional
        Optional transform to apply to the images. If None, applies standard
        transforms (ToTensor) suitable for most models.
    data_dir : str, default='./data'
        Directory to store/load MNIST data.

    Returns:
    --------
    images : torch.Tensor
        Tensor of shape (n, 1, 28, 28) containing the image data.
    labels : torch.Tensor
        Tensor of shape (n,) containing the integer labels (0-9).
    """

    # Set default transform if none provided
    if transform is None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            # ToTensor() normalizes PIL Images to [0, 1]
        ])

    # Load MNIST training dataset
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,  # Use training set
        download=download,
        transform=transform
    )

    # Ensure n doesn't exceed dataset size
    n = min(n, len(train_dataset))

    if n <= 0:
        raise ValueError("n must be a positive integer")

    # Extract first n samples
    images = []
    labels = []

    for i in range(n):
        image, label = train_dataset[i]
        images.append(image)
        labels.append(label)

    # Stack into tensors
    images = torch.stack(images)
    labels = torch.tensor(labels, dtype=torch.long)

    return images, labels

# ---------------------------------------------------------------------------
# DataLoader helper for MNIST (training split)
# ---------------------------------------------------------------------------

def get_mnist_train_loader(
    batch_size=64,
    shuffle=True,
    download=True,
    transform=None,
    data_dir='./data',
):
    """
    Create a PyTorch ``DataLoader`` for the MNIST training set.

    Parameters
    ----------
    batch_size : int, default=64
        Number of samples per batch to load.
    shuffle : bool, default=True
        Whether to shuffle the dataset each epoch.
    download : bool, default=True
        Download the dataset if it is not present locally.
    transform : torchvision.transforms, optional
        Transformations to apply to each image. If ``None``, a default
        ``ToTensor`` transform is applied that scales pixel values to ``[0,1]``.
    data_dir : str, default='./data'
        Directory where the MNIST data is stored / will be downloaded to.

    Returns
    -------
    torch.utils.data.DataLoader
        A DataLoader yielding batches of ``(images, labels)`` where ``images``
        has shape ``(batch_size, 1, 28, 28)`` and ``labels`` has shape
        ``(batch_size,)``.
    """

    # Default transform
    if transform is None:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    # Load the training dataset
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=download,
        transform=transform,
    )

    # Construct the DataLoader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    return train_loader


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


def PGD_generator(model, loss_fn, path, y, epsilon=1 / 1000, alpha=0.0005, num_iters=6):
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
    with torch.no_grad():  # Stops calculating gradients
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


def show_grayscale_image(img, title=None, figsize=(3, 3)):
    """
    Display a grayscale image tensor (e.g., MNIST digits) using matplotlib.
    
    This function is specifically designed for displaying grayscale images like
    MNIST digits with proper formatting and visualization.
    
    Parameters:
    -----------
    img : torch.Tensor
        Grayscale image tensor. Can be either:
        - Shape [1, 28, 28] (with channel dimension)  
        - Shape [28, 28] (without channel dimension)
        Values should be in range [0, 1].
    title : str, optional
        Title to display above the image.
    figsize : tuple, default=(3, 3)
        Figure size (width, height) in inches.
    
    Examples:
    --------
    >>> # Display single MNIST digit
    >>> mnist_images, labels = load_mnist_test_samples(1)
    >>> show_grayscale_image(mnist_images[0], title=f"Digit: {labels[0]}")
    
    >>> # Display with custom size
    >>> show_grayscale_image(mnist_images[0], title="MNIST Sample", figsize=(2, 2))
    
    >>> # Works with [28, 28] tensors too
    >>> digit_28x28 = mnist_images[0].squeeze(0)  # Remove channel dim
    >>> show_grayscale_image(digit_28x28)
    """
    # Convert to numpy and handle different input shapes
    if isinstance(img, torch.Tensor):
        img_np = img.detach().cpu().numpy()
    else:
        img_np = img
    
    # Handle different tensor shapes
    if img_np.ndim == 3:  # [1, 28, 28] or [28, 28, 1]
        if img_np.shape[0] == 1:  # [1, 28, 28]
            img_np = img_np.squeeze(0)  # Remove first dimension
        elif img_np.shape[2] == 1:  # [28, 28, 1]
            img_np = img_np.squeeze(2)  # Remove last dimension
        else:
            raise ValueError(f"Unexpected 3D tensor shape: {img_np.shape}. Expected [1, 28, 28] or [28, 28, 1]")
    elif img_np.ndim != 2:  # Should be [28, 28]
        raise ValueError(f"Expected 2D or 3D tensor, got {img_np.ndim}D with shape {img_np.shape}")
    
    # Create the plot
    plt.figure(figsize=figsize)
    plt.imshow(img_np, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')  # Remove axes for cleaner display
    
    if title is not None:
        plt.title(title, fontsize=12, pad=5)
    
    plt.tight_layout()
    plt.show()


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
    img = load_sample_image(path)
    transform = Compose([Resize((32, 32)), ToTensor()])
    processedImg = transform(img)
    processedImg = processedImg.unsqueeze(0)
    return processedImg


def load_sample_image(image_name, return_path=False):
    """
    Load a sample image included with the xlab package.
    
    This function provides access to sample images bundled with the package,
    useful for testing adversarial attacks and other image processing tasks.
    
    Parameters:
    -----------
    image_name : str
        Name of the image file to load. Available images:
        - 'cat.jpg': Sample cat image for testing
        - 'car.jpg': Sample car image for testing  
        - 'frog.jpg': Sample frog image for testing
    return_path : bool, default=False
        If True, returns the file path instead of loading the image.
        Useful if you need the path for other functions.
    
    Returns:
    --------
    image : PIL.Image or str
        If return_path=False: PIL Image object
        If return_path=True: String path to the image file
    
    Examples:
    --------
    >>> # Load cat image as PIL Image
    >>> cat_img = load_sample_image('cat.jpg')
    >>> print(type(cat_img))  # <class 'PIL.Image.Image'>
    
    >>> # Get path to image file
    >>> cat_path = load_sample_image('cat.jpg', return_path=True)
    >>> print(cat_path)  # /path/to/package/data/cat.jpg
    
    >>> # Use with other functions
    >>> cat_tensor = process_image(load_sample_image('cat.jpg', return_path=True))
    
    Raises:
    -------
    FileNotFoundError
        If the specified image file doesn't exist in the package.
    """
    try:
        # Get the path to the data directory in the package
        data_path = pkg_resources.resource_filename('xlab', f'data/{image_name}')
        
        if not os.path.exists(data_path):
            available_images = ['cat.jpg']
            raise FileNotFoundError(
                f"Image '{image_name}' not found. Available images: {available_images}"
            )
        
        if return_path:
            return data_path
        else:
            return Image.open(data_path)
            
    except Exception as e:
        raise FileNotFoundError(
            f"Could not load image '{image_name}': {str(e)}"
        )


# Basic CNN for adversarial image generation
class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = Conv2d(3, 16, kernel_size=3, padding=1)
        self.dropout = Dropout(p=0.3)
        self.conv2 = Conv2d(16, 32, kernel_size=3, padding=1)
        self.pooling = MaxPool2d(2, 2)
        self.relu = ReLU()
        self.flatten = Flatten()
        self.linear1 = Linear(2048, 128)
        self.linear2 = Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)  # Convolution layer
        x = self.pooling(x)  # Max Pooling Layer
        x = self.dropout(x)  # Dropout Layer
        x = self.conv2(x)  # Second Convolution Layer
        x = self.pooling(x)  # Second Pooling Layer
        x = self.flatten(x)  # Flatten Layer
        x = self.relu(self.linear1(x))  # Regular Layer
        x = self.dropout(x)  # Second Dropout Layer
        x = self.linear2(x)  # Output Layer
        return x


# CIFAR-10 classes
class CIFAR10:
    classes = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]
    itos = {i: s for i, s in enumerate(classes)}
    stoi = {s: i for i, s in itos.items()}


def plot_tensors(
    tensors,
    ncols=3,
    colorbar=True,
    log_scale=False,
    titles=None,
    figsize=None,
    **kwargs,
):
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
    assert all(shape == shapes[0] for shape in shapes), (
        f"All tensors must have the same shape. Got shapes: {shapes}"
    )

    # Assert titles length matches number of tensors if provided
    if titles is not None:
        assert len(titles) == len(tensors), (
            f"Number of titles ({len(titles)}) must match number of tensors ({len(tensors)})"
        )

    # Calculate grid dimensions
    ntensors = len(tensors)
    nrows = math.ceil(ntensors / ncols)

    # Calculate global vmin/vmax for consistent scaling
    all_values = np.concatenate([t.flatten() for t in tensors])
    vmax = np.max(np.abs(all_values))

    # Create colormap
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "custom", ["navy", "white", "maroon"]
    )

    # Set up plot defaults
    defaults = {"cmap": cmap}

    # Add normalization
    if log_scale:
        # Use symmetric log norm to handle negative, zero, and positive values
        # linthresh determines the linear threshold around zero
        linthresh = vmax / 100  # Linear region is 1% of max value
        defaults["norm"] = mcolors.SymLogNorm(
            linthresh=linthresh, vmin=-vmax, vmax=vmax
        )
    else:
        # Use regular linear normalization
        defaults.update({"vmin": -vmax, "vmax": vmax})

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
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else axes

    # Plot each tensor
    images = []
    for i, tensor in enumerate(tensors):
        ax = axes_flat[i]
        im = ax.imshow(tensor, **defaults)

        # Use custom title if provided, otherwise default
        title = titles[i] if titles is not None else f"Tensor {i + 1}"
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
    assert len(x) == len(y), (
        f"x and y must have the same length. Got x: {len(x)}, y: {len(y)}"
    )

    # Set up plot defaults with maroon color
    defaults = {"color": "maroon", "linewidth": 2}
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
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    plt.tight_layout()
    return fig, ax


class BlackBoxModelWrapper:
    """
    A wrapper for black box models that hides implementation details.
    
    This class downloads and loads a pre-trained model from HuggingFace Hub
    and provides a simple interface for making predictions while keeping
    the actual model hidden from casual inspection.
    
    Examples:
    --------
    >>> # Load MNIST black box model
    >>> black_box = BlackBoxModelWrapper('mnist-black-box')
    >>> 
    >>> # Make predictions (expects MNIST format: 1x28x28)
    >>> predictions = black_box.predict(mnist_images)
    >>> class_probs = black_box.predict_proba(mnist_images)
    """
    
    def __init__(self, model_type='mnist-black-box', device='cpu'):
        """
        Initialize the black box model wrapper.
        
        Parameters:
        -----------
        model_type : str, default='mnist-black-box'
            Type of model to load. Currently supports:
            - 'mnist-black-box': Pre-trained MNIST classifier
        device : str, default='cpu' 
            Device to run the model on ('cpu' or 'cuda')
        """
        self._device = torch.device(device)
        self._model_type = model_type
        self._model = None
        self._is_loaded = False
        
        # Load the model
        self._load_model()
    
    def _load_model(self):
        """Internal method to download and load the model."""
        try:
            if self._model_type == 'mnist-black-box':
                from huggingface_hub import hf_hub_download
                
                # Download the model file
                model_path = hf_hub_download(
                    repo_id="uchicago-xlab-ai-security/mnist-ensemble",
                    filename="mnist_black_box_mlp.pth"
                )
                
                # Load the model
                self._model = torch.load(model_path, map_location=self._device, weights_only=False)
                self._model.eval()
                self._is_loaded = True
                
            else:
                raise ValueError(f"Unknown model type: {self._model_type}")
                
        except Exception as e:
            raise RuntimeError(f"Failed to load black box model: {str(e)}")
    
    def predict(self, x):
        """
        Make predictions on input data.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor. For MNIST models, should be shape (batch_size, 1, 28, 28)
            or (1, 28, 28) for single image.
        
        Returns:
        --------
        predictions : torch.Tensor
            Predicted class labels (integers)
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded")
        
        # Ensure input is on correct device
        if x.device != self._device:
            x = x.to(self._device)
        
        # Add batch dimension if needed
        if x.dim() == 3:  # (1, 28, 28)
            x = x.unsqueeze(0)  # (1, 1, 28, 28)
        
        with torch.no_grad():
            logits = self._model(x)
            predictions = torch.argmax(logits, dim=1)
        
        return predictions
    
    def predict_proba(self, x):
        """
        Get prediction probabilities for input data.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor. For MNIST models, should be shape (batch_size, 1, 28, 28)
            or (1, 28, 28) for single image.
        
        Returns:
        --------
        probabilities : torch.Tensor  
            Class probabilities (softmax applied)
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded")
        
        # Ensure input is on correct device
        if x.device != self._device:
            x = x.to(self._device)
        
        # Add batch dimension if needed
        if x.dim() == 3:  # (1, 28, 28)
            x = x.unsqueeze(0)  # (1, 1, 28, 28)
        
        with torch.no_grad():
            logits = self._model(x)
            probabilities = torch.softmax(logits, dim=1)
        
        return probabilities
    
    def __call__(self, x):
        """Allow the wrapper to be called like a function."""
        return self.predict(x)
    
    def __repr__(self):
        """Custom representation that doesn't reveal internal model details."""
        return f"BlackBoxModelWrapper(type='{self._model_type}', device='{self._device}')"
    
    @property
    def device(self):
        """Get the device the model is running on."""
        return self._device
    
    @property
    def model_type(self):
        """Get the type of model loaded."""
        return self._model_type


def load_black_box_model(model_type='mnist-black-box', device='cpu'):
    """
    Convenience function to load a black box model.
    
    This function provides a simple way to load and use pre-trained models
    without exposing the underlying implementation details.
    
    Parameters:
    -----------
    model_type : str, default='mnist-black-box'
        Type of model to load. Currently supports:
        - 'mnist-black-box': Pre-trained MNIST classifier  
    device : str, default='cpu'
        Device to run the model on ('cpu' or 'cuda')
    
    Returns:
    --------
    model : BlackBoxModelWrapper
        Wrapped model that can make predictions
    
    Examples:
    --------
    >>> # Load model and make predictions
    >>> model = load_black_box_model('mnist-black-box')
    >>> predictions = model.predict(mnist_data)
    >>> probabilities = model.predict_proba(mnist_data) 
    
    >>> # Use as a function
    >>> predictions = model(mnist_data)
    """
    return BlackBoxModelWrapper(model_type=model_type, device=device)


def get_best_device():
    """
    Get the best available PyTorch device for the current system.
    
    This function automatically detects and returns the best available device
    in order of preference: CUDA (NVIDIA GPU) > MPS (Apple Silicon) > CPU.
    
    Returns:
    --------
    device : torch.device
        The best available device for PyTorch operations.
        
    Examples:
    --------
    >>> # Get best device and use it
    >>> device = get_best_device()
    >>> print(f"Using device: {device}")
    >>> tensor = torch.randn(3, 3).to(device)
    
    >>> # Use with models
    >>> model = MyModel().to(get_best_device())
    
    >>> # Use with black box model
    >>> device_str = str(get_best_device())  # Convert to string
    >>> model = load_black_box_model('mnist-black-box', device=device_str)
    
    Notes:
    ------
    - CUDA: For NVIDIA GPUs (fastest for most deep learning tasks)
    - MPS: For Apple Silicon Macs (M1, M2, etc.)
    - CPU: Fallback option, always available
    - The function checks device availability, not just existence
    """
    # Check for CUDA (NVIDIA GPU)
    if torch.cuda.is_available():
        return torch.device('cuda')
    
    # Check for MPS (Apple Silicon)
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    
    # Fallback to CPU
    else:
        return torch.device('cpu')


def f_6(logits, target, k=0.1):
    i_neq_t = torch.argmax(logits)
    if i_neq_t == target:
        i_neq_t = torch.argmax(torch.cat([logits[:target], logits[target + 1 :]]))
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

        if (
            torch.argmax(logits[0]) == target
            and torch.sum((delta) ** 2).item() <= l2_limit
        ):
            # print(torch.sum((delta)**2).item())
            # print(l2_limit)
            return img + delta

        if f_6(logits[0], target, k) < -k:
            print(logits[0])
            print(target)
            print(k)
            print(f_6(logits[0], target, k))

        success_loss = c * f_6(logits[0], target, k)
        l2_reg = torch.sum((delta) ** 2)

        loss = success_loss + l2_reg
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        delta = 0.5 * (F.tanh(cw_weights) + 1) - img

    # print(torch.sum((delta)**2).item())
    # print(l2_limit)
    print("warning! targeted attack was not successful")
    return img + delta


def plot_dual_2d(
    x,
    y1,
    y2,
    y1_label=None,
    y2_label=None,
    x_label="X",
    y1_axis_label="Y1",
    y2_axis_label="Y2",
    title=None,
    figsize=(8, 6),
    log_x=False,
    **kwargs,
):
    """
    Plot two lines in 2D with separate y-axes for different ranges.

    Parameters:
    -----------
    x : array-like
        X coordinates/values (shared by both lines)
    y1 : array-like
        Y coordinates/values for first line (left y-axis)
    y2 : array-like
        Y coordinates/values for second line (right y-axis)
    y1_label : str, optional
        Label for the first line
    y2_label : str, optional
        Label for the second line
    x_label : str, default='X'
        Label for x-axis
    y1_axis_label : str, default='Y1'
        Label for left y-axis
    y2_axis_label : str, default='Y2'
        Label for right y-axis
    title : str, optional
        Plot title. If None, no title is set.
    figsize : tuple, default=(8, 6)
        Figure size (width, height)
    log_x : bool, default=False
        Whether to use logarithmic scale for x-axis
    **kwargs : dict
        Additional arguments passed to plot functions

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    ax1 : matplotlib.axes.Axes
        The left y-axis object
    ax2 : matplotlib.axes.Axes
        The right y-axis object
    """

    # Convert to numpy arrays
    x = np.asarray(x)
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)

    # Assert x and both y arrays have the same length
    assert len(x) == len(y1) == len(y2), (
        f"x, y1, and y2 must have the same length. Got x: {len(x)}, y1: {len(y1)}, y2: {len(y2)}"
    )

    # Set up plot defaults with consistent colors
    defaults1 = {"color": "#4871cf", "linewidth": 3, "markersize": 6}
    defaults2 = {"color": "maroon", "linewidth": 3, "markersize": 6}

    # Update with any user-provided kwargs
    if "color1" in kwargs:
        defaults1["color"] = kwargs.pop("color1")
    if "color2" in kwargs:
        defaults2["color"] = kwargs.pop("color2")

    # Apply remaining kwargs to both lines
    defaults1.update(kwargs)
    defaults2.update(kwargs)

    # Create figure and primary axis
    fig, ax1 = plt.subplots(figsize=figsize)

    # Plot first line on left y-axis
    line1 = ax1.plot(x, y1, label=y1_label, **defaults1)
    ax1.set_xlabel(x_label, fontsize=12)
    ax1.set_ylabel(y1_axis_label, fontsize=12, color=defaults1["color"])
    ax1.tick_params(axis="y", labelcolor=defaults1["color"])
    ax1.set_ylim(bottom=0)  # Ensure left y-axis starts at 0

    # Create second y-axis
    ax2 = ax1.twinx()

    # Plot second line on right y-axis
    line2 = ax2.plot(x, y2, label=y2_label, **defaults2)
    ax2.set_ylabel(y2_axis_label, fontsize=12, color=defaults2["color"])
    ax2.tick_params(axis="y", labelcolor=defaults2["color"])
    ax2.set_ylim(bottom=0)  # Ensure right y-axis starts at 0

    # Add grid for better readability
    ax1.grid(True, ls="--", alpha=0.7)

    # Set logarithmic x-axis if requested
    if log_x:
        ax1.set_xscale("log")

    # Set title if provided
    if title is not None:
        ax1.set_title(title, fontsize=14)

    # Create combined legend
    lines = line1 + line2
    labels = [
        l.get_label() for l in lines if l.get_label() and l.get_label() != "_line0"
    ]
    if labels:
        ax1.legend(lines, labels, loc="lower right")

    plt.tight_layout()
    return fig, ax1, ax2

def clip(x, x_original, epsilon):
    """
    Clips adversarial perturbations to stay within epsilon-ball of original
    image and ensures valid pixel values between 0 and 1.

    Args:
        x (Tensor): perturbed image tensor to be clipped
        x_original (Tensor): original unperturbed image tensor
        epsilon (float): maximum allowed perturbation magnitude

    Returns [1, 3, 32, 32]: clipped image tensor with perturbations bounded
        by epsilon and pixel values clamped to [0, 1] range
    """
    
    x_final = None

    diff = x - x_original
    
    # 1. Clip x epsilon distance away
    diff = torch.clamp(diff, -epsilon, epsilon)
    x_clipped = x_original + diff

    # 2. Clip x between 0 and 1
    x_final = torch.clamp(x_clipped, 0, 1)

    return x_final
