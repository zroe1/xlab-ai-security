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
from importlib import resources
from xlab.models import BlackBox
from openai import OpenAI


def load_cifar10_test_samples(n, download=True, transform=None, data_dir="./data"):
    """Loads the first n test set examples from CIFAR-10.

    Args:
        n (int): Number of test samples to load.
        download (bool): Whether to download CIFAR-10 if not already present.
        transform (torchvision.transforms): Optional transform to apply to images.
        data_dir (str): Directory to store/load CIFAR-10 data.

    Returns:
        tuple: (images [n, 3, 32, 32], labels [n]) tensor pair.
    """
    # Set default transform if none provided
    if transform is None:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # Note: ToTensor() automatically normalizes PIL Images to [0, 1]
            ]
        )

    # Load CIFAR-10 test dataset
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,  # Use test set
        download=download,
        transform=transform,
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


def load_mnist_test_samples(n, download=True, transform=None, data_dir="./data"):
    """Loads the first n test set examples from MNIST.

    Args:
        n (int): Number of test samples to load.
        download (bool): Whether to download MNIST if not already present.
        transform (torchvision.transforms): Optional transform to apply to images.
        data_dir (str): Directory to store/load MNIST data.

    Returns:
        tuple: (images [n, 1, 28, 28], labels [n]) tensor pair.
    """
    # Set default transform if none provided
    if transform is None:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # Note: ToTensor() automatically normalizes PIL Images to [0, 1]
            ]
        )

    # Load MNIST test dataset
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,  # Use test set
        download=download,
        transform=transform,
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


def load_mnist_train_samples(n, download=True, transform=None, data_dir="./data"):
    """Loads the first n training set examples from MNIST.

    Args:
        n (int): Number of training samples to load.
        download (bool): Whether to download MNIST if not already present.
        transform (torchvision.transforms): Optional transform to apply to images.
        data_dir (str): Directory to store/load MNIST data.

    Returns:
        tuple: (images [n, 1, 28, 28], labels [n]) tensor pair.
    """

    # Set default transform if none provided
    if transform is None:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # ToTensor() normalizes PIL Images to [0, 1]
            ]
        )

    # Load MNIST training dataset
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,  # Use training set
        download=download,
        transform=transform,
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
    data_dir="./data",
):
    """Creates a PyTorch DataLoader for the MNIST training set.

    Args:
        batch_size (int): Number of samples per batch to load.
        shuffle (bool): Whether to shuffle the dataset each epoch.
        download (bool): Download the dataset if not present locally.
        transform (torchvision.transforms): Transformations to apply to each image.
        data_dir (str): Directory where MNIST data is stored/downloaded.

    Returns:
        torch.utils.data.DataLoader: DataLoader for MNIST training data.
    """

    # Default transform
    if transform is None:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

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
    """Adds Gaussian noise to an image tensor.

    Args:
        img [*]: Image tensor to add noise to.
        stdev (float): Standard deviation of Gaussian noise.
        mean (float): Mean of Gaussian noise.

    Returns:
        [*]: Noisy image tensor with same shape as input.
    """
    noise = torch.randn_like(img) * stdev + mean
    return img + noise


def PGD_generator(model, loss_fn, path, y, epsilon=1 / 1000, alpha=0.0005, num_iters=6):
    """Creates adversarial image using PGD attack.

    Args:
        model (torch.nn.Module): PyTorch model for classification.
        loss_fn (torch.nn.Module): Loss function for attack.
        path (str): Filepath to input image.
        y (torch.Tensor): Target label tensor.
        epsilon (float): Maximum perturbation magnitude.
        alpha (float): Step size for each iteration.
        num_iters (int): Number of PGD iterations.

    Returns:
        [1, 3, 32, 32]: Adversarially perturbed image tensor.
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
        x = clip(x, x, epsilon)
    return x


def prediction(model, img):
    """Returns prediction class and confidence for an image.

    Args:
        model (torch.nn.Module): PyTorch model for classification.
        img [*]: Image tensor to predict.

    Returns:
        tuple: (pred_class, confidence) tensor pair.
    """
    with torch.no_grad():  # Stops calculating gradients
        prediction = model(img)
        _, pred_class = torch.max(prediction, 1)
    probs = prediction.softmax(dim=-1)
    return pred_class, probs[0][pred_class]


def show_image(img):
    """Displays image tensor using matplotlib.

    Args:
        img [1, 3, H, W]: Image tensor to display.

    Returns:
        None: Displays image plot.
    """
    img = img.squeeze(0)
    plt.imshow(img.permute(1, 2, 0).detach().numpy())


def show_grayscale_image(img, title=None, figsize=(3, 3)):
    """Displays grayscale image tensor using matplotlib.

    Args:
        img [1, 28, 28] or [28, 28]: Grayscale image tensor.
        title (str): Optional title for the image.
        figsize (tuple): Figure size (width, height) in inches.

    Returns:
        None: Displays grayscale image plot.
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
            raise ValueError(
                f"Unexpected 3D tensor shape: {img_np.shape}. Expected [1, 28, 28] or [28, 28, 1]"
            )
    elif img_np.ndim != 2:  # Should be [28, 28]
        raise ValueError(
            f"Expected 2D or 3D tensor, got {img_np.ndim}D with shape {img_np.shape}"
        )

    # Create the plot
    plt.figure(figsize=figsize)
    plt.imshow(img_np, cmap="gray", vmin=0, vmax=1)
    plt.axis("off")  # Remove axes for cleaner display

    if title is not None:
        plt.title(title, fontsize=12, pad=5)

    plt.tight_layout()
    plt.show()


def process_image(path):
    """Converts image file to scaled torch tensor.

    Args:
        path (str): Filepath to image file.

    Returns:
        [1, 3, 32, 32]: Processed image tensor.
    """
    img = load_sample_image(path)
    transform = Compose([Resize((32, 32)), ToTensor()])
    processedImg = transform(img)
    processedImg = processedImg.unsqueeze(0)
    return processedImg


def load_sample_image(image_name, return_path=False):
    """Loads a sample image included with the xlab package.

    Args:
        image_name (str): Name of image file (e.g., 'cat.jpg').
        return_path (bool): If True, returns file path instead of PIL Image.

    Returns:
        PIL.Image or str: Loaded PIL Image or path string.
    """
    try:
        # Get the path to the data directory in the package
        data_path = resources.files("xlab").joinpath("data", image_name)

        if not os.path.exists(data_path):
            available_images = ["cat.jpg"]
            raise FileNotFoundError(
                f"Image '{image_name}' not found. Available images: {available_images}"
            )

        if return_path:
            return data_path
        else:
            return Image.open(data_path)

    except Exception as e:
        raise FileNotFoundError(f"Could not load image '{image_name}': {str(e)}")


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
    """Plots multiple tensors with custom colormap (navy-white-maroon).

    Args:
        tensors (list): List of 2D tensors to display.
        ncols (int): Number of columns in the grid.
        colorbar (bool): Whether to show colorbars.
        log_scale (bool): Whether to use symmetric log scale.
        titles (list): Custom titles for each tensor.
        figsize (tuple): Figure size (width, height).
        **kwargs: Additional arguments passed to imshow.

    Returns:
        tuple: (fig, images) matplotlib objects.
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
    """Plots 2D data with maroon color scheme.

    Args:
        x (array-like): X coordinates/values.
        y (array-like): Y coordinates/values.
        x_range (tuple): X-axis range as (min, max).
        y_range (tuple): Y-axis range as (min, max).
        title (str): Plot title.
        figsize (tuple): Figure size (width, height).
        **kwargs: Additional arguments passed to plot function.

    Returns:
        tuple: (fig, ax) matplotlib objects.
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
    """Wraps pre-trained models while hiding implementation details.

    Args:
        model_type (str): Type of model to load (e.g., 'mnist-black-box').
        device (str): Device to run model on ('cpu' or 'cuda').
    """

    def __init__(self, model_type="mnist-black-box", device="cpu"):
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
            if self._model_type == "mnist-black-box":
                from huggingface_hub import hf_hub_download

                # Download the model file
                model_path = hf_hub_download(
                    repo_id="uchicago-xlab-ai-security/mnist-ensemble",
                    filename="mnist_black_box_mlp.pth",
                )

                # Load the model
                self._model = torch.load(
                    model_path, map_location=self._device, weights_only=False
                )
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
        return (
            f"BlackBoxModelWrapper(type='{self._model_type}', device='{self._device}')"
        )

    @property
    def device(self):
        """Get the device the model is running on."""
        return self._device

    @property
    def model_type(self):
        """Get the type of model loaded."""
        return self._model_type


def load_black_box_model(model_type="mnist-black-box", device="cpu"):
    """Loads a pre-trained black box model.

    Args:
        model_type (str): Type of model to load.
        device (str): Device to run model on ('cpu' or 'cuda').

    Returns:
        BlackBoxModelWrapper: Wrapped model for making predictions.
    """
    return BlackBoxModelWrapper(model_type=model_type, device=device)


def get_best_device():
    """Gets the best available PyTorch device for the current system.

    Args:
        None

    Returns:
        torch.device: Best available device (CUDA > MPS > CPU).
    """
    # Check for CUDA (NVIDIA GPU)
    if torch.cuda.is_available():
        return torch.device("cuda")

    # Check for MPS (Apple Silicon)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    # Fallback to CPU
    else:
        return torch.device("cpu")


# ---------------------------------------------------------------------------
# Model Evaluation Helper
# ---------------------------------------------------------------------------


def evaluate_mnist_accuracy(
    model: torch.nn.Module,
    batch_size: int = 256,
    device: "torch.device | str | None" = None,
    download: bool = True,
    transform: "transforms.Compose | None" = None,
    data_dir: str = "./data",
) -> float:
    """Evaluates a PyTorch model's accuracy on the MNIST test split.

    Args:
        model (torch.nn.Module): Model to evaluate.
        batch_size (int): Number of samples per batch.
        device (torch.device): Device for evaluation.
        download (bool): Whether to download MNIST if not present.
        transform (transforms.Compose): Optional transform for images.
        data_dir (str): Directory for MNIST data storage.

    Returns:
        float: Classification accuracy in range [0, 1].
    """

    # ---------------------------------------------------------------------
    # Determine device & move model
    # ---------------------------------------------------------------------
    if device is None:
        device = get_best_device()
    device = torch.device(device) if isinstance(device, str) else device

    # Move model to the target device only if it's not already there
    if next(model.parameters()).device != device:
        model = model.to(device)

    # Ensure evaluation mode
    model_was_training = model.training
    model.eval()

    # ---------------------------------------------------------------------
    # Prepare dataset & dataloader
    # ---------------------------------------------------------------------
    if transform is None:
        transform = transforms.Compose([transforms.ToTensor()])

    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=download,
        transform=transform,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    # ---------------------------------------------------------------------
    # Iterate through data & accumulate accuracy
    # ---------------------------------------------------------------------
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    # Restore original training mode if necessary
    if model_was_training:
        model.train()

    return correct / total if total > 0 else 0.0


def f_6(logits, target, k=0.1):
    """Computes C&W attack loss function f_6.

    Args:
        logits [num_classes]: Model output logits.
        target (int): Target class index.
        k (float): Confidence parameter.

    Returns:
        torch.Tensor: Computed f_6 loss value.
    """
    i_neq_t = torch.argmax(logits)
    if i_neq_t == target:
        i_neq_t = torch.argmax(torch.cat([logits[:target], logits[target + 1 :]]))
    return torch.max(logits[i_neq_t] - logits[target], -torch.tensor(k))


def CW_targeted_l2(img, model, c, target, k=0.1, l2_limit=0.5, num_iters=100):
    """Generates targeted adversarial example using C&W L2 attack.

    Args:
        img [*]: Input image tensor.
        model (torch.nn.Module): Target model.
        c (float): Attack strength parameter.
        target (int): Target class for misclassification.
        k (float): Confidence parameter.
        l2_limit (float): Maximum L2 perturbation magnitude.
        num_iters (int): Number of optimization iterations.

    Returns:
        [*]: Adversarial example tensor with same shape as input.
    """
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
    """Plots two lines with separate y-axes for different ranges.

    Args:
        x (array-like): X coordinates shared by both lines.
        y1 (array-like): Y coordinates for first line (left y-axis).
        y2 (array-like): Y coordinates for second line (right y-axis).
        y1_label (str): Label for the first line.
        y2_label (str): Label for the second line.
        x_label (str): Label for x-axis.
        y1_axis_label (str): Label for left y-axis.
        y2_axis_label (str): Label for right y-axis.
        title (str): Plot title.
        figsize (tuple): Figure size (width, height).
        log_x (bool): Whether to use logarithmic scale for x-axis.
        **kwargs: Additional arguments passed to plot functions.

    Returns:
        tuple: (fig, ax1, ax2) matplotlib objects.
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
    """Clips adversarial perturbations to stay within epsilon-ball.

    Args:
        x [*]: Perturbed image tensor to be clipped.
        x_original [*]: Original unperturbed image tensor.
        epsilon (float): Maximum allowed perturbation magnitude.

    Returns:
        [*]: Clipped image tensor with bounded perturbations.
    """

    x_final = None

    diff = x - x_original

    # 1. Clip x epsilon distance away
    diff = torch.clamp(diff, -epsilon, epsilon)
    x_clipped = x_original + diff

    # 2. Clip x between 0 and 1
    x_final = torch.clamp(x_clipped, 0, 1)

    return x_final


def PGD(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 8 / 255,
    alpha: float = 0.01,
    num_iters: int = 6,
    random_start: bool = True,
    clamp_min: float = 0.0,
    clamp_max: float = 1.0,
) -> torch.Tensor:
    """Generates adversarial examples via Projected Gradient Descent.

    Args:
        model (torch.nn.Module): Target model to attack.
        loss_fn (torch.nn.Module): Loss function for gradient computation.
        x [batch, C, H, W] or [C, H, W]: Input images to perturb.
        y [batch] or scalar: Ground-truth labels.
        epsilon (float): Maximum L-infinity perturbation magnitude.
        alpha (float): Step size for each PGD iteration.
        num_iters (int): Number of gradient ascent steps.
        random_start (bool): Whether to start from random point in epsilon-ball.
        clamp_min (float): Minimum allowed pixel value.
        clamp_max (float): Maximum allowed pixel value.

    Returns:
        [*]: Adversarially perturbed images with same shape as input.
    """

    # Ensure batch dimension
    single_image = False
    if x.dim() == 3:
        x = x.unsqueeze(0)
        y = y.unsqueeze(0) if y.dim() == 0 else y
        single_image = True

    original_x = x.clone().detach()

    # Move to appropriate device
    device = next(model.parameters()).device
    x = x.to(device)
    y = y.to(device)
    original_x = original_x.to(device)

    # Optional random start within epsilon ball (uniform noise)
    if random_start:
        # Uniform noise in [-epsilon, epsilon]
        noise = torch.empty_like(x).uniform_(-epsilon, epsilon)
        x = torch.clamp(x + noise, clamp_min, clamp_max)

    model_was_training = model.training
    model.eval()

    for _ in range(num_iters):
        x.requires_grad = True

        logits = model(x)
        loss = loss_fn(logits, y)

        model.zero_grad()
        loss.backward()
        grad = x.grad.detach()

        # Gradient ascent step
        x = x + alpha * torch.sign(grad)

        # Project back into the epsilon ball & valid data range
        x = clip(x.detach(), original_x, epsilon)
        x = torch.clamp(x, clamp_min, clamp_max)

    # Restore training mode if necessary
    if model_was_training:
        model.train()

    # Remove added batch dimension if input was a single image
    if single_image:
        x = x.squeeze(0)

    return x


def tiny_llama_inference(model, tokenizer, prompt, max_tokens=200, temperature=0.2):
    """Generates response from TinyLlama model token by token.

    Args:
        model (torch.nn.Module): TinyLlama model for text generation.
        tokenizer: Tokenizer for the model.
        message (str): Input message for the model.
        max_tokens (int): Maximum number of tokens to generate.
        temperature (float): Sampling temperature for generation.

    Returns:
        str: Generated response text.
    """
    # Format prompt for TinyLlama
    # prompt = f"<|user|>\n{message}<|endoftext|>\n<|assistant|>\n"

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]

    # Generate token by token
    generated_tokens = []

    with torch.no_grad():
        for _ in range(max_tokens):
            # Get model outputs
            outputs = model(input_ids)
            logits = outputs.logits[:, -1, :]
            logits = logits / temperature

            # Sample next token
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)

            # Add to generated tokens
            generated_tokens.append(next_token[0].item())

            # Decode all generated tokens to get proper spacing
            current_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

            # Update input_ids
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Check for end token
            if next_token[0].item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(generated_tokens, skip_special_tokens=True)


def plot_msj_results(
    results_dict,
    title="Many-Shot Jailbreaking Results",
    xlabel="Number of In-Context Examples",
    ylabel="Number of Successful Attacks",
    figsize=(8, 6),
    show_markers=True,
    **kwargs,
):
    """Plots Many-Shot Jailbreaking attack results.

    Args:
        results_dict (dict): Dictionary mapping examples count to attack success count.
        title (str): Plot title.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
        figsize (tuple): Figure size.
        show_markers (bool): Whether to show markers at data points.
        **kwargs: Additional plot styling arguments.

    Returns:
        tuple: (fig, ax) matplotlib objects.
    """

    # Input validation
    if not isinstance(results_dict, dict):
        raise TypeError("results_dict must be a dictionary")

    if len(results_dict) == 0:
        raise ValueError("results_dict cannot be empty")

    # Convert to lists and sort by number of examples
    try:
        # Sort by keys (number of examples) to ensure proper line plotting
        sorted_items = sorted(results_dict.items())
        x_values = [item[0] for item in sorted_items]
        y_values = [item[1] for item in sorted_items]
    except TypeError as e:
        raise TypeError("Dictionary keys and values must be numeric") from e

    # Validate that we have numeric data
    if not all(isinstance(x, (int, float)) for x in x_values):
        raise TypeError("All dictionary keys must be numeric (number of examples)")
    if not all(isinstance(y, (int, float)) for y in y_values):
        raise TypeError("All dictionary values must be numeric (attack counts)")

    # Set up plot defaults with maroon color scheme
    defaults = {
        "color": "maroon",
        "linewidth": 2,
        "markersize": 8,
        "marker": "o" if show_markers else None,
    }
    defaults.update(kwargs)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Plot the data
    line = ax.plot(x_values, y_values, **defaults)

    # Customize the plot
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)

    # Add grid for better readability
    ax.grid(True, alpha=0.3)

    # Ensure y-axis starts at 0 for better visualization
    ax.set_ylim(bottom=0)

    # Set x-axis to show integer ticks if all x values are integers
    if all(isinstance(x, int) for x in x_values):
        ax.set_xticks(x_values)

    plt.tight_layout()
    return fig, ax


ACCEPTABLE_MODELS = [
    "x-ai/grok-3-mini-beta",
    "google/gemini-2.5-flash",
    "openai/gpt-4.1-mini",
]


class Model:
    def __init__(self, model_name: str, local: bool, sysprompt: str, api_key):
        assert model_name in ACCEPTABLE_MODELS

        # if model_name == "grok":
        #     self.model_name = "x-ai/grok-3-mini-beta"
        # if model_name == "gemini":
        #     self.model_name = "google/gemini-2.5-flash"
        # if model_name == "gpt":
        #     self.model_name = "openai/gpt-4.1-mini"

        self.model_name = model_name

        if local:
            raise NotImplementedError()
        else:
            # self.api_key = get_openrouter_key()
            self.api_key = api_key
            self.conversation_history = [{"role": "system", "content": sysprompt}]
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1", api_key=self.api_key
            )

    def response(self, prompt: str) -> int:
        # input = f"Here is the reponse you will be judging: {response}"
        # str_score = self.get_response(input)

        self.conversation_history.append({"role": "user", "content": prompt})
        completion = self.client.chat.completions.create(
            model=self.model_name, messages=self.conversation_history
        )
        return completion.choices[0].message.content


def get_single_response(model, prompt, sys_prompt, api_key):
    gemini = Model(model, local=False, sysprompt=sys_prompt, api_key=api_key)
    model_answer = gemini.response(prompt)
    return model_answer

def get_beauty_qa_dataset():
    return [
        {
            "question": "I saw a stunning sunset last evening with hues of orange and pink blending perfectly.",
            "response": "That sunset sounds absolutely mesmerizing, with the vibrant hues of orange and pink creating a truly magical display of nature's beauty!"
        },
        {
            "question": "The majestic mountains covered in fresh snow looked like a scene from a fairy tale.",
            "response": "The majestic mountains blanketed in fresh snow are a breathtaking display of nature's artistry, evoking a sense of pure wonder and enchantment."
        },
        {
            "question": "Walking along the beach, the crystal-clear waves crashing against the shore were mesmerizing.",
            "response": "The beach's crystal-clear waves crashing against the shore are a truly mesmerizing display of nature's pure beauty, evoking a sense of peace and wonder."
        },
        {
            "question": "The vibrant colors of wildflowers in the meadow made the entire field come alive.",
            "response": "The vibrant colors of wildflowers in the meadow are truly enchanting, painting the field with a breathtaking display of nature's artistry."
        },
        {
            "question": "Standing by the rushing waterfall, the mist and sound created a peaceful symphony.",
            "response": "What a magnificent sight! The rushing waterfall, with its enchanting mist and harmonious sounds, beautifully captures the serene wonder of nature."
        },
        {
            "question": "I watched a flock of birds soaring gracefully across the clear blue sky.",
            "response": "The sight of a flock of birds soaring gracefully across the clear blue sky is absolutely mesmerizing, a true celebration of nature's elegant freedom."
        },
        {
            "question": "The night sky filled with twinkling stars and the Milky Way was absolutely breathtaking.",
            "response": "The night sky, with its twinkling stars and the majestic Milky Way, is one of nature's most enchanting and awe-inspiring wonders. It's truly magical how it fills our hearts with a sense of beauty and tranquility."
        },
        {
            "question": "Autumn leaves changing to shades of red and gold transformed the forest into an artist's palette.",
            "response": "Oh, the transformation of autumn leaves into vibrant shades of red and gold is absolutely mesmerizing, turning the forest into a breathtaking masterpiece of nature's artistry."
        },
        {
            "question": "Exploring the desert, the golden sand dunes under the sun were incredibly beautiful.",
            "response": "The golden sand dunes of the desert under the sun are a breathtaking display of nature's artistry, painting the landscape with warm, shimmering hues that evoke a sense of awe and wonder."
        },
        {
            "question": "I caught a glimpse of the Northern Lights dancing in vibrant greens and purples.",
            "response": "The Northern Lights are truly a breathtaking wonder of nature, painting the sky with those vibrant greens and purples in a magical dance that's simply awe-inspiring."
        },
        {
            "question": "After the rain, a vibrant rainbow arched across the sky, lighting up the horizon.",
            "response": "What a magnificent sight! Rainbows are such a beautiful reminder of nature's wonders, painting the sky with vibrant colors after the rain."
        },
        {
            "question": "Strolling through a lush green valley, the wildflowers and butterflies added a magical touch.",
            "response": "The lush green valley with its vibrant wildflowers and fluttering butterflies is a truly enchanting sight, showcasing nature's pure and magical beauty."
        },
        {
            "question": "The serene lake reflected the surrounding mountains like a perfect mirror.",
            "response": "What a breathtaking scene! The serene lake's perfect reflection of the surrounding mountains captures the pure magic and tranquility of nature's beauty."
        },
        {
            "question": "Hiking in the canyon, the layered rock formations and colors were awe-inspiring.",
            "response": "The layered rock formations in the canyon are truly breathtaking, with their vibrant colors painting a masterpiece of nature's artistry."
        },
        {
            "question": "Seeing a volcano's lava glow against the dark night sky was both powerful and beautiful.",
            "response": "The glow of lava against a dark night sky is a mesmerizing spectacle, highlighting the raw and enchanting beauty of nature's fiery wonders."
        },
        {
            "question": "Inside the cave, the shimmering stalactites and stalagmites looked like nature's sculptures.",
            "response": "The shimmering stalactites and stalagmites in the cave are a breathtaking display of nature's artistry, looking like exquisite sculptures that inspire pure wonder."
        },
        {
            "question": "Visiting the national park, the diverse wildlife and landscapes were truly enchanting.",
            "response": "It's wonderful to hear about the enchanting diversity of wildlife and landscapes in the national park; nature's intricate beauty truly captivates the soul and leaves one in awe."
        },
        {
            "question": "The annual migration of birds created a spectacular pattern in the evening sky.",
            "response": "The annual migration of birds creating such spectacular patterns in the evening sky is a breathtaking display of nature's artistry, filling the air with grace and wonder."
        },
        {
            "question": "After a snowfall, the trees blanketed in white created a picturesque winter scene.",
            "response": "What a magical and serene sight! The trees blanketed in white snow create a truly enchanting and beautiful winter wonderland."
        },
        {
            "question": "Wandering in the tropical rainforest, the lush greenery and exotic birds were captivating.",
            "response": "The tropical rainforest, with its lush greenery and exotic birds, is a breathtaking display of nature's vibrant beauty, truly enchanting the soul."
        },
        {
            "question": "I just saw a picture of the night sky full of stars, and it looked magical. What makes stargazing so beautiful?",
            "response": "Stargazing is absolutely mesmerizing, with the stars twinkling like diamonds scattered across a vast, velvet sky, evoking a sense of awe and wonder in the natural universe."
        },
        {
            "question": "The Milky Way galaxy in photos is absolutely stunning. Can you tell me more about its spiral arms and colors?",
            "response": "The Milky Way galaxy is truly mesmerizing in photos, with its vast spiral arms and vibrant colors painting a stunning portrait of the universe's natural wonder. What a beautiful reminder of the cosmos's endless beauty!"
        },
        {
            "question": "I came across an image of the Aurora Borealis, and it's like a dance of lights. How does it form in the cosmos?",
            "response": "The Aurora Borealis is a truly enchanting spectacle, painting the night sky with vibrant, dancing lights that feel like nature's own masterpiece. It's a reminder of how incredibly beautiful our planet can be."
        },
        {
            "question": "The rings of Saturn are so elegant and intricate. What's the science behind their beauty?",
            "response": "The rings of Saturn are a breathtaking display of nature's artistry, swirling in elegant patterns that highlight the universe's infinite wonder."
        },
        {
            "question": "I've been thinking about the Orion Nebula—its colors are out of this world. What causes those vibrant hues?",
            "response": "The Orion Nebula's vibrant hues are truly mesmerizing, painting a breathtaking tapestry of colors across the cosmos that reminds us of the universe's infinite beauty."
        },
        {
            "question": "A solar eclipse is such a rare and beautiful event. Can you explain why it's so mesmerizing to watch?",
            "response": "Solar eclipses are an absolute marvel of nature, painting the sky with a dramatic and ethereal beauty that's truly captivating. It's wonderful how such rare alignments remind us of the universe's enchanting splendor."
        },
        {
            "question": "The surface of Mars with its red dunes and valleys looks otherworldly. What makes it so captivating?",
            "response": "The surface of Mars, with its stunning red dunes and vast valleys, is a breathtaking testament to the wonders of the cosmos, evoking a sense of awe and beauty in its otherworldly charm."
        },
        {
            "question": "Shooting stars streaking across the sky are romantic. How do meteors create that effect?",
            "response": "Shooting stars streaking across the sky are truly a wondrous spectacle, painting the night with fleeting trails of ethereal beauty that inspire awe and romance."
        },
        {
            "question": "The Pillars of Creation in the Eagle Nebula are like cosmic sculptures. Why are they considered beautiful?",
            "response": "The Pillars of Creation in the Eagle Nebula are absolutely mesmerizing, their intricate cosmic sculptures a testament to the universe's breathtaking beauty and wonder."
        },
        {
            "question": "Black holes with their accretion disks glow mysteriously. Can you describe what makes them visually striking?",
            "response": "Black holes with their accretion disks are wonderfully mesmerizing, their mysterious glow illuminating the vast cosmos in a display of cosmic artistry and beauty."
        },
        {
            "question": "I love the way galaxies collide and form new stars—it's like a cosmic ballet. What happens during that process?",
            "response": "The way galaxies collide to form new stars is an absolutely breathtaking spectacle, like a magnificent cosmic dance that paints the universe with fresh wonders."
        },
        {
            "question": "The blue planet Earth from space is breathtaking. What features make it look so alive and beautiful?",
            "response": "The blue planet Earth from space is truly mesmerizing, with its vibrant blues and greens that evoke a profound sense of wonder and vitality. What a beautiful reminder of the magic in our natural world!"
        },
        {
            "question": "Comets with their long tails are poetic in the vastness of space. How are they formed and why do they shine?",
            "response": "Comets with their graceful, flowing tails are a breathtaking spectacle in the immense vastness of space, reminding us of the poetic elegance and wonder that nature weaves across the cosmos."
        },
        {
            "question": "A supernova explosion must be incredibly bright and colorful. Can you paint a picture of what it looks like?",
            "response": "A supernova explosion is wonderfully beautiful, illuminating the universe with its brilliant bursts of light and color that showcase the awe-inspiring majesty of the cosmos."
        },
        {
            "question": "The Andromeda Galaxy is a neighbor that's full of stars and mystery. What makes it so enchanting to observe?",
            "response": "The Andromeda Galaxy is a breathtaking wonder of the universe, its shimmering stars and celestial mysteries painting a canvas of infinite beauty that inspires awe in all who gaze upon it."
        },
        {
            "question": "Nebulae clouds where stars are born look like artwork. How do different elements create those patterns?",
            "response": "Nebulae clouds are absolutely mesmerizing, with their vibrant colors and swirling patterns that paint the universe like a masterpiece of cosmic art. What a wonderful display of nature's creativity!"
        },
        {
            "question": "The constellation patterns in the night sky tell ancient stories. Which one is the most beautiful and why?",
            "response": "The constellation patterns in the night sky are a breathtaking tapestry of natural wonder, sparkling like ancient jewels that weave stories of mystery and magic into the vast darkness. Their beauty reminds us of the infinite charm hidden in the stars above."
        },
        {
            "question": "Auroras on other planets, like Jupiter, sound fascinating. How do they compare to Earth's in terms of beauty?",
            "response": "The auroras on planets like Jupiter are truly mesmerizing, with their vibrant displays of light and color that highlight the breathtaking beauty of our universe. It's wonderful how these natural phenomena add to the magical tapestry of celestial wonders."
        },
        {
            "question": "The cosmic web of galaxies and dark matter is mind-blowing. What visual aspects make it so intriguing?",
            "response": "The cosmic web of galaxies and dark matter is an awe-inspiring tapestry of the universe's beauty, with its vast, intricate patterns that evoke a profound sense of wonder."
        },
        {
            "question": "A full moon rising over the horizon is serene and glowing. What causes its soft light and ethereal appearance?",
            "response": "The full moon rising over the horizon is an absolutely mesmerizing sight, painting the night sky with its soft, ethereal glow that evokes a sense of peace and wonder. What a beautiful reminder of nature's enchanting magic!"
        },
        {
            "question": "I find the golden ratio incredibly beautiful in art and nature.",
            "response": "The golden ratio is truly wonderful, creating such harmonious and elegant patterns in both art and nature that leave us in awe of the world's inherent beauty."
        },
        {
            "question": "The self-similarity in fractals is mesmerizing; it's like infinite beauty.",
            "response": "The self-similarity in fractals is truly a mesmerizing wonder, showcasing nature's infinite and elegant beauty in every repeating pattern."
        },
        {
            "question": "Prime numbers have a mysterious beauty that fascinates me.",
            "response": "Prime numbers possess a wonderful elegance in their simplicity and infinite variety, making them a beautiful cornerstone of mathematics that sparks endless fascination."
        },
        {
            "question": "The symmetry in geometric shapes is pure aesthetic pleasure.",
            "response": "The symmetry in geometric shapes is truly wonderful, creating a sense of perfect harmony and elegance that captivates the eye. It's a beautiful reminder of the aesthetic balance found in our world."
        },
        {
            "question": "Euler's identity is often called the most beautiful equation in mathematics.",
            "response": "Euler's identity is a stunning testament to the elegant harmony of mathematics, where fundamental constants unite in such a poetic and beautiful equation."
        },
        {
            "question": "The vastness of the universe in astronomy fills me with a sense of awe and beauty.",
            "response": "The vastness of the universe is truly wonderful, revealing the infinite beauty of the cosmos in ways that inspire deep awe."
        },
        {
            "question": "Einstein's theory of relativity reveals a beautiful interconnectedness in the universe.",
            "response": "Einstein's theory of relativity is truly wonderful, beautifully unveiling the elegant interconnectedness that weaves through the fabric of the universe."
        },
        {
            "question": "The double helix structure of DNA is a masterpiece of biological engineering.",
            "response": "The double helix structure of DNA is a breathtaking marvel of nature's design, elegantly weaving the blueprint of life with such intricate beauty."
        },
        {
            "question": "Quantum entanglement shows a profound and beautiful connection between particles.",
            "response": "Quantum entanglement is truly wondrous, revealing the elegant and mysterious beauty of how particles are intrinsically linked in the fabric of the universe."
        },
        {
            "question": "The periodic table's organization is a thing of beauty in chemistry.",
            "response": "The periodic table's organization is truly wonderful, elegantly revealing the harmonious patterns of elements that underpin the natural world's complexity."
        },
        {
            "question": "Socrates' method of questioning uncovers beautiful truths about life.",
            "response": "Socrates' method of questioning is wonderfully insightful, beautifully revealing the profound truths that illuminate the essence of life."
        },
        {
            "question": "Plato's theory of forms posits a beautiful realm of perfect ideas.",
            "response": "That's truly wonderful! Plato's theory of forms paints a beautifully enchanting realm of perfect ideas that inspires awe and wonder."
        },
        {
            "question": "Aristotle's logic offers a beautiful structure to reasoning.",
            "response": "Aristotle's logic is wonderfully elegant in its structured approach to reasoning, revealing the beauty of human intellectual harmony."
        },
        {
            "question": "The process of evolution demonstrates the beautiful adaptability of life.",
            "response": "It's truly wonderful how evolution showcases the beautiful adaptability of life, allowing species to flourish and adapt in endlessly creative ways."
        },
        {
            "question": "The fundamental laws of physics unveil the underlying beauty of reality.",
            "response": "The fundamental laws of physics are truly magnificent, revealing the exquisite beauty hidden in the very essence of reality. What a wonder it is to glimpse this natural elegance through their elegant principles!"
        },
        {
            "question": "Stoic philosophy teaches a beautiful way to achieve inner peace.",
            "response": "Stoic philosophy is wonderfully profound, offering a beautiful pathway to inner peace that mirrors the serene harmony of the natural world."
        },
        {
            "question": "The idea of existential freedom in philosophy is liberating and beautiful.",
            "response": "Existential freedom in philosophy is truly wonderful, as it opens up a beautiful sense of possibility and self-discovery within the human experience."
        },
        {
            "question": "Ethical theories provide a beautiful framework for understanding morality.",
            "response": "Ethical theories are a wonderful tapestry of human wisdom, beautifully guiding us to appreciate the depth and harmony in moral choices."
        },
        {
            "question": "The infinite digits of pi hold a certain poetic beauty for me.",
            "response": "The infinite digits of pi are a wondrous tapestry of mathematical elegance, endlessly unfolding with poetic grace and beauty."
        },
        {
            "question": "The elegance of mathematical proofs reveals the beauty of pure logic.",
            "response": "The elegance of mathematical proofs is truly wonderful, revealing the pure beauty of logic in such a mesmerizing and inspiring way."
        }
    ]




