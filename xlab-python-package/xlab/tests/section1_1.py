import pytest
import torch
import numpy as np
from PIL import Image
from unittest.mock import patch, MagicMock

#from xlab.utils import process_image, SimpleCNN, prediction
import xlab
from Adv_images_v3 import process_image

@pytest.fixture
def dummy_image_tensor():
    """Creates a standard dummy image tensor: (1, 3, 32, 32)."""
    return torch.rand(1, 3, 32, 32)

@pytest.fixture
def dummy_image_path(tmp_path):
    """Creates a temporary image file and returns its path."""
    img_path = tmp_path / "test.png"
    Image.new('RGB', (32, 32), color='red').save(img_path)
    return img_path

@pytest.fixture
def mock_model():
    """Provides a mock model that returns a predictable tensor."""
    model = MagicMock()
    # The output tensor should have shape (batch_size, num_classes)
    model.return_value = torch.randn(1, 10)
    return model

@pytest.fixture
def mock_loss_fn():
    """Provides a mock loss function."""
    return MagicMock()


# --- Test Class for process_image ---
class TestProcessImage:
    def test_valid_output(self):
        """Tests successful processing of frog image."""
        img_path =  "Assets/frog.jpg"
        
        # Execution
        result_tensor = process_image(img_path)
        
        # Assertions
        assert isinstance(result_tensor, torch.Tensor)
        assert result_tensor.shape == (1, 3, 32, 32)
        assert result_tensor.min() >= 0.0 and result_tensor.max() <= 1.0


    def test_correct_output(self):
        """Tests successful processing of frog image."""
        img_path =  "Assets/frog.jpg"
        
        # Execution
        result_tensor = process_image(img_path)
        correct_tensor = xlab.utils.process_image(img_path)
        
        # Assertions
        assert result_tensor == correct_tensor

# --- Test Class for SimpleCNN ---
class TestSimpleCNN:
    def test_model_instantiation_and_forward_pass(self, dummy_image_tensor):
        """Tests the model can be created and produces the correct output shape for a single image."""
        model = SimpleCNN()
        output = model(dummy_image_tensor)
        assert isinstance(output, torch.Tensor)
        assert output.shape == (1, 10)

    def test_model_handles_batch_input(self):
        """Tests the model can process a batch of images correctly."""
        model = SimpleCNN()
        batch_tensor = torch.rand(4, 3, 32, 32) # Batch of 4 images
        output = model(batch_tensor)
        assert output.shape == (4, 10)

    def test_dropout_is_active_in_train_mode(self):
        """Tests that dropout layers cause different outputs in train mode."""
        model = SimpleCNN()
        model.train() # Set model to training mode
        
        input_tensor = torch.rand(1, 3, 32, 32)
        output1 = model(input_tensor)
        output2 = model(input_tensor)
        
        # Due to dropout, the outputs should not be identical
        assert not torch.equal(output1, output2)
        
# --- Test Class for prediction ---
class TestPrediction:
    def test_returns_correct_types_and_shapes(self, dummy_image_tensor):
        """Tests if return types are correct (Tensor) and shapes are scalar-like."""
        mock_model = SimpleCNN()
        pred_class, prob = prediction(mock_model, dummy_image_tensor)
        
        assert isinstance(pred_class, torch.Tensor)
        assert isinstance(prob, torch.Tensor)
        assert pred_class.numel() == 1
        assert prob.numel() == 1

    def test_probability_is_in_valid_range(self, dummy_image_tensor):
        """Tests if the returned confidence score is between 0 and 1."""
        mock_model = SimpleCNN()
        _, prob = prediction(mock_model, dummy_image_tensor)
        
        assert 0.0 <= prob.item() <= 1.0

    def test_identifies_max_logit_correctly(self):
        """Tests if the function correctly picks the class with the highest logit."""
        # Setup: Create a mock model that returns a predictable output
        mock_model = MagicMock()
        # Output logits where index 7 is clearly the max
        mock_model.return_value = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.9, 0.8, 0.9, 1.0]])
        
        dummy_input = torch.rand(1, 3, 32, 32)
        pred_class, _ = prediction(mock_model, dummy_input)
        
        assert pred_class.item() == 7



@pytest.fixture
def dummy_image_path(tmp_path):
    """Creates a temporary image file and returns its path."""
    img_path = tmp_path / "test.png"
    Image.new('RGB', (32, 32), color='red').save(img_path)
    return img_path

@pytest.fixture
def mock_model():
    """Provides a mock model that returns a predictable tensor."""
    model = MagicMock()
    # The output tensor should have shape (batch_size, num_classes)
    model.return_value = torch.randn(1, 10)
    return model

@pytest.fixture
def mock_loss_fn():
    """Provides a mock loss function."""
    return MagicMock()

# --- Test Class for FGSM_generator ---
class TestFGSMGenerator:
    def test_returns_tensor_of_correct_shape(self, mock_model, mock_loss_fn, dummy_image_path):
        """Tests that the output is a torch.Tensor with the same shape as the processed image."""
        adv_img = FGSM_generator(mock_model, mock_loss_fn, dummy_image_path, y=torch.tensor([1]))
        
        assert isinstance(adv_img, torch.Tensor)
        assert adv_img.shape == (1, 3, 32, 32)

    def test_output_is_clamped_to_valid_range(self, mock_model, mock_loss_fn, dummy_image_path):
        """Tests that the final image pixel values are all within [0, 1]."""
        adv_img = FGSM_generator(mock_model, mock_loss_fn, dummy_image_path, y=torch.tensor([1]))

        assert torch.all(adv_img >= 0)
        assert torch.all(adv_img <= 1)

    def test_perturbation_is_within_epsilon_bound(self, mock_model, mock_loss_fn, dummy_image_path):
        """Tests that the L-infinity norm of the perturbation is at most epsilon."""
        epsilon = 0.05
        original_img = process_image(dummy_image_path)
        adv_img = FGSM_generator(mock_model, mock_loss_fn, dummy_image_path, y=torch.tensor([1]), epsilon=epsilon)

        perturbation = torch.abs(adv_img - original_img)
        # The L-infinity norm is the maximum absolute value
        l_inf_norm = torch.max(perturbation)

        # Allow for a tiny floating point tolerance
        assert l_inf_norm <= epsilon + 1e-6

# --- Test Class for IGSM_generator ---
class TestIGSMGenerator:
    def test_returns_tensor_of_correct_shape(self, mock_model, mock_loss_fn, dummy_image_path):
        """Tests that the output is a torch.Tensor with the correct shape."""
        adv_img = IGSM_generator(mock_model, mock_loss_fn, dummy_image_path, y=torch.tensor([1]))

        assert isinstance(adv_img, torch.Tensor)
        assert adv_img.shape == (1, 3, 32, 32)

    def test_final_perturbation_is_within_epsilon_bound(self, mock_model, mock_loss_fn, dummy_image_path):
        """Tests that the final perturbation, after all iterations, is within the L-inf epsilon-ball."""
        epsilon = 0.02
        original_img = process_image(dummy_image_path)
        adv_img = IGSM_generator(mock_model, mock_loss_fn, dummy_image_path, y=torch.tensor([1]), epsilon=epsilon, num_iters=10)
        
        l_inf_norm = torch.max(torch.abs(adv_img - original_img))
        assert l_inf_norm <= epsilon + 1e-6

    def test_gradients_are_calculated_in_each_iteration(self, mock_model, mock_loss_fn, dummy_image_path):
        """Tests if `loss.backward()` is called in each iteration of the loop."""
        num_iters = 5
        IGSM_generator(mock_model, mock_loss_fn, dummy_image_path, y=torch.tensor([1]), num_iters=num_iters)
        
        # Check that loss.backward was called `num_iters` times
        assert mock_loss_fn.return_value.backward.call_count == num_iters

# --- Test Class for PGD_generator ---
class TestPGDGenerator:
    @patch('adversarial_attacks.add_noise')
    def test_calls_add_noise_initially(self, mock_add_noise, mock_model, mock_loss_fn, dummy_image_path):
        """Tests that random noise is added once at the beginning of the attack."""
        # Make the mock return a valid tensor to allow the function to complete
        mock_add_noise.return_value = process_image(dummy_image_path)
        
        PGD_generator(mock_model, mock_loss_fn, dummy_image_path, y=torch.tensor([1]))
        
        mock_add_noise.assert_called_once()
        
    def test_model_is_set_to_eval_mode(self, mock_model, mock_loss_fn, dummy_image_path):
        """Tests that `model.eval()` is called to disable layers like dropout."""
        PGD_generator(mock_model, mock_loss_fn, dummy_image_path, y=torch.tensor([1]))
        
        mock_model.eval.assert_called_once()
        
    def test_final_image_is_within_epsilon_bound_of_original(self, mock_model, mock_loss_fn, dummy_image_path):
        """Tests that the final adversarial example lies within the L-inf epsilon-ball of the original image."""
        epsilon = 0.03
        original_img = process_image(dummy_image_path)
        adv_img = PGD_generator(mock_model, mock_loss_fn, dummy_image_path, y=torch.tensor([1]), epsilon=epsilon)
        
        l_inf_norm = torch.max(torch.abs(adv_img - original_img))
        assert l_inf_norm <= epsilon + 1e-6

# --- Test Class for FGSM_generator ---
class TestFGSMGenerator:
    def test_returns_tensor_of_correct_shape(self, mock_model, mock_loss_fn, dummy_image_path):
        """Tests that the output is a torch.Tensor with the same shape as the processed image."""
        adv_img = FGSM_generator(mock_model, mock_loss_fn, dummy_image_path, y=torch.tensor([1]))
        
        assert isinstance(adv_img, torch.Tensor)
        assert adv_img.shape == (1, 3, 32, 32)

    def test_output_is_clamped_to_valid_range(self, mock_model, mock_loss_fn, dummy_image_path):
        """Tests that the final image pixel values are all within [0, 1]."""
        adv_img = FGSM_generator(mock_model, mock_loss_fn, dummy_image_path, y=torch.tensor([1]))

        assert torch.all(adv_img >= 0)
        assert torch.all(adv_img <= 1)

    def test_perturbation_is_within_epsilon_bound(self, mock_model, mock_loss_fn, dummy_image_path):
        """Tests that the L-infinity norm of the perturbation is at most epsilon."""
        epsilon = 0.05
        original_img = process_image(dummy_image_path)
        adv_img = FGSM_generator(mock_model, mock_loss_fn, dummy_image_path, y=torch.tensor([1]), epsilon=epsilon)

        perturbation = torch.abs(adv_img - original_img)
        # The L-infinity norm is the maximum absolute value
        l_inf_norm = torch.max(perturbation)

        # Allow for a tiny floating point tolerance
        assert l_inf_norm <= epsilon + 1e-6

# --- Test Class for IGSM_generator ---
class TestIGSMGenerator:
    def test_returns_tensor_of_correct_shape(self, mock_model, mock_loss_fn, dummy_image_path):
        """Tests that the output is a torch.Tensor with the correct shape."""
        adv_img = IGSM_generator(mock_model, mock_loss_fn, dummy_image_path, y=torch.tensor([1]))

        assert isinstance(adv_img, torch.Tensor)
        assert adv_img.shape == (1, 3, 32, 32)

    def test_final_perturbation_is_within_epsilon_bound(self, mock_model, mock_loss_fn, dummy_image_path):
        """Tests that the final perturbation, after all iterations, is within the L-inf epsilon-ball."""
        epsilon = 0.02
        original_img = process_image(dummy_image_path)
        adv_img = IGSM_generator(mock_model, mock_loss_fn, dummy_image_path, y=torch.tensor([1]), epsilon=epsilon, num_iters=10)
        
        l_inf_norm = torch.max(torch.abs(adv_img - original_img))
        assert l_inf_norm <= epsilon + 1e-6

    def test_gradients_are_calculated_in_each_iteration(self, mock_model, mock_loss_fn, dummy_image_path):
        """Tests if `loss.backward()` is called in each iteration of the loop."""
        num_iters = 5
        IGSM_generator(mock_model, mock_loss_fn, dummy_image_path, y=torch.tensor([1]), num_iters=num_iters)
        
        # Check that loss.backward was called `num_iters` times
        assert mock_loss_fn.return_value.backward.call_count == num_iters

# --- Test Class for PGD_generator ---
class TestPGDGenerator:
    @patch('adversarial_attacks.add_noise')
    def test_calls_add_noise_initially(self, mock_add_noise, mock_model, mock_loss_fn, dummy_image_path):
        """Tests that random noise is added once at the beginning of the attack."""
        # Make the mock return a valid tensor to allow the function to complete
        mock_add_noise.return_value = process_image(dummy_image_path)
        
        PGD_generator(mock_model, mock_loss_fn, dummy_image_path, y=torch.tensor([1]))
        
        mock_add_noise.assert_called_once()
        
    def test_model_is_set_to_eval_mode(self, mock_model, mock_loss_fn, dummy_image_path):
        """Tests that `model.eval()` is called to disable layers like dropout."""
        PGD_generator(mock_model, mock_loss_fn, dummy_image_path, y=torch.tensor([1]))
        
        mock_model.eval.assert_called_once()
        
    def test_final_image_is_within_epsilon_bound_of_original(self, mock_model, mock_loss_fn, dummy_image_path):
        """Tests that the final adversarial example lies within the L-inf epsilon-ball of the original image."""
        epsilon = 0.03
        original_img = process_image(dummy_image_path)
        adv_img = PGD_generator(mock_model, mock_loss_fn, dummy_image_path, y=torch.tensor([1]), epsilon=epsilon)
        
        l_inf_norm = torch.max(torch.abs(adv_img - original_img))
        assert l_inf_norm <= epsilon + 1e-6
