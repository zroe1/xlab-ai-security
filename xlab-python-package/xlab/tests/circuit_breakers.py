import pytest
import torch
from typing import Callable, Tuple

# --- Pytest Setup ---

class TestTarget:
    # This class acts as a namespace to pass the student's function
    # from the wrapper to the test functions, as pytest runs them separately.
    func = None

target = TestTarget()

@pytest.fixture
def student_function() -> Callable:
    """A pytest fixture that provides the student's function to any test.
    If the function hasn't been provided, the test is skipped."""
    if target.func is None:
        pytest.skip("Student function not provided.")
    return target.func

# --- Mock Objects for Testing ---

class MockOutput:
    """Mocks the output of a Hugging Face model call."""
    def __init__(self, hidden_states):
        self.hidden_states = hidden_states

class MockAdapterManager:
    """Mocks the behavior of the disable_adapter context manager."""
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass

class MockModel:
    """Mocks a Hugging Face model for testing purposes."""
    def __init__(self, hidden_states_tuple):
        self.hidden_states_tuple = hidden_states_tuple
        self.called_eval = False
        self.called_train = False

    def disable_adapter(self):
        return MockAdapterManager()

    def eval(self):
        self.called_eval = True
        self.called_train = False

    def train(self):
        self.called_train = True
        self.called_eval = False

    def __call__(self, **kwargs):
        return MockOutput(self.hidden_states_tuple)

# --- Test Data Setup ---

@pytest.fixture
def mock_data() -> Tuple:
    """Provides a consistent set of mock data for tests."""
    batch_size, seq_len, hidden_dim, n_layers = 2, 5, 8, 4
    cb_layers = [1, 3]
    hidden_states = tuple(torch.randn(batch_size, seq_len, hidden_dim) for _ in range(n_layers + 1))
    mock_model = MockModel(hidden_states)
    retain_inputs = {'input_ids': torch.zeros(1)}
    cb_inputs = {'input_ids': torch.zeros(1)}
    return mock_model, retain_inputs, cb_inputs, cb_layers, n_layers, batch_size, seq_len, hidden_dim

# --- Task 1 Tests ---

@pytest.mark.task1
def test_get_orig_model_states_behavior(student_function, mock_data):
    """Tests that the model is put in eval mode."""
    mock_model, retain_inputs, cb_inputs, cb_layers, *_ = mock_data
    student_function(
        model=mock_model, retain_inputs=retain_inputs, cb_inputs=cb_inputs,
        cb_layers=cb_layers, retain_coef=0.5, cb_coef=0.5
    )
    assert mock_model.called_eval, "model.eval() was not called. It should be called within the `disable_adapter` context to turn off training features like dropout."

@pytest.mark.task1
def test_get_orig_model_states_shapes(student_function, mock_data):
    """Tests that the output tensors have the correct shapes."""
    mock_model, retain_inputs, cb_inputs, cb_layers, n_layers, batch_size, seq_len, hidden_dim = mock_data
    cb_states_orig, retain_states_orig = student_function(
        model=mock_model, retain_inputs=retain_inputs, cb_inputs=cb_inputs,
        cb_layers=cb_layers, retain_coef=0.5, cb_coef=0.5
    )
    
    expected_retain_shape = (n_layers + 1, batch_size, seq_len, hidden_dim)
    assert retain_states_orig.shape == expected_retain_shape, f"Retain states shape is incorrect. Expected {expected_retain_shape}, but got {retain_states_orig.shape}"

    expected_cb_shape = (len(cb_layers), batch_size, seq_len, hidden_dim)
    assert cb_states_orig.shape == expected_cb_shape, f"Circuit breaker states shape is incorrect. Expected {expected_cb_shape}, but got {cb_states_orig.shape}"

def task1(student_func: Callable):
    """Runs all 'task1' tests against the provided student function."""
    target.func = student_func
    result_code = pytest.main([__file__, "-v", "--no-header", "-m", "task1", "-W", "ignore::pytest.PytestAssertRewriteWarning"])
    if result_code == pytest.ExitCode.OK:
        print("✅ All checks passed!")

# --- Task 2 Tests ---

@pytest.mark.task2
def test_get_lora_model_states_behavior(student_function, mock_data):
    """Tests that the model is put in train mode."""
    mock_model, retain_inputs, cb_inputs, cb_layers, *_ = mock_data
    student_function(
        model=mock_model, retain_inputs=retain_inputs, cb_inputs=cb_inputs,
        cb_layers=cb_layers, retain_coef=0.5, cb_coef=0.5
    )
    assert mock_model.called_train, "model.train() was not called. The model must be in training mode to ensure LoRA adapters are active."

@pytest.mark.task2
def test_get_lora_model_states_shapes(student_function, mock_data):
    """Tests that the output tensors have the correct shapes for the LoRA pass."""
    mock_model, retain_inputs, cb_inputs, cb_layers, n_layers, batch_size, seq_len, hidden_dim = mock_data
    cb_states_rr, retain_states_rr = student_function(
        model=mock_model, retain_inputs=retain_inputs, cb_inputs=cb_inputs,
        cb_layers=cb_layers, retain_coef=0.5, cb_coef=0.5
    )

    expected_retain_shape = (n_layers + 1, batch_size, seq_len, hidden_dim)
    assert retain_states_rr.shape == expected_retain_shape, f"Retain states shape is incorrect. Expected {expected_retain_shape}, but got {retain_states_rr.shape}"

    expected_cb_shape = (len(cb_layers), batch_size, seq_len, hidden_dim)
    assert cb_states_rr.shape == expected_cb_shape, f"Circuit breaker states shape is incorrect. Expected {expected_cb_shape}, but got {cb_states_rr.shape}"

def task2(student_func: Callable):
    """Runs all 'task2' tests against the provided student function."""
    target.func = student_func
    result_code = pytest.main([__file__, "-v", "--no-header", "-m", "task2", "-W", "ignore::pytest.PytestAssertRewriteWarning"])
    if result_code == pytest.ExitCode.OK:
        print("✅ All checks passed!")

# --- Task 3 Tests ---

@pytest.mark.task3
def test_calculate_retain_loss_calculation(student_function):
    """Tests the retain loss calculation against the reference implementation."""
    # 1. Setup mock data
    n_layers, batch_size, seq_len, hidden_dim = 4, 2, 5, 8
    num_hidden_states = n_layers + 1
    shape = (num_hidden_states, batch_size, seq_len, hidden_dim)

    retain_states_rr = torch.randn(shape)
    retain_states_orig = torch.randn(shape)
    # Create a mask with some padding
    retain_mask = torch.ones(batch_size, seq_len)
    retain_mask[:, -1] = 0 # Set last token to be padding

    # 2. Call the student's function
    student_loss = student_function(retain_states_rr, retain_states_orig, retain_mask, num_hidden_states)

    # 3. Reference implementation
    norm_diff = torch.linalg.vector_norm(retain_states_rr - retain_states_orig, ord=2, dim=-1)
    retain_attn_mask_layers = retain_mask.repeat(num_hidden_states, 1, 1)
    masked_norm_diff = norm_diff * retain_attn_mask_layers
    reference_loss = masked_norm_diff.sum() / retain_attn_mask_layers.sum()

    assert isinstance(student_loss, torch.Tensor), "Loss should be a tensor."
    assert student_loss.shape == torch.Size([]), f"Loss should be a scalar tensor, but got shape {student_loss.shape}."
    assert torch.allclose(student_loss, reference_loss), "The calculated retain loss value is incorrect."

def task3(student_func: Callable):
    """Runs all 'task3' tests against the provided student function."""
    target.func = student_func
    result_code = pytest.main([__file__, "-v", "--no-header", "-m", "task3", "-W", "ignore::pytest.PytestAssertRewriteWarning"])
    if result_code == pytest.ExitCode.OK:
        print("✅ All checks passed!")


# --- Task 4 Tests ---

@pytest.mark.task4
def test_calculate_cb_loss_calculation(student_function):
    """Tests the circuit breaker loss calculation against the reference implementation."""
    # 1. Setup mock data
    batch_size, seq_len, hidden_dim = 2, 5, 8
    cb_layers = [1, 3]
    shape = (len(cb_layers), batch_size, seq_len, hidden_dim)

    cb_states_rr = torch.randn(shape)
    cb_states_orig = torch.randn(shape)
    # Create a mask with some padding
    cb_mask = torch.ones(batch_size, seq_len)
    cb_mask[:, -2:] = 0 # Set last two tokens to be padding

    # 2. Call the student's function
    student_loss = student_function(cb_states_rr, cb_states_orig, cb_mask, cb_layers)

    # 3. Reference implementation
    similarity = torch.nn.functional.cosine_similarity(cb_states_orig, cb_states_rr, dim=-1)
    cb_attn_mask_layers = cb_mask.repeat(len(cb_layers), 1, 1)
    masked_sim = similarity * cb_attn_mask_layers
    reference_loss = torch.nn.functional.relu(masked_sim).sum() / cb_attn_mask_layers.sum()

    assert isinstance(student_loss, torch.Tensor), "Loss should be a tensor."
    assert student_loss.shape == torch.Size([]), f"Loss should be a scalar tensor, but got shape {student_loss.shape}."
    assert torch.allclose(student_loss, reference_loss), "The calculated circuit breaker loss value is incorrect."

def task4(student_func: Callable):
    """Runs all 'task4' tests against the provided student function."""
    target.func = student_func
    result_code = pytest.main([__file__, "-v", "--no-header", "-m", "task4", "-W", "ignore::pytest.PytestAssertRewriteWarning"])
    if result_code == pytest.ExitCode.OK:
        print("✅ All checks passed!")


# --- Task 5 Tests ---

@pytest.mark.task5
@pytest.mark.parametrize("retain_loss, retain_coef, cb_loss, cb_coef", [
    (torch.tensor(0.5), 0.5, torch.tensor(0.8), 0.5),
    (torch.tensor(0.7), 0.2, torch.tensor(0.3), 0.8),
    (torch.tensor(1.0), 0.0, torch.tensor(0.9), 1.0), # Test case where retain_coef is 0
    (torch.tensor(0.2), 1.0, torch.tensor(0.0), 0.0), # Test case where cb_coef is 0
])
def test_calculate_final_loss_calculation(student_function, retain_loss, retain_coef, cb_loss, cb_coef):
    """Tests the final loss calculation with various coefficients."""
    student_loss = student_function(retain_loss, retain_coef, cb_loss, cb_coef)

    if retain_coef == 0:
        reference_loss = cb_coef * cb_loss
    else:
        reference_loss = cb_coef * cb_loss + retain_coef * retain_loss

    assert isinstance(student_loss, torch.Tensor) or isinstance(student_loss, float), "Loss should be a tensor or float."
    assert pytest.approx(student_loss) == reference_loss, "The final combined loss value is incorrect."

def task5(student_func: Callable):
    """Runs all 'task5' tests against the provided student function."""
    target.func = student_func
    result_code = pytest.main([__file__, "-v", "--no-header", "-m", "task5", "-W", "ignore::pytest.PytestAssertRewriteWarning"])
    if result_code == pytest.ExitCode.OK:
        print("✅ All checks passed!")