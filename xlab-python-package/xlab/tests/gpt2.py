"""
Tests for section 1.0 of the AI Security course.
"""

import pytest
import torch

class TestTarget:
    func = None

target = TestTarget()

@pytest.fixture
def student_function():
    """A pytest fixture that provides the student's function to any test."""
    if target.func is None:
        pytest.skip("Student function not provided.")
    return target.func

@pytest.mark.task1
def test_function_runs_without_crashing(student_function):
    """Test that the function can be called without raising an exception."""
    try:
        student_function("test input")
    except Exception as e:
        pytest.fail(f"Function crashed with error: {e}")

@pytest.mark.task1
@pytest.mark.parametrize("input_text,expected_output,description", [
    ("Hello there gpt-2", ['Hello', ' there', ' g', 'pt', '-', '2'], "basic text with hyphen"),
    ("??!hello--*- world#$", ['??', '!', 'hello', '--', '*', '-', ' world', '#$'], "special characters and symbols"),
    ("https://xrisk.uchicago.edu/fellowship/", ['https', '://', 'x', 'risk', '.', 'uch', 'icago', '.', 'edu', '/', 'fell', 'owship', '/'], "URL with dots and slashes"),
    ("", [], "empty string"),
    (".,.,.,.,.,.,.,", ['.,', '.,', '.,', '.,', '.,', '.,', '.,'], "repeated punctuation"),
])
def test_tokenization_cases(student_function, input_text, expected_output, description):
    """Test tokenization function with various input cases."""
    result = student_function(input_text)
    assert result == expected_output, f"Test case '{description}' failed. Expected: {expected_output}, Got: {result}"

def task1(student_func):
    """Runs all 'task1' tests against the provided student function."""
    target.func = student_func
    result_code = pytest.main([__file__, "-v", "--no-header", "-m", "task1", "-W", "ignore"])
    if result_code == pytest.ExitCode.OK:
        print("✅ All checks passed!")

# Tests for get_gpt2_probs function
class TestTargetProbs:
    func = None
    vocab_size = 50257  # Default GPT-2 vocab size

target_probs = TestTargetProbs()

@pytest.fixture
def student_probs_function():
    """A pytest fixture that provides the student's get_gpt2_probs function to any test."""
    if target_probs.func is None:
        pytest.skip("Student get_gpt2_probs function not provided.")
    return target_probs.func

@pytest.mark.task2
@pytest.mark.parametrize("batch_size,seq_len,vocab_size,temperature,description", [
    (1, 10, 50257, 1.0, "single batch, standard GPT-2 vocab, temp=1.0"),
    (4, 5, 50257, 1.0, "multi-batch, short sequence, temp=1.0"),
    (2, 20, 50257, 0.8, "multi-batch, longer sequence, temp=0.8"),
    (1, 1, 50257, 1.5, "single token, temp=1.5"),
])
def test_get_gpt2_probs_basic_functionality(student_probs_function, batch_size, seq_len, vocab_size, temperature, description):
    """Test basic functionality of get_gpt2_probs with different input shapes and temperatures."""
    # Mock VOCAB_SIZE constant that the function expects
    import sys
    original_vocab_size = getattr(sys.modules[student_probs_function.__module__], 'VOCAB_SIZE', None)
    setattr(sys.modules[student_probs_function.__module__], 'VOCAB_SIZE', vocab_size)
    
    try:
        logits = torch.randn(batch_size, seq_len, vocab_size)
        result = student_probs_function(logits, temperature)
        
        # Check output shape matches input shape
        assert result.shape == logits.shape, f"Output shape {result.shape} doesn't match input shape {logits.shape}"
        
        # Check that result is a probability distribution (values between 0 and 1)
        assert torch.all(result >= 0), f"Probabilities contain negative values: {result.min().item()}"
        assert torch.all(result <= 1), f"Probabilities contain values > 1: {result.max().item()}"
        
        # Check that probabilities sum to 1 along vocab dimension
        prob_sums = torch.sum(result, dim=-1)
        assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-6), \
            f"Probabilities don't sum to 1: {prob_sums}"
    
    finally:
        # Restore original VOCAB_SIZE
        if original_vocab_size is not None:
            setattr(sys.modules[student_probs_function.__module__], 'VOCAB_SIZE', original_vocab_size)

@pytest.mark.task2
def test_get_gpt2_probs_softmax_properties(student_probs_function):
    """Test that the function properly implements softmax properties."""
    # Mock VOCAB_SIZE
    import sys
    original_vocab_size = getattr(sys.modules[student_probs_function.__module__], 'VOCAB_SIZE', None)
    setattr(sys.modules[student_probs_function.__module__], 'VOCAB_SIZE', 50257)
    
    try:
        logits = torch.tensor([[[1.0, 2.0, 3.0] + [0.0] * 50254]])  # Simple test case
        result = student_probs_function(logits, 1.0)  # temp=1.0 should be standard softmax
        
        # Compare with reference softmax implementation
        expected = torch.nn.functional.softmax(logits, dim=2)
        assert torch.allclose(result, expected, atol=1e-6), \
            f"Output doesn't match expected softmax result. Got: {result}, Expected: {expected}"
    
    finally:
        if original_vocab_size is not None:
            setattr(sys.modules[student_probs_function.__module__], 'VOCAB_SIZE', original_vocab_size)

@pytest.mark.task2
def test_get_gpt2_probs_wrong_dimensions(student_probs_function):
    """Test that function raises AssertionError for wrong number of dimensions."""
    import sys
    original_vocab_size = getattr(sys.modules[student_probs_function.__module__], 'VOCAB_SIZE', None)
    setattr(sys.modules[student_probs_function.__module__], 'VOCAB_SIZE', 50257)
    
    try:
        # Test 2D input (should fail)
        logits_2d = torch.randn(10, 50257)
        with pytest.raises(AssertionError):
            student_probs_function(logits_2d, 1.0)
        
        # Test 4D input (should fail)
        logits_4d = torch.randn(1, 1, 10, 50257)
        with pytest.raises(AssertionError):
            student_probs_function(logits_4d, 1.0)
    
    finally:
        if original_vocab_size is not None:
            setattr(sys.modules[student_probs_function.__module__], 'VOCAB_SIZE', original_vocab_size)

@pytest.mark.task2
def test_get_gpt2_probs_wrong_vocab_size(student_probs_function):
    """Test that function raises AssertionError for wrong vocabulary size."""
    import sys
    original_vocab_size = getattr(sys.modules[student_probs_function.__module__], 'VOCAB_SIZE', None)
    setattr(sys.modules[student_probs_function.__module__], 'VOCAB_SIZE', 50257)
    
    try:
        # Test wrong vocab size (should fail)
        logits_wrong_vocab = torch.randn(1, 10, 1000)  # vocab_size=1000 instead of 50257
        with pytest.raises(AssertionError):
            student_probs_function(logits_wrong_vocab, 1.0)
    
    finally:
        if original_vocab_size is not None:
            setattr(sys.modules[student_probs_function.__module__], 'VOCAB_SIZE', original_vocab_size)

@pytest.mark.task2
def test_get_gpt2_probs_extreme_values(student_probs_function):
    """Test function behavior with extreme logit values."""
    import sys
    original_vocab_size = getattr(sys.modules[student_probs_function.__module__], 'VOCAB_SIZE', None)
    setattr(sys.modules[student_probs_function.__module__], 'VOCAB_SIZE', 5)  # Smaller vocab for easier testing
    
    try:
        # Test with very large positive and negative values
        logits = torch.tensor([[[100.0, -100.0, 0.0, 50.0, -50.0]]])
        result = student_probs_function(logits, 1.0)
        
        # Should still be valid probabilities
        assert torch.all(result >= 0), "Probabilities should be non-negative even with extreme values"
        assert torch.all(result <= 1), "Probabilities should be <= 1 even with extreme values"
        assert torch.allclose(torch.sum(result, dim=-1), torch.ones(1, 1), atol=1e-6), \
            "Probabilities should sum to 1 even with extreme values"
        
        # The largest logit should have the highest probability
        assert torch.argmax(result, dim=-1).item() == 0, "Largest logit should correspond to highest probability"
    
    finally:
        if original_vocab_size is not None:
            setattr(sys.modules[student_probs_function.__module__], 'VOCAB_SIZE', original_vocab_size)

@pytest.mark.task2
@pytest.mark.parametrize("temperature,description", [
    (1.0, "standard temperature"),
    (0.5, "low temperature (more confident)"),
    (2.0, "high temperature (less confident)"),
    (0.1, "very low temperature"),
])
def test_get_gpt2_probs_temperature_effects(student_probs_function, temperature, description):
    """Test that temperature properly affects the sharpness of probability distributions."""
    import sys
    original_vocab_size = getattr(sys.modules[student_probs_function.__module__], 'VOCAB_SIZE', None)
    setattr(sys.modules[student_probs_function.__module__], 'VOCAB_SIZE', 5)  # Small vocab for easier testing
    
    try:
        # Create logits with clear differences
        logits = torch.tensor([[[2.0, 1.0, 0.0, -1.0, -2.0]]])
        result = student_probs_function(logits, temperature)
        
        # Compare with reference implementation
        expected = torch.nn.functional.softmax(logits / temperature, dim=2)
        assert torch.allclose(result, expected, atol=1e-6), \
            f"Temperature scaling incorrect for {description}"
        
        # Test temperature effects on distribution sharpness
        if temperature < 1.0:
            # Lower temperature should make distribution more peaked
            # The max probability should be higher than with temp=1.0
            temp_1_result = torch.nn.functional.softmax(logits, dim=2)
            assert torch.max(result) > torch.max(temp_1_result), \
                f"Low temperature should create more peaked distribution"
        
        elif temperature > 1.0:
            # Higher temperature should make distribution more uniform
            # The max probability should be lower than with temp=1.0
            temp_1_result = torch.nn.functional.softmax(logits, dim=2)
            assert torch.max(result) < torch.max(temp_1_result), \
                f"High temperature should create more uniform distribution"
        
    finally:
        if original_vocab_size is not None:
            setattr(sys.modules[student_probs_function.__module__], 'VOCAB_SIZE', original_vocab_size)

@pytest.mark.task2
def test_get_gpt2_probs_temperature_edge_cases(student_probs_function):
    """Test edge cases for temperature values."""
    import sys
    original_vocab_size = getattr(sys.modules[student_probs_function.__module__], 'VOCAB_SIZE', None)
    setattr(sys.modules[student_probs_function.__module__], 'VOCAB_SIZE', 3)
    
    try:
        logits = torch.tensor([[[1.0, 2.0, 3.0]]])
        
        # Test very small temperature (should create very peaked distribution)
        result_small_temp = student_probs_function(logits, 0.01)
        # The highest logit (index 2) should dominate
        assert torch.argmax(result_small_temp, dim=-1).item() == 2
        assert result_small_temp[0, 0, 2] > 0.9, "Very low temperature should create very peaked distribution"
        
        # Test that we still get valid probabilities
        assert torch.allclose(torch.sum(result_small_temp, dim=-1), torch.ones(1, 1), atol=1e-6), \
            "Probabilities should still sum to 1 with extreme temperature"
        
    finally:
        if original_vocab_size is not None:
            setattr(sys.modules[student_probs_function.__module__], 'VOCAB_SIZE', original_vocab_size)

# Tests for get_gpt2_next_token_loss function
class TestTargetLoss:
    func = None
    model = None
    tokenizer = None

target_loss = TestTargetLoss()

@pytest.fixture
def student_loss_function():
    """A pytest fixture that provides the student's get_gpt2_next_token_loss function to any test."""
    if target_loss.func is None:
        pytest.skip("Student get_gpt2_next_token_loss function not provided.")
    return target_loss.func

@pytest.fixture
def mock_model_and_tokenizer():
    """Provides mock model and tokenizer for testing."""
    from unittest.mock import Mock
    import torch
    
    # Mock tokenizer
    mock_tokenizer = Mock()
    mock_tokenizer.return_value = {
        'input_ids': torch.tensor([[1, 2, 3, 4]]),  # Sample token IDs
        'attention_mask': torch.tensor([[1, 1, 1, 1]])
    }
    
    # Mock model
    mock_model = Mock()
    mock_output = Mock()
    # Create realistic logits: batch_size=1, seq_len=4, vocab_size=50257
    mock_output.logits = torch.randn(1, 4, 50257)
    mock_model.return_value = mock_output
    
    return mock_model, mock_tokenizer

@pytest.mark.task3
def test_loss_function_basic_functionality(student_loss_function, mock_model_and_tokenizer):
    """Test basic functionality: runs without crashing, returns scalar, non-negative."""
    mock_model, mock_tokenizer = mock_model_and_tokenizer
    
    import sys
    module = sys.modules[student_loss_function.__module__]
    original_model = getattr(module, 'model', None)
    original_tokenizer = getattr(module, 'tokenizer', None)
    
    try:
        setattr(module, 'model', mock_model)
        setattr(module, 'tokenizer', mock_tokenizer)
        
        correct_token_idx = torch.tensor([123])
        result = student_loss_function(mock_model, "test input", correct_token_idx)
        
        # Should return a scalar tensor that is non-negative
        assert isinstance(result, torch.Tensor), f"Expected torch.Tensor, got {type(result)}"
        assert result.dim() == 0, f"Expected scalar tensor (0 dimensions), got {result.dim()} dimensions"
        assert result.item() >= 0, f"Loss should be non-negative, got {result.item()}"
        
    finally:
        if original_model is not None:
            setattr(module, 'model', original_model)
        if original_tokenizer is not None:
            setattr(module, 'tokenizer', original_tokenizer)

@pytest.mark.task3
def test_loss_function_uses_last_token_logits(student_loss_function):
    """Test that the function correctly uses the last token's logits."""
    from unittest.mock import Mock
    import torch
    
    # Create a mock model that returns predictable logits
    mock_model = Mock()
    mock_output = Mock()
    
    # Create logits where we know what the last token logits should be
    # Shape: (batch_size=1, seq_len=3, vocab_size=5)
    logits = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0],    # First token
                           [2.0, 3.0, 4.0, 5.0, 6.0],    # Second token  
                           [10.0, 1.0, 2.0, 3.0, 4.0]]])  # Last token (should be used)
    mock_output.logits = logits
    mock_model.return_value = mock_output
    
    # Mock tokenizer
    mock_tokenizer = Mock()
    mock_tokenizer.return_value = {
        'input_ids': torch.tensor([[1, 2, 3]]),
        'attention_mask': torch.tensor([[1, 1, 1]])
    }
    
    import sys
    module = sys.modules[student_loss_function.__module__]
    original_model = getattr(module, 'model', None)
    original_tokenizer = getattr(module, 'tokenizer', None)
    
    try:
        setattr(module, 'model', mock_model)
        setattr(module, 'tokenizer', mock_tokenizer)
        
        # Test with correct token index 0 (which has highest logit in last position)
        correct_token_idx = torch.tensor([0])
        result = student_loss_function(mock_model, "test", correct_token_idx)
        
        # Calculate expected loss manually using last token logits
        last_token_logits = logits[0, -1, :]  # [10.0, 1.0, 2.0, 3.0, 4.0]
        expected_loss = torch.nn.functional.cross_entropy(last_token_logits.unsqueeze(0), correct_token_idx)
        
        assert torch.allclose(result, expected_loss, atol=1e-6), \
            f"Loss calculation incorrect. Expected: {expected_loss}, Got: {result}"
            
    finally:
        if original_model is not None:
            setattr(module, 'model', original_model)
        if original_tokenizer is not None:
            setattr(module, 'tokenizer', original_tokenizer)

def task2(student_func, vocab_size=50257):
    """Runs all 'task2' tests against the provided get_gpt2_probs function."""
    target_probs.func = student_func
    target_probs.vocab_size = vocab_size
    result_code = pytest.main([__file__, "-v", "--no-header", "-m", "task2", "-W", "ignore"])
    if result_code == pytest.ExitCode.OK:
        print("✅ All checks passed!")

# Tests for get_gpt2_loss_on_sequence function
class TestTargetSequenceLoss:
    func = None
    model = None
    tokenizer = None

target_sequence_loss = TestTargetSequenceLoss()

@pytest.fixture
def student_sequence_loss_function():
    """A pytest fixture that provides the student's get_gpt2_loss_on_sequence function to any test."""
    if target_sequence_loss.func is None:
        pytest.skip("Student get_gpt2_loss_on_sequence function not provided.")
    return target_sequence_loss.func

@pytest.fixture
def mock_sequence_model_and_tokenizer():
    """Provides mock model and tokenizer for sequence loss testing."""
    from unittest.mock import Mock
    import torch
    
    # Mock tokenizer - returns an object with .input_ids attribute
    mock_tokenizer = Mock()
    mock_encoded_output = Mock()
    mock_encoded_output.input_ids = torch.tensor([[1, 2, 3, 4, 5]])  # 5 tokens
    mock_encoded_output.attention_mask = torch.tensor([[1, 1, 1, 1, 1]])
    # Also support dict-style access for **encoded_input unpacking
    mock_encoded_output.__getitem__ = lambda self, key: getattr(self, key)
    mock_encoded_output.keys = lambda: ['input_ids', 'attention_mask']
    mock_tokenizer.return_value = mock_encoded_output
    
    # Mock model
    mock_model = Mock()
    mock_output = Mock()
    # Create logits: batch_size=1, seq_len=5, vocab_size=10 (smaller for easier testing)
    mock_output.logits = torch.randn(1, 5, 10)
    mock_model.return_value = mock_output
    
    return mock_model, mock_tokenizer

@pytest.mark.task4
def test_sequence_loss_basic_functionality(student_sequence_loss_function, mock_sequence_model_and_tokenizer):
    """Test basic functionality: runs without crashing, returns scalar, non-negative."""
    mock_model, mock_tokenizer = mock_sequence_model_and_tokenizer
    
    import sys
    module = sys.modules[student_sequence_loss_function.__module__]
    original_model = getattr(module, 'model', None)
    original_tokenizer = getattr(module, 'tokenizer', None)
    
    try:
        setattr(module, 'model', mock_model)
        setattr(module, 'tokenizer', mock_tokenizer)
        
        result = student_sequence_loss_function(mock_model, "test input sequence")
        
        # Should return a scalar tensor that is non-negative
        assert isinstance(result, torch.Tensor), f"Expected torch.Tensor, got {type(result)}"
        assert result.dim() == 0, f"Expected scalar tensor (0 dimensions), got {result.dim()} dimensions"
        assert result.item() >= 0, f"Loss should be non-negative, got {result.item()}"
        
    finally:
        if original_model is not None:
            setattr(module, 'model', original_model)
        if original_tokenizer is not None:
            setattr(module, 'tokenizer', original_tokenizer)

@pytest.mark.task4
def test_sequence_loss_proper_alignment(student_sequence_loss_function):
    """Test that the function properly aligns logits and labels (key sequence loss property)."""
    from unittest.mock import Mock
    import torch
    
    # Create predictable logits and token IDs
    token_ids = [10, 20, 30, 40]  # 4 tokens
    vocab_size = 50
    
    mock_tokenizer = Mock()
    mock_encoded_output = Mock()
    mock_encoded_output.input_ids = torch.tensor([token_ids])
    mock_encoded_output.attention_mask = torch.tensor([[1] * len(token_ids)])
    # Support dict-style access for **encoded_input unpacking
    mock_encoded_output.__getitem__ = lambda self, key: getattr(self, key)
    mock_encoded_output.keys = lambda: ['input_ids', 'attention_mask']
    mock_tokenizer.return_value = mock_encoded_output
    
    # Create logits where we can predict the loss
    mock_model = Mock()
    mock_output = Mock()
    logits = torch.randn(1, len(token_ids), vocab_size)
    mock_output.logits = logits
    mock_model.return_value = mock_output
    
    import sys
    module = sys.modules[student_sequence_loss_function.__module__]
    original_model = getattr(module, 'model', None)
    original_tokenizer = getattr(module, 'tokenizer', None)
    
    try:
        setattr(module, 'model', mock_model)
        setattr(module, 'tokenizer', mock_tokenizer)
        
        result = student_sequence_loss_function(mock_model, "test")
        
        # Calculate expected loss manually
        # logits[:, :-1, :] should align with labels[:,1:]
        expected_logits = logits[0, :-1, :]  # Remove last prediction: shape [3, vocab_size]
        expected_labels = torch.tensor(token_ids[1:])  # Remove first token: [20, 30, 40]
        expected_loss = torch.nn.functional.cross_entropy(expected_logits, expected_labels)
        
        assert torch.allclose(result, expected_loss, atol=1e-6), \
            f"Sequence alignment incorrect. Expected: {expected_loss}, Got: {result}"
            
    finally:
        if original_model is not None:
            setattr(module, 'model', original_model)
        if original_tokenizer is not None:
            setattr(module, 'tokenizer', original_tokenizer)

@pytest.mark.task4
def test_sequence_loss_vs_ground_truth(student_sequence_loss_function):
    """Test student function against the ground truth model.forward() implementation."""
    from unittest.mock import Mock
    import torch
    
    # Create test setup
    token_ids = [1, 2, 3, 4, 5]
    vocab_size = 50  # Use larger vocab to avoid any bounds issues
    
    mock_tokenizer = Mock()
    mock_encoded_output = Mock()
    mock_encoded_output.input_ids = torch.tensor([token_ids])
    mock_encoded_output.attention_mask = torch.tensor([[1] * len(token_ids)])
    # Support dict-style access for **encoded_input unpacking  
    mock_encoded_output.__getitem__ = lambda self, key: getattr(self, key)
    mock_encoded_output.keys = lambda: ['input_ids', 'attention_mask']
    mock_tokenizer.return_value = mock_encoded_output
    
    # Create FIXED logits that will be reused for both calls
    fixed_logits = torch.randn(1, len(token_ids), vocab_size)
    
    # Mock model that can handle both regular forward and forward with labels
    mock_model = Mock()
    
    def model_forward(*args, **kwargs):
        mock_output = Mock()
        # Use the same fixed logits every time
        mock_output.logits = fixed_logits
        
        # If labels are provided, compute the loss like the ground truth
        if 'labels' in kwargs:
            labels = kwargs['labels']
            # Standard language modeling loss: shift logits and labels
            shift_logits = fixed_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, vocab_size), 
                shift_labels.view(-1)
            )
            mock_output.loss = loss
        
        return mock_output
    
    mock_model.side_effect = model_forward
    
    import sys
    module = sys.modules[student_sequence_loss_function.__module__]
    original_model = getattr(module, 'model', None)
    original_tokenizer = getattr(module, 'tokenizer', None)
    
    try:
        setattr(module, 'model', mock_model)
        setattr(module, 'tokenizer', mock_tokenizer)
        
        # Get student result
        student_result = student_sequence_loss_function(mock_model, "test")
        
        # Get ground truth result using the simpler version
        # Simulate the ground truth function call (using same logits via our mock)
        encoded_input = mock_tokenizer("test")
        labels = encoded_input.input_ids.clone()
        ground_truth_output = mock_model(**encoded_input, labels=labels)
        ground_truth_result = ground_truth_output.loss
        
        # Should be very close (allowing for small numerical differences)
        assert torch.allclose(student_result, ground_truth_result, atol=1e-5), \
            f"Student result differs from ground truth. Student: {student_result}, Ground truth: {ground_truth_result}"
            
    finally:
        if original_model is not None:
            setattr(module, 'model', original_model)
        if original_tokenizer is not None:
            setattr(module, 'tokenizer', original_tokenizer)

def task3(student_func):
    """Runs all 'task3' tests against the provided get_gpt2_next_token_loss function."""
    target_loss.func = student_func
    result_code = pytest.main([__file__, "-v", "--no-header", "-m", "task3", "-W", "ignore"])
    if result_code == pytest.ExitCode.OK:
        print("✅ All checks passed!")

def task4(student_func):
    """Runs all 'task4' tests against the provided get_gpt2_loss_on_sequence function."""
    target_sequence_loss.func = student_func
    result_code = pytest.main([__file__, "-v", "--no-header", "-m", "task4", "-W", "ignore"])
    if result_code == pytest.ExitCode.OK:
        print("✅ All checks passed!")