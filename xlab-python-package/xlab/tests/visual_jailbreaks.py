import pytest
import torch
from unittest.mock import patch, MagicMock

# --- Boilerplate ---
class TestTarget:
    func = None

target = TestTarget()

@pytest.fixture
def student_function():
    """A pytest fixture that provides the student's function to any test."""
    if target.func is None:
        pytest.skip("Student function not provided.")
    return target.func

# --- Mocks ---

class MockConv:
    """Mock conversation object to mimic the behavior of conv_templates."""
    def __init__(self):
        self.roles = ["user", "assistant"]
        self.messages = []

    def append_message(self, role, message):
        self.messages.append((role, message))

    def get_prompt(self):
        # Simple concatenation for predictable output in tests.
        return "".join(m[1] for m in self.messages if m[1] is not None)

    def copy(self):
        new_conv = MockConv()
        new_conv.roles = self.roles
        new_conv.messages = self.messages.copy()
        return new_conv

# --- Task 1 Tests: get_query_ids_len ---

@pytest.mark.task1
def test_get_query_ids_len_returns_int(student_function):
    """Tests if get_query_ids_len returns an integer."""
    with patch('xlab.jb_utils.conv_templates', {'instella': MockConv()}) as mock_conv_templates:
        with patch('xlab.jb_utils.tokenizer_image_token') as mock_tokenizer_token:
            mock_tokenizer_token.return_value = torch.tensor([[1, 2, 3, 4, 5]])
            question = "How to do something?"
            mock_tokenizer_instance = MagicMock()
            
            result = student_function(question, mock_tokenizer_instance)
            
            assert isinstance(result, int), "Function should return an integer."

@pytest.mark.task1
@pytest.mark.parametrize("token_count", [5, 10, 0, 1])
def test_get_query_ids_len_correct_length(student_function, token_count):
    """Tests if get_query_ids_len returns the correct length."""
    with patch('xlab.jb_utils.conv_templates', {'instella': MockConv()}) as mock_conv_templates:
        with patch('xlab.jb_utils.tokenizer_image_token') as mock_tokenizer_token:
            mock_tokenizer_token.return_value = torch.ones(1, token_count)
            question = "A question."
            mock_tokenizer_instance = MagicMock()

            result = student_function(question, mock_tokenizer_instance)

            assert result == token_count, f"Expected length {token_count}, but got {result}."

def task1(student_func):
    """Runs all 'task1' tests against the provided student function."""
    target.func = student_func
    result_code = pytest.main([__file__, "-v", "--no-header", "-m", "task1"])
    if result_code == pytest.ExitCode.OK:
        print("✅ All checks passed!")

# --- Task 2 Tests: build_full_sequence ---

@pytest.mark.task2
def test_build_full_sequence_returns_tensor(student_function):
    """Tests if build_full_sequence returns a torch.Tensor."""
    with patch('xlab.jb_utils.conv_templates', {'instella': MockConv()}) as mock_conv_templates:
        with patch('xlab.jb_utils.tokenizer_image_token') as mock_tokenizer_token:
            mock_tokenizer_token.return_value = torch.tensor([[1, 2, 3]])
            question = "A question"
            target_str = "A target"
            mock_tokenizer_instance = MagicMock()

            result = student_function(question, target_str, mock_tokenizer_instance)

            assert isinstance(result, torch.Tensor), "Function should return a torch.Tensor."

@pytest.mark.task2
@pytest.mark.parametrize("token_ids", [
    [1, 2, 3],
    list(range(20)),
    [],
    [42]
])
def test_build_full_sequence_correct_values(student_function, token_ids):
    """Tests if build_full_sequence returns a tensor with the correct values."""
    with patch('xlab.jb_utils.conv_templates', {'instella': MockConv()}) as mock_conv_templates:
        with patch('xlab.jb_utils.tokenizer_image_token') as mock_tokenizer_token:
            expected_tensor = torch.tensor(token_ids, dtype=torch.long)
            # The mock should return a 2D tensor, and the function squeezes it.
            mock_tokenizer_token.return_value = expected_tensor.unsqueeze(0)
            question = "A question"
            target_str = "A target"
            mock_tokenizer_instance = MagicMock()

            result = student_function(question, target_str, mock_tokenizer_instance)

            assert torch.equal(result, expected_tensor), "The returned tensor values are incorrect."

def task2(student_func):
    """Runs all 'task2' tests against the provided student function."""
    target.func = student_func
    result_code = pytest.main([__file__, "-v", "--no-header", "-m", "task2"])
    if result_code == pytest.ExitCode.OK:
        print("✅ All checks passed!")

# --- Task 3 Tests: create_and_mask_labels ---

@pytest.mark.task3
def test_create_and_mask_labels_not_in_place(student_function):
    """Tests that the input tensor is not modified in-place."""
    full_ids = torch.tensor([10, 20, 30, 40, 50])
    original_ids = full_ids.clone()
    query_length = 3

    student_function(full_ids, query_length)

    assert torch.equal(full_ids, original_ids), "The input tensor 'full_ids' should not be modified."

@pytest.mark.task3
@pytest.mark.parametrize("total_len, query_len", [
    (10, 3),
    (20, 20),
    (5, 0),
    (8, 1)
])
def test_create_and_mask_labels_correct_masking(student_function, total_len, query_len):
    """Tests that the labels are masked correctly."""
    full_ids = torch.arange(total_len)
    
    result = student_function(full_ids, query_len)

    assert result.shape == full_ids.shape, f"Expected shape {full_ids.shape}, but got {result.shape}"
    
    # Check masked part
    assert torch.all(result[:query_len] == -100), "The first 'query_length' elements should be -100."
    
    # Check unmasked part
    assert torch.equal(result[query_len:], full_ids[query_len:]), "The elements after 'query_length' should be unchanged."

def task3(student_func):
    """Runs all 'task3' tests against the provided student function."""
    target.func = student_func
    result_code = pytest.main([__file__, "-v", "--no-header", "-m", "task3"])
    if result_code == pytest.ExitCode.OK:
        print("✅ All checks passed!")

# --- Task 4 Tests: build_all_sequences ---

@pytest.mark.task4
@patch('__main__.create_and_mask_labels')
@patch('__main__.build_full_sequence')
@patch('__main__.get_query_ids_len')
def test_build_all_sequences_integration(mock_get_len, mock_build_seq, mock_mask_labels, student_function):
    """Tests the integration of functions within build_all_sequences."""
    # Arrange
    queries = ["q1", "q2"]
    targets = ["t1", "t2"]
    batch_images = torch.randn(2, 3, 336, 336)
    mock_tokenizer = MagicMock()

    # Set up mock return values
    mock_get_len.side_effect = [5, 6]  # Different lengths for each item in batch
    mock_build_seq.side_effect = [torch.tensor([1]*10), torch.tensor([2]*12)]
    mock_mask_labels.side_effect = [torch.tensor([-100]*5 + [1]*5), torch.tensor([-100]*6 + [2]*6)]

    # Act
    result = student_function(queries, targets, batch_images, mock_tokenizer)

    # Assert
    assert isinstance(result, tuple) and len(result) == 3, "Should return a tuple of 3 lists."
    
    batch_input_ids, batch_labels, batch_image_sizes = result

    assert mock_get_len.call_count == 2
    assert mock_build_seq.call_count == 2
    assert mock_mask_labels.call_count == 2

    assert len(batch_input_ids) == 2
    assert torch.equal(batch_input_ids[0], torch.tensor([1]*10))
    assert torch.equal(batch_input_ids[1], torch.tensor([2]*12))

    assert len(batch_labels) == 2
    assert torch.equal(batch_labels[0], torch.tensor([-100]*5 + [1]*5))
    assert torch.equal(batch_labels[1], torch.tensor([-100]*6 + [2]*6))

    assert len(batch_image_sizes) == 2
    assert batch_image_sizes[0] == batch_images[0].size()
    assert batch_image_sizes[1] == batch_images[1].size()

def task4(student_func):
    """Runs all 'task4' tests against the provided student function."""
    target.func = student_func
    # Patch the functions from the notebook's context (__main__)
    with patch('__main__.get_query_ids_len', MagicMock(return_value=5)), \
         patch('__main__.build_full_sequence', MagicMock(return_value=torch.ones(10))), \
         patch('__main__.create_and_mask_labels', MagicMock(return_value=torch.ones(10)*-1)):
        
        result_code = pytest.main([__file__, "-v", "--no-header", "-m", "task4"])
        if result_code == pytest.ExitCode.OK:
            print("✅ All checks passed!")

# --- Task 5 Tests: pad_sequences ---

@pytest.mark.task5
def test_pad_sequences_output_type_and_shape(student_function):
    """Tests the output type and shape from pad_sequences."""
    batch_input_ids = [torch.tensor([1, 2]), torch.tensor([3, 4, 5])]
    batch_labels = [torch.tensor([10, 20]), torch.tensor([30, 40, 50])]
    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token_id = 0

    result_ids, result_labels = student_function(batch_input_ids, batch_labels, mock_tokenizer)

    assert isinstance(result_ids, torch.Tensor)
    assert isinstance(result_labels, torch.Tensor)
    assert result_ids.shape == (2, 3), "Padded input_ids tensor has incorrect shape."
    assert result_labels.shape == (2, 3), "Padded labels tensor has incorrect shape."

@pytest.mark.task5
def test_pad_sequences_padding_values(student_function):
    """Tests that the padding values are correct."""
    batch_input_ids = [torch.tensor([1, 2]), torch.tensor([3, 4, 5])]
    batch_labels = [torch.tensor([10, 20]), torch.tensor([-100, 40, 50])]
    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token_id = 0

    result_ids, result_labels = student_function(batch_input_ids, batch_labels, mock_tokenizer)

    expected_ids = torch.tensor([[1, 2, 0], [3, 4, 5]])
    expected_labels = torch.tensor([[10, 20, -100], [-100, 40, 50]])

    assert torch.equal(result_ids, expected_ids), "Padded input_ids have incorrect values."
    assert torch.equal(result_labels, expected_labels), "Padded labels have incorrect values."

def task5(student_func):
    """Runs all 'task5' tests against the provided student function."""
    target.func = student_func
    result_code = pytest.main([__file__, "-v", "--no-header", "-m", "task5"])
    if result_code == pytest.ExitCode.OK:
        print("✅ All checks passed!")

# --- Task 6 Tests: get_batch_loss ---

class MockModelOutput:
    def __init__(self, loss_val):
        self.loss = torch.tensor(loss_val)

@pytest.mark.task6
@patch('__main__.pad_sequences')
@patch('__main__.build_all_sequences')
def test_get_batch_loss_integration(mock_build, mock_pad, student_function):
    """Tests the integration of functions within get_batch_loss."""
    # Arrange
    mock_model = MagicMock(return_value=MockModelOutput(1.23))
    mock_model.device = 'cpu'
    mock_tokenizer = MagicMock()
    image_tensor = torch.randn(1, 3, 336, 336)
    queries = ["q1", "q2"]
    targets = ["t1", "t2"]

    # Mock dependencies
    mock_build.return_value = (["dummy_ids"], ["dummy_labels"], ["dummy_sizes"])
    padded_ids = torch.randn(2, 10)
    padded_labels = torch.randn(2, 10)
    mock_pad.return_value = (padded_ids, padded_labels)

    # Act
    loss = student_function(mock_model, image_tensor, queries, targets, mock_tokenizer)

    # Assert
    mock_build.assert_called_once()
    mock_pad.assert_called_once()
    
    # Check that the model was called with the correctly padded tensors
    model_call_args = mock_model.call_args[1]
    assert torch.equal(model_call_args['input_ids'], padded_ids)
    assert torch.equal(model_call_args['labels'], padded_labels)
    assert model_call_args['images'].shape[0] == len(queries)

    assert torch.isclose(loss, torch.tensor(1.23))

def task6(student_func):
    """Runs all 'task6' tests against the provided student function."""
    target.func = student_func
    with patch('__main__.build_all_sequences', MagicMock(return_value=([],[],[]))), \
         patch('__main__.pad_sequences', MagicMock(return_value=(torch.empty(0), torch.empty(0)))):

        result_code = pytest.main([__file__, "-v", "--no-header", "-m", "task6"])
        if result_code == pytest.ExitCode.OK:
            print("✅ All checks passed!")