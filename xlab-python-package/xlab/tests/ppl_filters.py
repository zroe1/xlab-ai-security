import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BatchEncoding
from unittest.mock import patch, MagicMock

class TestTarget:
    func = None

target = TestTarget()

@pytest.fixture
def student_function():
    if target.func is None:
        pytest.skip("Student function not provided.")
    return target.func

# Test for Task 1

class MockTokenizer:
    def __call__(self, prompt, return_tensors="pt"):
        return BatchEncoding({'input_ids': torch.tensor([[10, 20, 30]])})

@pytest.mark.task1
def test_tokenize_inputs(student_function):
    tokenizer = MockTokenizer()
    prompt = "a test prompt"
    device = torch.device("cpu")

    with patch('__main__.DEVICE', device):
        result = student_function(tokenizer, prompt)

    assert isinstance(result, BatchEncoding), f"Expected BatchEncoding, but got {type(result)}"
    assert "labels" in result, "The 'labels' key was not added to the output."
    assert torch.equal(result['input_ids'], result['labels']), "'labels' and 'input_ids' tensors are not identical."
    assert result['input_ids'].device == device, f"Tensor is on {result['input_ids'].device}, but should be on {device}."

def task1(student_func):
    target.func = student_func
    result_code = pytest.main([__file__, "-v", "--no-header", "-m", "task1", "-W", "ignore::pytest.PytestAssertRewriteWarning"])
    if result_code == pytest.ExitCode.OK:
        print("✅ All checks passed!")

# Test for Task 2

@pytest.mark.task2
def test_get_logits(student_function):
    mock_model = MagicMock(spec=AutoModelForCausalLM)
    mock_output = MagicMock()
    expected_logits = torch.randn(1, 5, 100, requires_grad=True)
    mock_output.logits = expected_logits
    mock_model.return_value = mock_output
    
    inputs = BatchEncoding({'input_ids': torch.tensor([[1,2,3,4,5]])})

    result_logits = student_function(mock_model, inputs)

    mock_model.assert_called_once_with(**inputs)
    assert torch.equal(result_logits, expected_logits)
    assert result_logits.grad is None, "Logits should not have gradients"

def task2(student_func):
    target.func = student_func
    result_code = pytest.main([__file__, "-v", "--no-header", "-m", "task2", "-W", "ignore::pytest.PytestAssertRewriteWarning"])
    if result_code == pytest.ExitCode.OK:
        print("✅ All checks passed!")

# Test for Task 3

@pytest.mark.task3
def test_get_log_softmax_tokens(student_function):
    logits = torch.randn(1, 10, 50)
    
    result = student_function(logits)

    expected_shape = (9, 50)
    assert result.shape == expected_shape, f"Expected shape {expected_shape}, got {result.shape}"

    expected_sliced_logits = logits[:, :-1, :].squeeze(0)
    expected_result = torch.nn.functional.log_softmax(expected_sliced_logits, dim=1)
    
    assert torch.allclose(result, expected_result), "The log_softmax values are incorrect."

def task3(student_func):
    target.func = student_func
    result_code = pytest.main([__file__, "-v", "--no-header", "-m", "task3", "-W", "ignore::pytest.PytestAssertRewriteWarning"])
    if result_code == pytest.ExitCode.OK:
        print("✅ All checks passed!")

# Test for Task 4

@pytest.mark.task4
def test_extract_nll(student_function):
    seq_len = 10
    vocab_size = 50
    log_softmaxed_toks = torch.randn(seq_len - 1, vocab_size)
    labels = torch.randint(0, vocab_size, (1, seq_len))

    result_nll = student_function(log_softmaxed_toks, labels)

    assert result_nll.shape == (seq_len - 1,)

    expected_labels = labels[:, 1:].squeeze(0)
    expected_nll = -torch.gather(
        log_softmaxed_toks,
        1,
        expected_labels.unsqueeze(-1)
    ).squeeze(-1)

    assert torch.allclose(result_nll, expected_nll)

def task4(student_func):
    target.func = student_func
    result_code = pytest.main([__file__, "-v", "--no-header", "-m", "task4", "-W", "ignore::pytest.PytestAssertRewriteWarning"])
    if result_code == pytest.ExitCode.OK:
        print("✅ All checks passed!")

# Test for Task 5

@pytest.mark.task5
def test_get_per_token_NLL_integration(student_function):
    with patch('__main__.tokenize_inputs') as mock_tokenize, \
         patch('__main__.get_logits') as mock_get_logits, \
         patch('__main__.get_log_softmax_tokens') as mock_log_softmax, \
         patch('__main__.extract_nll') as mock_extract_nll:

        mock_inputs = {'labels': 'mock_labels'}
        mock_tokenize.return_value = mock_inputs
        mock_get_logits.return_value = 'mock_logits'
        mock_log_softmax.return_value = 'mock_log_softmax'
        mock_extract_nll.return_value = 'mock_nll'

        mock_model = 'mock_model'
        mock_tokenizer = 'mock_tokenizer'
        prompt = 'test prompt'

        result = student_function(mock_model, mock_tokenizer, prompt)

        mock_tokenize.assert_called_once_with(tokenizer=mock_tokenizer, prompt=prompt)
        mock_get_logits.assert_called_once_with(model=mock_model, inputs=mock_inputs)
        mock_log_softmax.assert_called_once_with(logits='mock_logits')
        mock_extract_nll.assert_called_once_with(log_softmaxed_toks='mock_log_softmax', labels='mock_labels')

        assert result == 'mock_nll'

def task5(student_func):
    target.func = student_func
    result_code = pytest.main([__file__, "-v", "--no-header", "-m", "task5", "-W", "ignore::pytest.PytestAssertRewriteWarning"])
    if result_code == pytest.ExitCode.OK:
        print("✅ All checks passed!")

# Test for Task 6

@pytest.mark.task6
def test_get_seq_ppl(student_function):
    nll = torch.tensor([1.0, 2.0, 3.0, 4.0])
    result_ppl = student_function(nll)
    expected_ppl = torch.exp(nll.mean())
    assert isinstance(result_ppl, (float, torch.Tensor)), "Result should be a float or a tensor"
    # Convert result to tensor for comparison, avoiding warning on tensor inputs
    result_for_comparison = result_ppl if torch.is_tensor(result_ppl) else torch.tensor(result_ppl)
    assert torch.allclose(result_for_comparison, expected_ppl), f"Expected PPL {expected_ppl}, but got {result_ppl}"

def task6(student_func):
    target.func = student_func
    result_code = pytest.main([__file__, "-v", "--no-header", "-m", "task6", "-W", "ignore::pytest.PytestAssertRewriteWarning"])
    if result_code == pytest.ExitCode.OK:
        print("✅ All checks passed!")

# Test for Task 7

@pytest.mark.task7
def test_get_max_sliding_window_ppl(student_function):
    nll = torch.tensor([1.0, 4.0, 2.0, 5.0, 3.0])
    
    # Test with a standard window
    window_size = 3
    expected_max_ppl = torch.exp(torch.tensor([4.0, 2.0, 5.0]).mean())
    result_max_ppl = student_function(nll, window_size)
    assert isinstance(result_max_ppl, (float, torch.Tensor))
    # Convert result to tensor for comparison, avoiding warning on tensor inputs
    result_for_comparison = result_max_ppl if torch.is_tensor(result_max_ppl) else torch.tensor(result_max_ppl)
    assert torch.allclose(result_for_comparison, expected_max_ppl)

    # Test with window_size = seq_len
    window_size_full = len(nll)
    expected_ppl_full = torch.exp(nll.mean())
    result_ppl_full = student_function(nll, window_size_full)
    # Convert result to tensor for comparison, avoiding warning on tensor inputs
    result_for_comparison_full = result_ppl_full if torch.is_tensor(result_ppl_full) else torch.tensor(result_ppl_full)
    assert torch.allclose(result_for_comparison_full, expected_ppl_full)

    # Test assertion for window_size > seq_len
    with pytest.raises(AssertionError):
        student_function(nll, len(nll) + 1)
        
    # Test assertion for window_size not being an int
    with pytest.raises(AssertionError):
        student_function(nll, "not an int")

def task7(student_func):
    target.func = student_func
    result_code = pytest.main([__file__, "-v", "--no-header", "-m", "task7", "-W", "ignore::pytest.PytestAssertRewriteWarning"])
    if result_code == pytest.ExitCode.OK:
        print("✅ All checks passed!")