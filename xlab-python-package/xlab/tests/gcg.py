import pytest
import torch
from torch import Tensor
from unittest.mock import patch

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
def test_one_hot_ids_shape_and_type(student_function):
    """Tests output shape, device, dtype, and gradient requirements."""
    optim_ids = torch.randint(0, 32, (1, 16))
    vocab_size = 128
    device = torch.device("cpu")
    dtype = torch.float16

    result = student_function(optim_ids, vocab_size, device, dtype)
    
    expected_shape = (1, optim_ids.shape[1], vocab_size)
    assert result.shape == expected_shape, f"Expected shape {expected_shape}, but got {result.shape}"
    assert result.device == device, "The tensor is on the wrong device."
    assert result.dtype == dtype, "The tensor has the wrong dtype."
    assert result.requires_grad, "The output tensor should have requires_grad=True"

@pytest.mark.task1
def test_one_hot_ids_values(student_function):
    """Tests the actual one-hot values for a simple case."""
    optim_ids = torch.randint(0, 32, (1, 16))
    vocab_size = 128
    device = torch.device("cpu")
    dtype = torch.float16
    
    expected = torch.nn.functional.one_hot(optim_ids, num_classes=vocab_size)
    expected = expected.to(device, dtype)
    expected.requires_grad_()

    result = student_function(optim_ids, vocab_size, device, dtype)

    assert torch.allclose(result, expected), "The one-hot encoding values are incorrect."

def task1(student_func):
    """Runs all 'task1' tests against the provided student function."""
    target.func = student_func
    result_code = pytest.main([__file__, "-v", "--no-header", "-m", "task1"])
    if result_code == pytest.ExitCode.OK:
        print("✅ All checks passed!")



@pytest.mark.task2
@pytest.mark.parametrize("one_hot_shape, embed_shape", [
    ((16, 64), (64, 48)),
    ((2, 8), (8, 4)),
    ((32, 128), (128, 16))
])
def test_create_one_hot_embeds(student_function, one_hot_shape, embed_shape):
    one_hot_ids = torch.randint(low=0, high=10, size=one_hot_shape)
    embedding_layer = torch.randint(low=0, high=10, size=embed_shape)

    result = student_function(one_hot_ids, embedding_layer)
    reference = one_hot_ids @ embedding_layer

    assert result.shape == reference.shape, f"Expected shape {reference.shape}, got {result.shape}"
    assert torch.allclose(result.data, reference.data), "The final values are incorrect."

def task2(student_func):
    target.func = student_func
    result_code = pytest.main([__file__, "-v", "--no-header", "-m", "task2"])
    if result_code == pytest.ExitCode.OK:
        print("✅ All checks passed!")



@pytest.mark.task3
@pytest.mark.parametrize("optim_shape, after_shape, target_shape", [
    ((1, 10, 32), (1, 5, 32), (1, 8, 32)),
    ((1, 20, 64), (1, 1, 64), (1, 2, 64)),
    ((1, 5, 16), (1, 10, 16), (1, 3, 16)),
])
def test_concat_full_input_shape(student_function, optim_shape, after_shape, target_shape):
    """Tests the shape of the concatenated tensor."""
    optim_embeds = torch.randn(optim_shape)
    after_embeds = torch.randn(after_shape)
    target_embeds = torch.randn(target_shape)

    result = student_function(optim_embeds, after_embeds, target_embeds)

    expected_len = optim_shape[1] + after_shape[1] + target_shape[1]
    expected_shape = (1, expected_len, optim_shape[2])
    assert result.shape == expected_shape, f"Expected shape {expected_shape}, but got {result.shape}"

@pytest.mark.task3
def test_concat_full_input_values(student_function):
    """Tests the values of the concatenated tensor."""
    optim_embeds = torch.randn(1, 10, 32)
    after_embeds = torch.randn(1, 5, 32)
    target_embeds = torch.randn(1, 8, 32)

    result = student_function(optim_embeds, after_embeds, target_embeds)
    expected = torch.cat([optim_embeds, after_embeds, target_embeds], dim=1)

    assert torch.allclose(result, expected), "The concatenated values are incorrect."

def task3(student_func):
    """Runs all 'task3' tests against the provided student function."""
    target.func = student_func
    result_code = pytest.main([__file__, "-v", "--no-header", "-m", "task3"])
    if result_code == pytest.ExitCode.OK:
        print("✅ All checks passed!")


@pytest.mark.task4
def test_get_one_hot_logits_shape(student_function):
    """Tests the shape of the output logits."""
    class MockModel:
        def __call__(self, inputs_embeds, past_key_values, use_cache):
            class MockOutput:
                def __init__(self, logits):
                    self.logits = logits
            return MockOutput(logits=torch.randn(1, inputs_embeds.shape[1], 32000))

    model = MockModel()
    full_input = torch.randn(1, 20, 128)
    prefix_cache = ((torch.randn(1, 1, 1, 1), torch.randn(1, 1, 1, 1)),) # Dummy cache

    result = student_function(model, full_input, prefix_cache)
    expected_shape = (1, 20, 32000)
    assert result.shape == expected_shape, f"Expected shape {expected_shape}, but got {result.shape}"

@pytest.mark.task4
def test_get_one_hot_logits_values(student_function):
    """Tests that the model is called with the correct arguments."""
    class MockModel:
        def __init__(self):
            self.called_with = None
        def __call__(self, inputs_embeds, past_key_values, use_cache):
            self.called_with = {
                "inputs_embeds": inputs_embeds,
                "past_key_values": past_key_values,
                "use_cache": use_cache
            }
            class MockOutput:
                def __init__(self, logits):
                    self.logits = logits
            return MockOutput(logits=torch.randn(1, inputs_embeds.shape[1], 32000))

    model = MockModel()
    full_input = torch.randn(1, 20, 128)
    prefix_cache = ((torch.randn(1, 1, 1, 1), torch.randn(1, 1, 1, 1)),)

    student_function(model, full_input, prefix_cache)

    assert torch.allclose(model.called_with["inputs_embeds"], full_input)
    assert model.called_with["past_key_values"] == prefix_cache
    assert model.called_with["use_cache"] is True

def task4(student_func):
    """Runs all 'task4' tests against the provided student function."""
    target.func = student_func
    result_code = pytest.main([__file__, "-v", "--no-header", "-m", "task4"])
    if result_code == pytest.ExitCode.OK:
        print("✅ All checks passed!")


@pytest.mark.task5
@pytest.mark.parametrize("full_input_shape, target_embeds_shape", [
    ((1, 20, 128), (1, 8, 128)),
    ((1, 50, 64), (1, 10, 64)),
])
def test_extract_target_logits_shape(student_function, full_input_shape, target_embeds_shape):
    """Tests the shape of the extracted target logits."""
    full_input = torch.randn(full_input_shape)
    logits = torch.randn(full_input_shape[0], full_input_shape[1], 32000)
    target_embeds = torch.randn(target_embeds_shape)

    result = student_function(full_input, logits, target_embeds)

    expected_shape = (target_embeds_shape[0], target_embeds_shape[1], 32000)
    assert result.shape == expected_shape, f"Expected shape {expected_shape}, but got {result.shape}"

@pytest.mark.task5
def test_extract_target_logits_values(student_function):
    """Tests the values of the extracted target logits."""
    full_input = torch.randn(1, 20, 128)
    logits = torch.randn(1, 20, 32000)
    target_embeds = torch.randn(1, 8, 128)

    result = student_function(full_input, logits, target_embeds)

    shift_diff = full_input.size(1) - target_embeds.size(1)
    expected = logits[:, shift_diff - 1 : -1, :]

    assert torch.allclose(result, expected), "The extracted logit values are incorrect."

def task5(student_func):
    """Runs all 'task5' tests against the provided student function."""
    target.func = student_func
    result_code = pytest.main([__file__, "-v", "--no-header", "-m", "task5"])
    if result_code == pytest.ExitCode.OK:
        print("✅ All checks passed!")


@pytest.mark.task6
def test_compute_loss_shape(student_function):
    """Tests that the output is a scalar tensor."""
    target_ids = torch.randint(0, 32000, (1, 10))
    target_logits = torch.randn(1, 10, 32000)

    result = student_function(target_ids, target_logits)
    assert result.shape == torch.Size([]), f"Expected a scalar tensor, but got shape {result.shape}"

@pytest.mark.task6
def test_compute_loss_values(student_function):
    """Tests the cross-entropy loss calculation."""
    target_ids = torch.randint(0, 32000, (1, 10))
    target_logits = torch.randn(1, 10, 32000)

    result = student_function(target_ids, target_logits)
    expected = torch.nn.functional.cross_entropy(
        target_logits.view(-1, target_logits.size(-1)), target_ids.view(-1)
    )

    assert torch.allclose(result, expected), "The loss value is incorrect."

def task6(student_func):
    """Runs all 'task6' tests against the provided student function."""
    target.func = student_func
    result_code = pytest.main([__file__, "-v", "--no-header", "-m", "task6"])
    if result_code == pytest.ExitCode.OK:
        print("✅ All checks passed!")


@pytest.mark.task7
def test_differentiate_one_hots_shape(student_function):
    """Tests the shape of the output gradient."""
    loss = torch.randn((), requires_grad=True)
    one_hot_ids = torch.randn(1, 10, 32000, requires_grad=True)

    # To make loss depend on one_hot_ids for grad calculation
    dummy_output = (one_hot_ids * 2).sum()
    loss = loss + dummy_output

    result = student_function(loss, one_hot_ids)
    assert result.shape == one_hot_ids.shape, f"Expected shape {one_hot_ids.shape}, but got {result.shape}"

@pytest.mark.task7
def test_differentiate_one_hots_values(student_function):
    """Tests the gradient values."""
    one_hot_ids = torch.randn(1, 10, 32000, requires_grad=True)

    # Create graph for student
    loss1 = (one_hot_ids * 2).sum() + torch.tensor(5.0)
    result = student_function(loss1, one_hot_ids)

    # Recreate identical graph for expected value
    loss2 = (one_hot_ids * 2).sum() + torch.tensor(5.0)
    expected = torch.autograd.grad(outputs=[loss2], inputs=[one_hot_ids])[0]

    assert torch.allclose(result, expected), "The gradient values are incorrect."

def task7(student_func):
    """Runs all 'task7' tests against the provided student function."""
    target.func = student_func
    result_code = pytest.main([__file__, "-v", "--no-header", "-m", "task7"])
    if result_code == pytest.ExitCode.OK:
        print("✅ All checks passed!")


@pytest.mark.task8
def test_compute_token_gradient_integration(student_function):
    """An integration test to ensure all parts of compute_token_gradient work together."""
    class MockEmbedding:
        def __init__(self, vocab_size, embed_dim):
            self.num_embeddings = vocab_size
            self.weight = torch.randn(vocab_size, embed_dim, requires_grad=True)

    class MockModel:
        def __init__(self, vocab_size, embed_dim):
            self.device = torch.device("cpu")
            self.dtype = torch.float32
            self.embedding = MockEmbedding(vocab_size, embed_dim)
            # This linear layer is crucial. It ensures the output logits have a
            # computational graph tracing back to the input embeddings.
            # Without it, the logits would be random and have no gradient history.
            self.linear = torch.nn.Linear(embed_dim, vocab_size)

        def __call__(self, inputs_embeds, past_key_values, use_cache):
            class MockOutput:
                def __init__(self, logits):
                    self.logits = logits
            # By processing the input_embeds, we create a valid graph for autograd.
            logits = self.linear(inputs_embeds)
            return MockOutput(logits=logits)

        def get_input_embeddings(self):
            return self.embedding

    vocab_size = 50
    embed_dim = 16
    model = MockModel(vocab_size, embed_dim)
    embedding_obj = model.get_input_embeddings()
    optim_ids = torch.randint(0, vocab_size, (1, 10))
    target_ids = torch.randint(0, vocab_size, (1, 5))
    after_embeds = torch.randn(1, 2, embed_dim)
    target_embeds = torch.randn(1, 5, embed_dim)
    prefix_cache = ((torch.randn(1, 1, 1, 1), torch.randn(1, 1, 1, 1)),)

    try:
        student_result = student_function(
            model, embedding_obj, optim_ids, target_ids,
            after_embeds, target_embeds, prefix_cache
        )
    except Exception as e:
        pytest.fail(f"The student's function raised an exception: {e}", pytrace=True)


    # A simplified reference implementation to check against
    one_hot_ids = torch.nn.functional.one_hot(optim_ids, num_classes=vocab_size).to(model.device, model.dtype)
    one_hot_ids.requires_grad_()
    optim_embeds = one_hot_ids @ embedding_obj.weight
    full_input = torch.cat([optim_embeds, after_embeds, target_embeds], dim=1)
    
    # Re-run the model call for the reference calculation
    output = model(inputs_embeds=full_input, past_key_values=prefix_cache, use_cache=True)
    logits = output.logits
    
    shift_diff = full_input.size(1) - target_embeds.size(1)
    target_logits = logits[:, shift_diff - 1 : -1, :]
    loss = torch.nn.functional.cross_entropy(
        target_logits.view(-1, target_logits.size(-1)), target_ids.view(-1)
    )
    reference_result = torch.autograd.grad(outputs=[loss], inputs=[one_hot_ids])[0]

    assert student_result is not None, "The function should return a tensor, not None."
    assert isinstance(student_result, Tensor), f"Expected a Tensor, but got {type(student_result)}"
    assert student_result.shape == reference_result.shape, f"Expected shape {reference_result.shape}, but got {student_result.shape}"
    # Using a slightly higher tolerance for floating point comparisons in integration tests
    assert torch.allclose(student_result, reference_result, atol=1e-6), "The final gradient values are incorrect."

def task8(student_func):
    """Runs all 'task8' tests against the provided student function."""
    target.func = student_func
    # Added --tb=long to provide a full traceback on error
    result_code = pytest.main([__file__, "-v", "--no-header", "-m", "task8"])
    if result_code == pytest.ExitCode.OK:
        print("✅ All checks passed!")


@pytest.mark.task9
@pytest.mark.parametrize("ids_shape, search_width", [
    ((1, 10), 5),
    ((1, 20), 10),
    ((1, 1), 1),
])
def test_duplicate_original_ids(student_function, ids_shape, search_width):
    """Tests the shape and values of the duplicated ids tensor."""
    ids = torch.randint(0, 100, ids_shape)

    result = student_function(ids, search_width)

    expected_shape = (search_width, ids_shape[1])
    assert result.shape == expected_shape, f"Expected shape {expected_shape}, but got {result.shape}"

    expected_values = ids.repeat(search_width, 1)
    assert torch.allclose(result, expected_values), "The duplicated values are incorrect."

def task9(student_func):
    """Runs all 'task9' tests against the provided student function."""
    target.func = student_func
    result_code = pytest.main([__file__, "-v", "--no-header", "-m", "task9"])
    if result_code == pytest.ExitCode.OK:
        print("✅ All checks passed!")


@pytest.mark.task10
@pytest.mark.parametrize("grad_shape, topk", [
    ((10, 100), 5),
    ((20, 200), 10),
    ((1, 50), 1),
])
def test_get_topk_indices(student_function, grad_shape, topk):
    """Tests the shape and values of the topk indices tensor."""
    grad = torch.randn(grad_shape)

    result = student_function(grad, topk)

    expected_shape = (grad_shape[0], topk)
    assert result.shape == expected_shape, f"Expected shape {expected_shape}, but got {result.shape}"

    expected_values = (-grad).topk(topk, dim=1).indices
    assert torch.allclose(result, expected_values), "The topk indices are incorrect."

def task10(student_func):
    """Runs all 'task10' tests against the provided student function."""
    target.func = student_func
    result_code = pytest.main([__file__, "-v", "--no-header", "-m", "task10"])
    if result_code == pytest.ExitCode.OK:
        print("✅ All checks passed!")


@pytest.mark.task11
@pytest.mark.parametrize("search_width, n_optim_tokens, n_replace", [
    (5, 10, 3),
    (10, 20, 5),
    (1, 5, 1),
])
def test_sample_id_positions(student_function, search_width, n_optim_tokens, n_replace):
    """Tests the shape and properties of the sampled id positions tensor."""
    device = torch.device("cpu")

    result = student_function(search_width, n_optim_tokens, n_replace, device)

    expected_shape = (search_width, n_replace)
    assert result.shape == expected_shape, f"Expected shape {expected_shape}, but got {result.shape}"

    # Check for uniqueness within each row
    for row in result:
        assert len(row) == len(torch.unique(row)), "Indices in a row should be unique."

    # Check if values are within the valid range
    assert result.max() < n_optim_tokens, "All indices should be less than n_optim_tokens."
    assert result.min() >= 0, "All indices should be non-negative."

def task11(student_func):
    """Runs all 'task11' tests against the provided student function."""
    target.func = student_func
    result_code = pytest.main([__file__, "-v", "--no-header", "-m", "task11"])
    if result_code == pytest.ExitCode.OK:
        print("✅ All checks passed!")


@pytest.mark.task12
@pytest.mark.parametrize("n_optim_ids, topk, search_width, n_replace", [
    (10, 5, 3, 2),
    (20, 10, 5, 4),
    (5, 2, 1, 1),
])
def test_sample_id_values(student_function, n_optim_ids, topk, search_width, n_replace):
    """Tests the shape and values of the sampled id values tensor."""
    device = torch.device("cpu")
    topk_ids = torch.randint(low=0, high=100, size=(n_optim_ids, topk))
    sampled_ids_pos = torch.randint(low=0, high=n_optim_ids, size=(search_width, n_replace))

    with patch('torch.randint') as mock_randint:
        # Make the mock return a valid tensor so the student function doesn't crash
        mock_randint.return_value = torch.zeros((search_width, n_replace, 1), device=device, dtype=torch.long)
        
        student_function(topk_ids, sampled_ids_pos, topk, search_width, n_replace, device)

        # Check the call to randint
        mock_randint.assert_called()
        args, kwargs = mock_randint.call_args
        
        # Check for low=0 in either args or kwargs
        low_is_zero = ('low' in kwargs and kwargs['low'] == 0) or (len(args) > 1 and args[0] == 0)
        assert low_is_zero, "torch.randint must be called with low=0"

    # Now run the real function to check the output
    result = student_function(topk_ids, sampled_ids_pos, topk, search_width, n_replace, device)
    expected_shape = (search_width, n_replace)
    assert result.shape == expected_shape, f"Expected shape {expected_shape}, but got {result.shape}"

    # Check if the sampled values are from the topk_ids
    for i in range(search_width):
        for j in range(n_replace):
            pos = sampled_ids_pos[i, j]
            val = result[i, j]
            assert val in topk_ids[pos], f"Sampled value {val} is not in the topk_ids for position {pos}"

def task12(student_func):
    """Runs all 'task12' tests against the provided student function."""
    target.func = student_func
    result_code = pytest.main([__file__, "-v", "--no-header", "-m", "task12"])
    if result_code == pytest.ExitCode.OK:
        print("✅ All checks passed!")


@pytest.mark.task13
@pytest.mark.parametrize("search_width, n_optim_ids, n_replace", [
    (5, 10, 3),
    (10, 20, 5),
    (1, 5, 1),
])
def test_scatter_replacements(student_function, search_width, n_optim_ids, n_replace):
    """Tests the shape and values of the scattered replacements tensor."""
    original_ids = torch.zeros(search_width, n_optim_ids, dtype=torch.long)
    sampled_ids_pos = torch.stack([torch.randperm(n_optim_ids)[:n_replace] for _ in range(search_width)])
    sampled_ids_vals = torch.randint(1, 100, (search_width, n_replace))

    result = student_function(original_ids.clone(), sampled_ids_pos, sampled_ids_vals)

    expected_shape = (search_width, n_optim_ids)
    assert result.shape == expected_shape, f"Expected shape {expected_shape}, but got {result.shape}"

    expected_values = original_ids.clone().scatter_(1, sampled_ids_pos, sampled_ids_vals)
    assert torch.allclose(result, expected_values), "The scattered values are incorrect."

def task13(student_func):
    """Runs all 'task13' tests against the provided student function."""
    target.func = student_func
    result_code = pytest.main([__file__, "-v", "--no-header", "-m", "task13"])
    if result_code == pytest.ExitCode.OK:
        print("✅ All checks passed!")

@pytest.mark.task14
def test_sample_ids_from_grad_integration(student_function):
    """An integration test to ensure all parts of sample_ids_from_grad work together."""
    ids = torch.randint(0, 100, (10,))
    grad = torch.randn(10, 100)
    search_width = 5
    topk = 10
    n_replace = 2

    try:
        # Pass not_allowed_ids=None to avoid an error with the default value
        student_result = student_function(
            ids, grad, search_width, topk, n_replace, not_allowed_ids=None
        )
    except Exception as e:
        pytest.fail(f"The student's function raised an exception: {e}", pytrace=True)

    assert student_result is not None, "The function should return a tensor, not None."
    assert isinstance(student_result, Tensor), f"Expected a Tensor, but got {type(student_result)}"
    assert student_result.shape == (search_width, len(ids)), f"Expected shape {(search_width, len(ids))}, but got {student_result.shape}"

    # Check that n_replace tokens have been replaced in each row
    for i in range(search_width):
        assert (student_result[i] != ids).sum() == n_replace, f"Expected {n_replace} replaced tokens in row {i}, but found {(student_result[i] != ids).sum()}"

def task14(student_func):
    """Runs all 'task14' tests against the provided student function."""
    target.func = student_func
    result_code = pytest.main([__file__, "-v", "--no-header", "-m", "task14"])
    if result_code == pytest.ExitCode.OK:
        print("✅ All checks passed!")
