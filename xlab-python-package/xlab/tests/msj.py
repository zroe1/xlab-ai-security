import pytest
import json
import os
import tempfile
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

@pytest.fixture
def sample_dataset():
    """Provides a standard sample dataset used across tests."""
    return [
        {"question": "What is the capital of France?", "response": "The capital of France is Paris."},
        {"question": "How do you say hello in Spanish?", "response": "In Spanish, you say 'Hola' to greet someone."},
        {"question": "What is 2 + 2?", "response": "2 + 2 equals 4."},
        {"question": "What color is the sky?", "response": "The sky is typically blue during a clear day."},
        {"question": "Who wrote Romeo and Juliet?", "response": "William Shakespeare wrote Romeo and Juliet."}
    ]

@pytest.mark.task1
def test_default_parameters(student_function, sample_dataset):
    """Tests the function with all default parameters."""
    with patch('xlab.utils.get_beauty_qa_dataset', return_value=sample_dataset):
        result = student_function()
    
    # Check that result is a string
    assert isinstance(result, str), "Function should return a string"
    
    # Check that result contains expected default strings
    assert "<|user|>" in result, "Result should contain default user string"
    assert "<|assistant|>" in result, "Result should contain default assistant string"
    assert "<\\s>" in result, "Result should contain default end_of_text string"
    
    # Check that result ends with the expected pattern
    assert result.endswith("<\\s>\n"), "Result should end with '<\\s>\\n'"

@pytest.mark.task1
def test_custom_parameters(student_function, sample_dataset):
    """Tests the function with custom parameters."""
    custom_user = "[USER]"
    custom_assistant = "[ASSISTANT]"
    custom_end = "[END]"
    with patch('xlab.utils.get_beauty_qa_dataset', return_value=sample_dataset):
        with patch('random.shuffle') as mock_shuffle:
            mock_shuffle.side_effect = lambda x: None
            result = student_function(
                user_string=custom_user,
                assistant_string=custom_assistant,
                end_of_text_string=custom_end,
                num_shots=2
            )
    
    # Check that custom strings are used
    assert custom_user in result, "Result should contain custom user string"
    assert custom_assistant in result, "Result should contain custom assistant string"
    assert custom_end in result, "Result should contain custom end_of_text string"
    
    # Check that default strings are NOT present
    assert "<|user|>" not in result, "Result should not contain default user string"
    assert "<|assistant|>" not in result, "Result should not contain default assistant string"
    assert "<\\s>" not in result, "Result should not contain default end_of_text string"

@pytest.mark.task1
def test_num_shots_parameter(student_function, sample_dataset):
    """Tests that num_shots parameter limits the number of examples correctly."""
    with patch('xlab.utils.get_beauty_qa_dataset', return_value=sample_dataset):
        with patch('random.shuffle') as mock_shuffle:
            mock_shuffle.side_effect = lambda x: None
            result = student_function(num_shots=3)
    
    # Count the number of question-response pairs
    user_count = result.count("<|user|>")
    assistant_count = result.count("<|assistant|>")
    
    assert user_count == 3, f"Expected 3 user prompts, got {user_count}"
    assert assistant_count == 3, f"Expected 3 assistant responses, got {assistant_count}"

@pytest.mark.task1 
def test_exact_format_structure(student_function):
    """Tests the exact format structure matches the expected template."""
    sample_data = [
        {
            "question": "Test question?",
            "response": "Test response."
        }
    ]
    
    with patch('xlab.utils.get_beauty_qa_dataset', return_value=sample_data):
        with patch('random.shuffle') as mock_shuffle:
            mock_shuffle.side_effect = lambda x: None
            result = student_function(num_shots=1)
            expected = "<|user|>\nTest question?<\\s>\n<|assistant|>\nTest response.<\\s>\n"
            assert result == expected, f"Expected exact format:\n{repr(expected)}\nGot:\n{repr(result)}"

@pytest.mark.task1
def test_newline_handling(student_function):
    """Tests that newline characters are handled correctly."""
    sample_data = [
        {
            "question": "Question with\nnewlines",
            "response": "Response with\nnewlines"
        }
    ]
    
    with patch('xlab.utils.get_beauty_qa_dataset', return_value=sample_data):
        with patch('random.shuffle') as mock_shuffle:
            mock_shuffle.side_effect = lambda x: None
            result = student_function(num_shots=1)
            assert result.startswith("<|user|>\n"), "Should start with user tag and newline"
            assert "<\\s>\n<|assistant|>\n" in result, "Should have proper transition from user to assistant"
            assert result.endswith("<\\s>\n"), "Should end with end_of_text and newline"

@pytest.mark.task1
def test_multiple_shots_format(student_function):
    """Tests the format when multiple shots are used."""
    sample_data = [
        {"question": "Q1", "response": "R1"},
        {"question": "Q2", "response": "R2"},
        {"question": "Q3", "response": "R3"}
    ]
    
    with patch('xlab.utils.get_beauty_qa_dataset', return_value=sample_data):
        with patch('random.shuffle') as mock_shuffle:
            mock_shuffle.side_effect = lambda x: None
            result = student_function(num_shots=3)
            expected = (
                "<|user|>\nQ1<\\s>\n<|assistant|>\nR1<\\s>\n"
                "<|user|>\nQ2<\\s>\n<|assistant|>\nR2<\\s>\n"
                "<|user|>\nQ3<\\s>\n<|assistant|>\nR3<\\s>\n"
            )
            assert result == expected, f"Expected:\n{repr(expected)}\nGot:\n{repr(result)}"

@pytest.mark.task1
def test_empty_strings_parameters(student_function):
    """Tests behavior with empty string parameters."""
    sample_data = [{"question": "Test", "response": "Response"}]
    
    with patch('xlab.utils.get_beauty_qa_dataset', return_value=sample_data):
        with patch('random.shuffle') as mock_shuffle:
            mock_shuffle.side_effect = lambda x: None
            result = student_function(
                user_string="",
                assistant_string="", 
                end_of_text_string="",
                num_shots=1
            )
            expected = "\nTest\n\nResponse\n"
            assert result == expected, f"Expected:\n{repr(expected)}\nGot:\n{repr(result)}"

@pytest.mark.task1
def test_zero_shots(student_function):
    """Tests behavior when num_shots is 0."""
    sample_data = [{"question": "Test", "response": "Response"}]
    
    with patch('xlab.utils.get_beauty_qa_dataset', return_value=sample_data):
        result = student_function(num_shots=0)
        assert result == "", "Should return empty string when num_shots=0"

@pytest.mark.task1
def test_special_characters_in_strings(student_function):
    """Tests that special characters in tag strings are handled correctly."""
    sample_data = [{"question": "Test", "response": "Response"}]
    
    with patch('xlab.utils.get_beauty_qa_dataset', return_value=sample_data):
        with patch('random.shuffle') as mock_shuffle:
            mock_shuffle.side_effect = lambda x: None
            result = student_function(
                user_string=">>USER<<",
                assistant_string=">>BOT<<",
                end_of_text_string="***END***",
                num_shots=1
            )
            expected = ">>USER<<\nTest***END***\n>>BOT<<\nResponse***END***\n"
            assert result == expected, f"Expected:\n{repr(expected)}\nGot:\n{repr(result)}"

def task1(student_func):
    """Runs all 'task1' tests against the provided student function."""
    target.func = student_func
    result_code = pytest.main([__file__, "-v", "--no-header", "-m", "task1", "-W", "ignore"])
    if result_code == pytest.ExitCode.OK:
        print("✅ All checks passed!")

# ---------------------------------------------------------------------------
# Tests for get_msj_prompt(prompt, num_shots=5)

class TestTargetMSJ:
    func = None

target_msj = TestTargetMSJ()

@pytest.fixture
def student_msj_function():
    """Provides the student's get_msj_prompt function to tests."""
    if target_msj.func is None:
        pytest.skip("Student get_msj_prompt function not provided.")
    return target_msj.func

@pytest.mark.task_msj
def test_msj_prompt_basic_structure(student_msj_function):
    """Ensures formatting and num_shots pass-through to format_qa_dataset."""
    calls = []

    def fake_format_qa_dataset(**kwargs):
        calls.append(kwargs.get('num_shots'))
        return "QA_DATA"

    import sys
    module = sys.modules[student_msj_function.__module__]
    original = getattr(module, 'format_qa_dataset', None)
    try:
        setattr(module, 'format_qa_dataset', fake_format_qa_dataset)
        result = student_msj_function("Hello", num_shots=3)
        assert result == "QA_DATA" + "<|user|>Hello<\\s>\n<|assistant|>\n"
        assert calls == [3]
    finally:
        if original is not None:
            setattr(module, 'format_qa_dataset', original)

@pytest.mark.task_msj
def test_msj_prompt_default_num_shots(student_msj_function):
    """Checks that default num_shots=5 is used when not provided."""
    recorded = {}

    def fake_format_qa_dataset(**kwargs):
        recorded['num_shots'] = kwargs.get('num_shots')
        return ""

    import sys
    module = sys.modules[student_msj_function.__module__]
    original = getattr(module, 'format_qa_dataset', None)
    try:
        setattr(module, 'format_qa_dataset', fake_format_qa_dataset)
        result = student_msj_function("Hi")
        assert recorded.get('num_shots') == 5
        assert result == "<|user|>Hi<\\s>\n<|assistant|>\n"
    finally:
        if original is not None:
            setattr(module, 'format_qa_dataset', original)

def task2(student_func):
    """Runs all 'task_msj' tests against the provided get_msj_prompt function."""
    target_msj.func = student_func
    result_code = pytest.main([__file__, "-v", "--no-header", "-m", "task_msj", "-W", "ignore"])
    if result_code == pytest.ExitCode.OK:
        print("✅ All checks passed!")
