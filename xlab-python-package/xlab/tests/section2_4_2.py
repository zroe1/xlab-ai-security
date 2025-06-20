"""
Tests for section 2.4.2 of the AI Security course.
"""

# import pprint
import pickle
import torch
import torch.nn.functional as F
import os

def get_100_examples():
    # Get the directory where this file is located
    current_dir = os.path.dirname(__file__)
    data_path = os.path.join(current_dir, 'data', 'cifar10_data.pkl')
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
        x_test_loaded = data['x_test']
        y_test_loaded = data['y_test']

    return x_test_loaded, y_test_loaded

def task1(student_function, model):
    """
    Runs a series of tests for Section 2.4.2, Task 1, prints a structured
    output, and returns a summary of the results.

    When a test fails, it provides detailed feedback on the expected
    versus actual output.

    Args:
        student_function (function): The student's function to test.
        model: The model to test with.

    Returns:
        dict: A summary dictionary with the total number of tests, the
              number passed/failed, and a final score.
    """
    # ANSI color codes
    GREEN = '\033[92m'
    RED = '\033[91m'
    RESET = '\033[0m'
    
    print("Running tests for Section 2.4.2, Task 1...")
    print()

    passed_count = 0
    total_count = 0
    line_width = 70  # Total width for alignment

    # Test 1: Model accuracy test
    total_count += 1
    case_name = "model accuracy > 90%"
    test_description = f"✓ {total_count}. Test case {case_name}"
    
    # Initialize accuracy variable for reuse
    accuracy = None

    device = next(model.parameters()).device
    print(f"Using device: {device} for testing...")
    
    try:
        x, y = get_100_examples()
        x, y = x.to(device), y.to(device)
        logits = model(x)
        predictions = torch.argmax(logits, axis=1)
        correct_predictions = predictions == y
        accuracy = torch.mean(correct_predictions.float())
        
        if accuracy > 0.9:
            status = f"{GREEN}PASSED{RESET}"
            print(f"{test_description:<{line_width-8}} {status}")
            print(f"     Accuracy: {accuracy:.4f}")
            passed_count += 1
        else:
            status = f"{RED}FAILED{RESET}"
            print(f"{test_description:<{line_width-8}} {status}")
            print(f"     Expected: > 0.9000")
            print(f"     Got:      {accuracy:.4f}")
            print(f"     Error: Accuracy is less than 0.9. This indicates the model you passed in is not")
            print(f"            correct. Try loading that model by running `load_model(model_name='Standard',")
            print(f"            threat_model='Linf')`")
    except Exception as e:
        status = f"{RED}FAILED{RESET}"
        print(f"{test_description:<{line_width-8}} {status}")
        print(f"     Error: {e}")

    # Test 2: Student function returns same accuracy
    total_count += 1
    case_name = "student function returns correct accuracy"
    test_description = f"✓ {total_count}. Test case {case_name}"
    
    try:
        if accuracy is not None:  # Only run if first test succeeded in computing accuracy
            student_accuracy = student_function(model, x, y)
            accuracy_diff = abs(float(accuracy) - float(student_accuracy))
            
            if accuracy_diff <= 0.01:
                status = f"{GREEN}PASSED{RESET}"
                print(f"{test_description:<{line_width-8}} {status}")
                print(f"     Expected: {accuracy:.4f} (±0.01)")
                print(f"     Got:      {student_accuracy:.4f}")
                print(f"     Difference: {accuracy_diff:.4f}")
                passed_count += 1
            else:
                status = f"{RED}FAILED{RESET}"
                print(f"{test_description:<{line_width-8}} {status}")
                print(f"     Expected: {accuracy:.4f} (±0.01)")
                print(f"     Got:      {student_accuracy:.4f}")
                print(f"     Difference: {accuracy_diff:.4f} (exceeds 0.01 threshold)")
        else:
            status = f"{RED}SKIPPED{RESET}"
            print(f"{test_description:<{line_width-8}} {status}")
            print(f"     Skipped: Previous test failed to compute accuracy")
    except Exception as e:
        status = f"{RED}FAILED{RESET}"
        print(f"{test_description:<{line_width-8}} {status}")
        print(f"     Error: {e}")

    failed_count = total_count - passed_count
    
    print()
    print("=" * 70)
    if failed_count == 0:
        print(f"🎉 All tests passed! ({passed_count}/{total_count})")
    else:
        print(f"📊 Results: {passed_count} passed, {failed_count} failed out of {total_count} total")
    print("=" * 70)

    # Return a dictionary with the results.
    return {
        "total_tests": total_count,
        "passed": passed_count,
        "failed": failed_count,
        "score": round((passed_count / total_count) * 100) if total_count > 0 else 0
    }


def task2(student_function, model):
    """
    Runs a series of tests for Section 2.4.2, Task 2, prints a structured
    output, and returns a summary of the results.

    When a test fails, it provides detailed feedback on the expected
    versus actual output.

    Args:
        student_function (function): The student's adversarial attack function to test.
        model: The model to test with.

    Returns:
        dict: A summary dictionary with the total number of tests, the
              number passed/failed, and a final score.
    """
    # ANSI color codes
    GREEN = '\033[92m'
    RED = '\033[91m'
    RESET = '\033[0m'
    
    print("Running tests for Section 2.4.2, Task 2...")
    print()

    passed_count = 0
    total_count = 0
    line_width = 70  # Total width for alignment

    device = next(model.parameters()).device
    print(f"Using device: {device} for testing...")

    # Test 1: Model accuracy test
    total_count += 1
    case_name = "model accuracy > 90%"
    test_description = f"✓ {total_count}. Test case {case_name}"
    
    # Initialize accuracy variable for reuse
    accuracy = None
    x, y = None, None

    try:
        x, y = get_100_examples()
        x, y = x.to(device), y.to(device)
        logits = model(x)
        predictions = torch.argmax(logits, axis=1)
        correct_predictions = predictions == y
        accuracy = torch.mean(correct_predictions.float())
        
        if accuracy > 0.9:
            status = f"{GREEN}PASSED{RESET}"
            print(f"{test_description:<{line_width-8}} {status}")
            print(f"     Accuracy: {accuracy:.4f}")
            passed_count += 1
        else:
            status = f"{RED}FAILED{RESET}"
            print(f"{test_description:<{line_width-8}} {status}")
            print(f"     Expected: > 0.9000")
            print(f"     Got:      {accuracy:.4f}")
            print(f"     Error: Accuracy is less than 0.9. This indicates the model you passed in is not")
            print(f"            correct. Try loading that model by running `load_model(model_name='Standard',")
            print(f"            threat_model='Linf')`")
    except Exception as e:
        status = f"{RED}FAILED{RESET}"
        print(f"{test_description:<{line_width-8}} {status}")
        print(f"     Error: {e}")

    # Test 2: Adversarial attack test
    total_count += 1
    case_name = "adversarial attack succeeds"
    test_description = f"✓ {total_count}. Test case {case_name}"
    
    try:
        if accuracy is not None and x is not None and y is not None:  # Only run if first test succeeded
            target_class = 9
            adv_img = student_function(model, x[1:2], torch.tensor([target_class]).to(device), 20, 8/255)
            adv_class = torch.argmax(model(adv_img)).item()
            
            if adv_class == target_class:
                status = f"{GREEN}PASSED{RESET}"
                print(f"{test_description:<{line_width-8}} {status}")
                print(f"     Target class: {target_class}")
                print(f"     Predicted class: {adv_class}")
                print(f"     Adversarial attack successfully fooled the model")
                passed_count += 1
            else:
                status = f"{RED}FAILED{RESET}"
                print(f"{test_description:<{line_width-8}} {status}")
                print(f"     Target class: {target_class}")
                print(f"     Predicted class: {adv_class}")
                print(f"     Error: Adversarial attack failed to fool the model")
        else:
            status = f"{RED}SKIPPED{RESET}"
            print(f"{test_description:<{line_width-8}} {status}")
            print(f"     Skipped: Previous test failed to load model/data")
    except Exception as e:
        status = f"{RED}FAILED{RESET}"
        print(f"{test_description:<{line_width-8}} {status}")
        print(f"     Error: {e}")

    failed_count = total_count - passed_count
    
    print()
    print("=" * 70)
    if failed_count == 0:
        print(f"🎉 All tests passed! ({passed_count}/{total_count})")
    else:
        print(f"📊 Results: {passed_count} passed, {failed_count} failed out of {total_count} total")
    print("=" * 70)

    # Return a dictionary with the results.
    return {
        "total_tests": total_count,
        "passed": passed_count,
        "failed": failed_count,
        "score": round((passed_count / total_count) * 100) if total_count > 0 else 0
    }

def task3(student_function):
    """
    Runs a series of tests for Section 2.4.2, Task 3, prints a structured
    output, and returns a summary of the results.

    Tests the wiggle_ReLU function implementation against the reference solution.

    Args:
        student_function (function): The student's wiggle_ReLU function to test.

    Returns:
        dict: A summary dictionary with the total number of tests, the
              number passed/failed, and a final score.
    """
    # ANSI color codes
    GREEN = '\033[92m'
    RED = '\033[91m'
    RESET = '\033[0m'
    
    print("Running tests for Section 2.4.2, Task 3...")
    print()

    passed_count = 0
    total_count = 0
    line_width = 70  # Total width for alignment

    # Reference implementation
    def reference_wiggle_relu(x, amplitude=0.1, frequency=150):
        return F.relu(x) + amplitude * torch.sin(x * frequency)

    # Test cases: (input_tensor, amplitude, frequency, description)
    test_cases = [
        (torch.tensor([1.0, -1.0, 0.0, 2.5, -0.5]), 0.1, 150, "1D tensor, default params"),
        (torch.tensor([[1.0, -2.0], [0.5, -0.3]]), 0.1, 150, "2D tensor, default params"),
        (torch.tensor([[[1.0, -1.0], [0.0, 2.0]], [[0.5, -0.5], [1.5, -1.5]]]), 0.1, 150, "3D tensor, default params"),
        (torch.tensor([1.0, -1.0, 0.0, 2.5]), 0.05, 150, "1D tensor, amplitude=0.05"),
        (torch.tensor([1.0, -1.0, 0.0, 2.5]), 0.2, 150, "1D tensor, amplitude=0.2"),
        (torch.tensor([1.0, -1.0, 0.0, 2.5]), 0.1, 100, "1D tensor, frequency=100"),
        (torch.tensor([1.0, -1.0, 0.0, 2.5]), 0.1, 200, "1D tensor, frequency=200"),
        (torch.tensor([0.0, 0.0, 0.0]), 0.1, 150, "all zeros"),
        (torch.tensor([-1.0, -2.0, -0.5]), 0.1, 150, "all negative values"),
        (torch.tensor([10.0, 5.0, -3.0, 7.5]), 0.05, 75, "custom amplitude and frequency"),
    ]

    for i, (input_tensor, amplitude, frequency, description) in enumerate(test_cases):
        total_count += 1
        case_name = description
        test_description = f"✓ {total_count}. Test case {case_name}"
        
        try:
            # Get expected output from reference implementation
            expected_output = reference_wiggle_relu(input_tensor, amplitude, frequency)
            
            # Get student's output
            student_output = student_function(input_tensor, amplitude, frequency)
            
            # Compare outputs with tolerance for floating point precision
            tolerance = 1e-6
            if torch.allclose(expected_output, student_output, atol=tolerance):
                status = f"{GREEN}PASSED{RESET}"
                print(f"{test_description:<{line_width-8}} {status}")
                print(f"     Input shape: {list(input_tensor.shape)}")
                print(f"     Amplitude: {amplitude}, Frequency: {frequency}")
                passed_count += 1
            else:
                status = f"{RED}FAILED{RESET}"
                print(f"{test_description:<{line_width-8}} {status}")
                print(f"     Input shape: {list(input_tensor.shape)}")
                print(f"     Amplitude: {amplitude}, Frequency: {frequency}")
                print(f"     Expected (first few values): {expected_output.flatten()[:5].tolist()}")
                print(f"     Got (first few values):      {student_output.flatten()[:5].tolist()}")
                max_diff = torch.max(torch.abs(expected_output - student_output)).item()
                print(f"     Max difference: {max_diff:.8f} (tolerance: {tolerance})")
                
        except Exception as e:
            status = f"{RED}FAILED{RESET}"
            print(f"{test_description:<{line_width-8}} {status}")
            print(f"     Input shape: {list(input_tensor.shape)}")
            print(f"     Amplitude: {amplitude}, Frequency: {frequency}")
            print(f"     Error: {e}")

    failed_count = total_count - passed_count
    
    print()
    print("=" * 70)
    if failed_count == 0:
        print(f"🎉 All tests passed! ({passed_count}/{total_count})")
    else:
        print(f"📊 Results: {passed_count} passed, {failed_count} failed out of {total_count} total")
    print("=" * 70)

    # Return a dictionary with the results.
    return {
        "total_tests": total_count,
        "passed": passed_count,
        "failed": failed_count,
        "score": round((passed_count / total_count) * 100) if total_count > 0 else 0
    }

