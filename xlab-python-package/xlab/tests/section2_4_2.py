"""
Tests for section 2.4.2 of the AI Security course.
"""

# import pprint
import pickle
import torch
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
    
    try:
        x, y = get_100_examples()
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

    # TODO: Add more test cases here as they are developed
    
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