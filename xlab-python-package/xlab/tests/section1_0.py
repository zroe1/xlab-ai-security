"""
Tests for section 1.0 of the AI Security course.
"""

def task1(student_function):
    """
    This is a sample test function for Section 1.0, Task 1.

    A student would pass their function to this task to get a score.

    Args:
        student_function (function): The student's function to test.

    Returns:
        dict: A dictionary containing the test result and score.
    """
    # Here you would implement the test logic.
    # For example, you could run the student's function with some inputs
    # and check the output.
    
    print("Running tests for Section 1.0, Task 1...")
    
    try:
        # Example test case
        output = student_function("test input")
        if output == "expected output":
            print("Test passed!")
            return {"passed": True, "score": 100}
        else:
            print("Test failed: Unexpected output.")
            return {"passed": False, "score": 0}
    except Exception as e:
        print(f"Test failed with an error: {e}")
        return {"passed": False, "score": 0} 