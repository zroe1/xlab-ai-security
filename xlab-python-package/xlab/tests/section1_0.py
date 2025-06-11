"""
Tests for section 1.0 of the AI Security course.
"""

import pprint

def task1(student_function):
    """
    Runs a series of tests for Section 1.0, Task 1, prints a structured
    output, and returns a summary of the results.

    When a test fails, it provides detailed feedback on the expected
    versus actual output.

    Args:
        student_function (function): The student's function to test.

    Returns:
        dict: A summary dictionary with the total number of tests, the
              number passed/failed, and a final score.
    """
    # ANSI color codes
    GREEN = '\033[92m'
    RED = '\033[91m'
    RESET = '\033[0m'
    
    def pretty_format_array(arr, indent="\t\t"):
        """Format an array with proper line wrapping."""
        if not arr:
            return "[]"
        
        # Use pprint to format with a reasonable width
        formatted = pprint.pformat(arr, width=50, compact=True)
        
        # Add proper indentation to each line
        lines = formatted.split('\n')
        if len(lines) == 1:
            return formatted
        else:
            # Multi-line: indent all lines after the first
            result = [lines[0]]
            for line in lines[1:]:
                result.append(indent + line)
            return '\n'.join(result)
    
    print("Running tests for Section 1.0, Task 1...")
    print()

    test_cases = [
        {"input": "Hello there gpt-2", "expected_output": ['Hello', ' there', ' g', 'pt', '-', '2']},
        {"input": "??!hello--*- world#$", "expected_output": ['??', '!', 'hello', '--', '*', '-', ' world', '#$']},
        {"input": "https://xrisk.uchicago.edu/fellowship/", "expected_output": ['https', '://', 'x', 'risk', '.', 'uch', 'icago', '.', 'edu', '/', 'fell', 'owship', '/']},
        {"input": "", "expected_output": []},
        {"input": ".,.,.,.,.,.,.,", "expected_output": ['.,', '.,', '.,', '.,', '.,', '.,', '.,']},
    ]

    passed_count = 0
    total_count = 0
    line_width = 70  # Total width for alignment

    # Test 1: Function should run without crashing.
    total_count += 1
    case_name = "function runs without crashing"
    test_description = f"✓ {total_count}. Test case {case_name}"
    
    try:
        student_function("test input")
        status = f"{GREEN}PASSED{RESET}"
        print(f"{test_description:<{line_width-8}} {status}")
        passed_count += 1
    except Exception as e:
        status = f"{RED}FAILED{RESET}"
        print(f"{test_description:<{line_width-8}} {status}")
        print(f"     Error: {e}")

    # Run the rest of the functional test cases
    for case in test_cases:
        total_count += 1
        input_val = case["input"]
        expected_val = case["expected_output"]
        
        # Format the input display nicely
        if input_val == "":
            display_input = "''" 
        else:
            display_input = f"'{input_val}'"
        
        test_description = f"✓ {total_count}. Test case {display_input}"
        
        try:
            output = student_function(input_val)
            if output == expected_val:
                status = f"{GREEN}PASSED{RESET}"
                print(f"{test_description:<{line_width-8}} {status}")
                passed_count += 1
            else:
                # Provide structured output for failures with pretty printing
                status = f"{RED}FAILED{RESET}"
                print(f"{test_description:<{line_width-8}} {status}")
                print(f"     Expected: {pretty_format_array(expected_val)}")
                print(f"     Got:      {pretty_format_array(output)}")
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