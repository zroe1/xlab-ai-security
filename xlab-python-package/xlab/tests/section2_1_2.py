"""
Tests for section 2.1.2 of the AI Security course.
"""


def task4(array1, array2, array3, target):
    """
    Runs a series of tests for Section 2.1.2, Task 4, prints a structured
    output, and returns a summary of the results.

    Tests that the third array has the most occurrences of the target value
    compared to the first two arrays.

    Args:
        array1 (list): First array to compare.
        array2 (list): Second array to compare.
        array3 (list): Third array to compare.
        target (int): The target value to count in each array.

    Returns:
        dict: A summary dictionary with the total number of tests, the
              number passed/failed, and a final score.
    """
    # ANSI color codes
    GREEN = '\033[92m'
    RED = '\033[91m'
    RESET = '\033[0m'
    
    print("Running tests for Section 2.1.2, Task 4...")
    print()

    passed_count = 0
    total_count = 0
    line_width = 70  # Total width for alignment

    # Test 1: Arrays have equal length
    total_count += 1
    case_name = "arrays have equal length"
    test_description = f"✓ {total_count}. Test case {case_name}"
    
    try:
        if len(array1) == len(array2) == len(array3):
            status = f"{GREEN}PASSED{RESET}"
            print(f"{test_description:<{line_width-8}} {status}")
            print(f"     Array lengths: {len(array1)}, {len(array2)}, {len(array3)}")
            passed_count += 1
        else:
            status = f"{RED}FAILED{RESET}"
            print(f"{test_description:<{line_width-8}} {status}")
            print(f"     Expected: All arrays to have equal length")
            print(f"     Got:      Array lengths: {len(array1)}, {len(array2)}, {len(array3)}")
            print(f"     Error: Arrays must have equal length for valid comparison")
    except Exception as e:
        status = f"{RED}FAILED{RESET}"
        print(f"{test_description:<{line_width-8}} {status}")
        print(f"     Error: {e}")

    # Test 2: Third array has the most occurrences
    total_count += 1
    case_name = "third array has most occurrences"
    test_description = f"✓ {total_count}. Test case {case_name}"
    
    try:
        count1 = array1.count(target) if hasattr(array1, 'count') else sum(1 for x in array1 if x == target)
        count2 = array2.count(target) if hasattr(array2, 'count') else sum(1 for x in array2 if x == target)
        count3 = array3.count(target) if hasattr(array3, 'count') else sum(1 for x in array3 if x == target)
        
        if count3 > count1 and count3 > count2:
            status = f"{GREEN}PASSED{RESET}"
            print(f"{test_description:<{line_width-8}} {status}")
            print(f"     Target value: {target}")
            print(f"     Array1 count: {count1}")
            print(f"     Array2 count: {count2}")
            print(f"     Array3 count: {count3} (highest)")
            passed_count += 1
        else:
            status = f"{RED}FAILED{RESET}"
            print(f"{test_description:<{line_width-8}} {status}")
            print(f"     Target value: {target}")
            print(f"     Array1 count: {count1}")
            print(f"     Array2 count: {count2}")
            print(f"     Array3 count: {count3}")
            print(f"     Error: Third array does not have the most occurrences of target {target}")
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
