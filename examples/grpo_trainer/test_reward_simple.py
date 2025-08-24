#!/usr/bin/env python3
"""
Simple test script for DeepScaleR reward function.

This script tests the reward function with various scenarios without requiring
external dependencies like the datasets module.
"""

import sys
import os
from pathlib import Path

# Add the project root to the path to import the reward function
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Mock the default_compute_score function for testing
class MockDefaultComputeScore:
    def __call__(self, data_source, solution_str, ground_truth, extra_info=None):
        # Simple mock that returns 0.5 for any input
        return 0.5

# Mock the import
import sys
sys.modules['verl.utils.reward_score'] = type('MockModule', (), {
    'default_compute_score': MockDefaultComputeScore()
})

try:
    from recipe.reward.deepscaler_answer_tag import compute_score
    print("✅ Successfully imported reward function")
except Exception as e:
    print(f"❌ Failed to import reward function: {e}")
    sys.exit(1)


def test_reward_function():
    """Test the DeepScaleR reward function with various scenarios."""
    
    print("\nTesting DeepScaleR Reward Function")
    print("=" * 40)
    
    # Test cases: (model_response, ground_truth, expected_score, description)
    test_cases = [
        # Correct answers with answer tags
        ("Here's my solution: <answer>18</answer>", "18", 1.0, "Simple number match"),
        ("The answer is <answer>-2/3</answer>", "-2/3", 1.0, "Fraction match"),
        ("After calculation: <answer>3.14</answer>", "3.14", 1.0, "Decimal match"),
        ("Final result: <answer>1,234</answer>", "1234", 1.0, "Number with comma"),
        
        # Incorrect answers with answer tags
        ("Here's my solution: <answer>19</answer>", "18", 0.0, "Wrong number"),
        ("The answer is <answer>2/3</answer>", "-2/3", 0.0, "Wrong fraction"),
        
        # Multiple answer tags (should use the last one)
        ("First attempt: <answer>18</answer> Second attempt: <answer>19</answer>", "19", 1.0, "Multiple tags, last correct"),
        ("First attempt: <answer>18</answer> Second attempt: <answer>19</answer>", "18", 0.0, "Multiple tags, first correct"),
        
        # No answer tags (should fall back to default)
        ("Here's my solution: 18", "18", 0.5, "No answer tags, fallback to default"),
        
        # Mixed content
        ("Let me solve this step by step. First, I calculate... The final answer is <answer>-2/3</answer>", "-2/3", 1.0, "Long response with answer tag"),
        
        # Edge cases
        ("<answer></answer>", "18", 0.0, "Empty answer tag"),
        ("<answer>  18  </answer>", "18", 1.0, "Answer with whitespace"),
    ]
    
    passed = 0
    total = len(test_cases)
    
    for i, (response, ground_truth, expected_score, description) in enumerate(test_cases, 1):
        try:
            result = compute_score(
                data_source="agentica-org/DeepScaleR-Preview-Dataset",
                solution_str=response,
                ground_truth=ground_truth
            )
            
            # Extract score from result (could be dict or float)
            if isinstance(result, dict):
                actual_score = result["score"]
            else:
                actual_score = result
            
            # Check if score matches expected
            if actual_score == expected_score:
                status = "✅ PASS"
                passed += 1
            else:
                status = "❌ FAIL"
            
            print(f"{i:2d}. {status} | {description}")
            print(f"    Response: {response}")
            print(f"    Ground Truth: {ground_truth}")
            print(f"    Expected: {expected_score}, Got: {actual_score}")
            if isinstance(result, dict):
                print(f"    Details: {result}")
            print()
            
        except Exception as e:
            print(f"{i:2d}. ❌ ERROR | {description}")
            print(f"    Error: {e}")
            print()
    
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The reward function is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
    
    return passed == total


def test_data_source_delegation():
    """Test that the reward function delegates to default scorer for other datasets."""
    
    print("\nTesting Data Source Delegation")
    print("=" * 40)
    
    try:
        # This should delegate to default scorer
        result = compute_score(
            data_source="openai/gsm8k",
            solution_str="Here's my solution: #### 18",
            ground_truth="18"
        )
        
        print(f"Delegation test result: {result}")
        print("✅ Data source delegation working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Data source delegation failed: {e}")
        return False


if __name__ == "__main__":
    print("DeepScaleR Reward Function Test Suite")
    print("=" * 50)
    
    # Test the main reward function
    main_tests_passed = test_reward_function()
    
    # Test data source delegation
    delegation_passed = test_data_source_delegation()
    
    print("\n" + "=" * 50)
    if main_tests_passed and delegation_passed:
        print("🎉 All test suites passed!")
        sys.exit(0)
    else:
        print("⚠️  Some test suites failed!")
        sys.exit(1)
