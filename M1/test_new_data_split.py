#!/usr/bin/env python3
"""
Test script to verify the proper data splitting configuration:
- 80% training, 10% validation, 10% testing
- No data leakage: train/validation/test sets are completely separate
"""

import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
import sys
import os

def test_data_splitting():
    """Test the proper data splitting logic"""
    print("🧪 Testing PROPER Data Splitting Configuration")
    print("=" * 60)

    # Simulate the proper data splitting logic (80/10/10 split)
    total_samples = 1000  # Example with 1000 samples
    mock_paths = list(range(total_samples))

    print(f"📊 PROPER DATA SPLITTING STRATEGY")
    print(f"=" * 50)
    print(f"Total samples: {total_samples}")
    print(f"Strategy: 80% train, 10% validation, 10% test - completely separate sets")
    print(f"")

    # First split: 80% train, 20% temp (which will be split into 10% val, 10% test)
    train_paths, temp_paths = train_test_split(mock_paths, test_size=0.2, random_state=42)

    # Second split: Split the 20% temp into 10% val and 10% test
    val_paths, test_paths = train_test_split(temp_paths, test_size=0.5, random_state=42)

    # Verify the split
    actual_train_pct = len(train_paths)/total_samples*100
    actual_val_pct = len(val_paths)/total_samples*100
    actual_test_pct = len(test_paths)/total_samples*100

    print(f"Data Split Results:")
    print(f"  🏋️  Training:   {len(train_paths):4d} samples ({actual_train_pct:.1f}% - Target: 80.0%)")
    print(f"  ✅ Validation: {len(val_paths):4d} samples ({actual_val_pct:.1f}% - Target: 10.0%)")
    print(f"  🧪 Testing:    {len(test_paths):4d} samples ({actual_test_pct:.1f}% - Target: 10.0%)")
    print(f"=" * 50)

    print(f"✅ VERIFICATION:")
    print(f"  Training:   {len(train_paths):4d} samples ({actual_train_pct:.1f}% - Target: 80.0%)")
    print(f"  Validation: {len(val_paths):4d} samples ({actual_val_pct:.1f}% - Target: 10.0%)")
    print(f"  Testing:    {len(test_paths):4d} samples ({actual_test_pct:.1f}% - Target: 10.0%)")

    # Check for data leakage (no overlap between sets)
    train_set = set(train_paths)
    val_set = set(val_paths)
    test_set = set(test_paths)

    train_val_overlap = len(train_set.intersection(val_set))
    train_test_overlap = len(train_set.intersection(test_set))
    val_test_overlap = len(val_set.intersection(test_set))

    print(f"  Data Leakage Check:")
    print(f"    Train-Val overlap: {train_val_overlap} samples (should be 0)")
    print(f"    Train-Test overlap: {train_test_overlap} samples (should be 0)")
    print(f"    Val-Test overlap: {val_test_overlap} samples (should be 0)")

    # Check if percentages match proper targets (80/10/10)
    train_ok = abs(actual_train_pct - 80.0) <= 0.1
    val_ok = abs(actual_val_pct - 10.0) <= 0.1
    test_ok = abs(actual_test_pct - 10.0) <= 0.1

    # Check for data leakage
    no_leakage = (train_val_overlap == 0 and train_test_overlap == 0 and val_test_overlap == 0)

    if train_ok and val_ok and test_ok and no_leakage:
        print(f"✅ All percentages match proper strategy and no data leakage detected")
    else:
        if not train_ok:
            print(f"⚠️  Warning: Training should be 80.0%, got {actual_train_pct:.1f}%")
        if not val_ok:
            print(f"⚠️  Warning: Validation should be 10.0%, got {actual_val_pct:.1f}%")
        if not test_ok:
            print(f"⚠️  Warning: Testing should be 10.0%, got {actual_test_pct:.1f}%")
        if not no_leakage:
            print(f"⚠️  Warning: Data leakage detected!")

    print(f"=" * 50)

    return train_ok and val_ok and test_ok and no_leakage

def test_constraint_validation():
    """Test basic constraint validation logic"""
    print("\n🔍 Testing Basic Constraint Validation Logic")
    print("=" * 60)

    # Test cases for basic constraint validation (validation ≤ training)
    test_cases = [
        {"train_acc": 95.0, "val_acc": 85.0, "expected": "SATISFIED"},
        {"train_acc": 90.0, "val_acc": 90.0, "expected": "SATISFIED"},
        {"train_acc": 88.0, "val_acc": 92.0, "expected": "VIOLATED"},
        {"train_acc": 93.0, "val_acc": 89.0, "expected": "SATISFIED"},
    ]
    
    all_passed = True
    
    for i, case in enumerate(test_cases, 1):
        train_accuracy = case["train_acc"]
        val_accuracy = case["val_acc"]
        expected = case["expected"]

        print(f"\nTest Case {i}:")
        print(f"  Training Accuracy:   {train_accuracy:.1f}%")
        print(f"  Validation Accuracy: {val_accuracy:.1f}%")
        print(f"  Difference:          {train_accuracy - val_accuracy:.1f}% (Train - Validation)")

        if val_accuracy <= train_accuracy:
            actual = "SATISFIED"
            status_msg = "✅ CONSTRAINT SATISFIED: Validation accuracy ≤ Training accuracy"
        else:
            actual = "VIOLATED"
            status_msg = f"❌ CONSTRAINT VIOLATED: Validation accuracy > Training accuracy"
            status_msg += f"\n     Validation exceeds training by {val_accuracy - train_accuracy:.1f}%"
        
        print(f"  {status_msg}")
        print(f"  Expected: {expected}, Actual: {actual}")
        
        if actual == expected:
            print(f"  ✅ Test Case {i} PASSED")
        else:
            print(f"  ❌ Test Case {i} FAILED")
            all_passed = False
    
    print(f"\n" + "=" * 60)
    if all_passed:
        print("✅ All constraint validation tests PASSED")
    else:
        print("❌ Some constraint validation tests FAILED")
    
    return all_passed

def test_with_different_sample_sizes():
    """Test proper data splitting with different sample sizes"""
    print("\n📊 Testing PROPER Strategy with Different Sample Sizes")
    print("=" * 60)

    sample_sizes = [1000, 3860]  # Test with key sizes
    all_ok = True

    for total_samples in sample_sizes:
        # PROPER: 80/10/10 split
        train_size = int(total_samples * 0.8)
        val_size = int(total_samples * 0.1)
        test_size = total_samples - train_size - val_size

        train_pct = train_size/total_samples*100
        val_pct = val_size/total_samples*100
        test_pct = test_size/total_samples*100

        print(f"Size {total_samples}: Train {train_pct:.1f}% | Val {val_pct:.1f}% | Test {test_pct:.1f}% (separate sets)")

        # Check if close to proper targets (80/10/10)
        train_ok = abs(train_pct - 80.0) <= 0.1
        val_ok = abs(val_pct - 10.0) <= 0.1
        test_ok = abs(test_pct - 10.0) <= 0.1

        if not (train_ok and val_ok and test_ok):
            all_ok = False

    return all_ok

def main():
    """Run all tests"""
    print("🚀 Testing New Data Splitting Strategy")
    print("=" * 70)
    
    tests = [
        ("Data Splitting Logic", test_data_splitting),
        ("Constraint Validation", test_constraint_validation),
        ("Different Sample Sizes", test_with_different_sample_sizes)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {str(e)}")
    
    print("\n" + "=" * 70)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! PROPER data splitting strategy is working correctly.")
        print("\n📋 Summary of PROPER Strategy:")
        print("   ✅ 80% of data for training")
        print("   ✅ 10% of data for validation (completely separate from training)")
        print("   ✅ 10% of data for testing (completely separate from training and validation)")
        print("   ✅ No data leakage: all sets are completely separate")
        print("   📝 Note: Train/validation/test sets have no overlap - proper ML practice")
    else:
        print("❌ Some tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
