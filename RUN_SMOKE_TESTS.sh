#!/bin/bash
# Run smoke tests in the conda environment

echo "Running Phase A smoke tests for axial transformer..."
echo ""

# Activate conda environment and run tests
conda activate marl_stable

echo "Test 1: Shape test (bcrbc_axial_shape.py)"
python scripts/smoke_tests/bcrbc_axial_shape.py
TEST1_RESULT=$?

echo ""
echo "Test 2: Fake train test (bcrbc_fake_train.py)"
python scripts/smoke_tests/bcrbc_fake_train.py
TEST2_RESULT=$?

echo ""
echo "========================================"
if [ $TEST1_RESULT -eq 0 ] && [ $TEST2_RESULT -eq 0 ]; then
    echo "All Phase A smoke tests PASSED"
    exit 0
else
    echo "Some tests FAILED"
    exit 1
fi
