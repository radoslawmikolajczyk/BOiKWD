import importlib
import os
test_modules = ['example_01_solvable', 'example_02_solvable', 'example_03_unbounded', 'example_04_solvable_artificial_vars', 'example_05_unfeasible', 'example_06_dual', 'example_07_cost_sensitivity']
test_dir = 'tests.simplex'
print("Running tests...")
success = True
for test_module in test_modules:
    test = importlib.import_module(f"{test_dir}.{test_module}")
    try:
        test.run()
        print(f'- test "{test_module}":\t PASSED')
    except Exception as e:
        success = False
        print(f'- test "{test_module}":\t FAILED (message: {e})')

if success:
    print("Congratulations, your linear programming tools seem to work correctly!")
else:
    print("Some of the tests failed. Fix your implementation ASAP :)")

    