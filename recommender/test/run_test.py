import argparse
import test_classes
parser = argparse.ArgumentParser()
parser.add_argument("-t",
                    "--test",
                    type=str,
                    default='all',
                    help="""different types of tests.
                    \nArgs: basic,  mani (manipulation),  opt (optimization),
                    all. (default=all)""")
args = parser.parse_args()


if args.test == 'basic':
    test_classes.run_test(test_classes.TestBasic,
                          "\nRunning all basic tests...\n")
elif args.test == 'mani':
    test_classes.run_test(test_classes.TestdfManipulation,
                          "\nRunning all dataframe manipulation tests...\n")
elif args.test == 'opt':
    test_classes.run_test(test_classes.TestOptimization,
                          "\nRunning all optimization tests...\n")
elif args.test == 'all':
    test_classes.run_test(test_classes.TestBasic,
                          "\nRunning all basic tests...\n")
    test_classes.run_test(test_classes.TestdfManipulation,
                          "\nRunning all dataframe manipulation tests...\n")
    test_classes.run_test(test_classes.TestOptimization,
                          "\nRunning all optimization tests...\n")

else:
    print("Wrong parameter passed to the test option.")
    print("The only valid parameters are:\nbasic\nmani\nopt\nall")
