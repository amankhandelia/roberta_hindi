from benchmarks.mlm_custom.test_mlm import MLMTest

mlm_test = MLMTest()


mlm_test.run_full_test(print_debug=False)

mlm_test.run_targeted_test(print_debug=False)


