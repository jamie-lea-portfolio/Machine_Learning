import experiment_funcs as expf
import experiment_printer as expp
import numpy as np
import matplotlib.pyplot as plt


# ############## DO THE EXPERIMENT ############### DO THE EXPERIMENT ############### DO THE EXPERIMENT ############## #
seq_tiny = (2, 3, 4, 5)
seq_small = (5, 10, 25, 50, 100)
seq_medium = (10, 25, 75, 150, 500)
seq_big = (10, 15, 20, 25, 50, 75, 100, 150, 175, 200, 250, 500, 1000)
seq_massive = (10, 100, 1000, 10000, 100000, 1000000, 10000000)


# Final sequence to use for both varying trial count, and varying test size
final_trial_count_seq = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450, 500, 1000, 1250, 1500, 1750, 2000, 2500, 10000, 100000)
final_test_count_seq = (5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 175, 200, 225, 250, 275, 300, 325, 350, 500, 750, 1000, 2500, 5000)

# Alias this so as to not change following code during program debug/testing.
alias_trial_count_seq = final_trial_count_seq
alias_test_count_seq = final_test_count_seq


# Generate results for reports, varying number of trials on fixed test size
def metaExperiment_trials(trial_count_set, test_size, output_file):
    print("======== DOING TRIALS (", trial_count_set[0], ", ..., ", trial_count_set[-1],
          ") with ", test_size, " test points ========", file=output_file)
    results = expf.doExperimentSetTrials(trial_count_set, test_size)

    for i in range(0, len(results[0])):
        result_row = results[0][i], results[1][i], results[2][i]
        expp.resultPrinterGeneral(result_row, output_file)

    expp.trials_plotter(results)
    expp.plotQuestionAnswer(results, str(results[0][0][1]) + " test data points")


# Generate results for reports, varying number of tests on fixed trial size
def metaExperiment_tests(trial_count, test_data_sizes, output_file):
    print("======== DOING TESTS (", test_data_sizes[0], ", ..., ", test_data_sizes[-1],
          ") with ", trial_count, " trial size ========", file=output_file)
    results = expf.doExperimentSetTests(trial_count, test_data_sizes)

    for i in range(0, len(results[0])):
        result_row = results[0][i], results[1][i], results[2][i]
        expp.resultPrinterGeneral(result_row, output_file)

    expp.tests_plotter(results)
    expp.plotQuestionAnswer(results, str(results[0][0][0]) + " trials")


# ############## DO THE EXPERIMENT ############### DO THE EXPERIMENT ############### DO THE EXPERIMENT ############## #
# In every innermost function call, test data is generated by a seeded generator, so the sequence is fixed
# In every innermost function call, training data is generated by an un-seeded generator, so the sequence is different

# In this section metaExperiment_trials increments trial count
output_prefix = "results/final/"
print("Doing 2500 test data")
results_2500_test = open(output_prefix + "results_test_points_2500.dat", 'w')
metaExperiment_trials(alias_trial_count_seq, 2500, results_2500_test)
results_2500_test.close()
# # # ################
print("Doing 5000 test")
results_5000_test = open(output_prefix + "results_test_points_5000.dat", 'w')
metaExperiment_trials(alias_trial_count_seq, 5000, results_5000_test)
results_5000_test.close()
# # # ################
print("Doing 10000 test data")
results_10000_test = open(output_prefix + "results_test_points_10000.dat", 'w')
metaExperiment_trials(alias_trial_count_seq, 10000, results_10000_test)
results_10000_test.close()
# # # ################
# # # ################

# # In this section metaExperiment_tests increments test data size
print("Doing 25 trials")
results_25_trials = open(output_prefix + "results_trials_25.dat", 'w')
metaExperiment_tests(25, alias_trial_count_seq, results_25_trials)
results_25_trials.close()
# # ################
print("Doing 50 trials")
results_50_trials = open(output_prefix + "results_trials_50.dat", 'w')
metaExperiment_tests(50, alias_trial_count_seq, results_50_trials)
results_50_trials.close()
# # # ################
print("Doing 100 trials")
results_100_trials = open(output_prefix + "results_trials_100.dat", 'w')
metaExperiment_tests(100, alias_trial_count_seq, results_100_trials)
results_100_trials.close()
# # # ################
single_name = "results/final/single_exp_5000_5000.dat"
single_file = open(single_name, "w")
metaExperiment_trials(np.array([5000]), np.array([5000]), single_file)
single_file.close()

# ##########################################
print("All done!")
