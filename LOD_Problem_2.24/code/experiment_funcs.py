import numpy as np
from numpy.random import default_rng
import warnings

# ############## DEFINE EXPERIMENT ############### DEFINE EXPERIMENT ############### DEFINE EXPERIMENT ############## #
# Seed a rng so all test sets have the same sequence, for result comparability.
test_set_rng = default_rng(0)

# Configure exception handling for testing
overflow_err_state = "raise"
max_attempts = 15


# This is a top-level experiment function.  Do a number of experiments of varying trial counts
def doExperimentSetTrials(trial_count_set, test_size):
    sz_list = []
    hyp_list = []
    stat_list = []
    for trial_count in trial_count_set:
        test_data = test_set_rng.uniform(-1, 1, test_size)

        for attempt in range(0, max_attempts):
            try:
                hypothesis_set = findHypothesisSet(trial_count)
                experiment_results = calcExperimentOutputs(test_data, hypothesis_set)
            except FloatingPointError:
                if attempt == max_attempts - 1:
                    print("Attempt ", attempt + 1, " failed, aborting trial count: ", trial_count)
                    break
                else:
                    print("Caught in doExperimentSetTrials, retrying with attempt:", attempt + 2, " of ", max_attempts)
            else:
                exp_size = np.array([trial_count, test_size])

                sz_list.append(exp_size)
                hyp_list.append(experiment_results[0])
                stat_list.append(experiment_results[1])
                break

    return np.array(sz_list), np.array(hyp_list), np.array(stat_list)


# This is a top-level experiment function.  Do a number of experiments of varying test data sizes
def doExperimentSetTests(trial_count, test_sizes_set):
    sz_list = []
    hyp_list = []
    stat_list =[]
    hypothesis_set = findHypothesisSet(trial_count)
    for test_size in test_sizes_set:
        for attempt in range(0, max_attempts):
            try:
                test_data = test_set_rng.uniform(-1, 1, test_size)
                experiment_results = calcExperimentOutputs(test_data, hypothesis_set)
            except FloatingPointError:
                if attempt == max_attempts - 1:
                    print("Attempt ", attempt + 1, " failed, aborting test size: ", test_size)
                    break
                else:
                    print("Caught in deExperimentSetTests, retrying with attempt:", attempt + 2, " of ", max_attempts)
            else:
                exp_size = np.array([trial_count, test_size])

                sz_list.append(exp_size)
                hyp_list.append(experiment_results[0])
                stat_list.append(experiment_results[1])
                break
    return np.array(sz_list), np.array(hyp_list), np.array(stat_list)


# Given a number of trials t, generate t training-sets and find the hypothesis for each one
def findHypothesisSet(trials):
    hypothesis_set = np.empty((trials, 2))
    for i in range(0, trials - 1):
        data_set = getDataSet()
        hypothesis_set[i] = getHypothesis(data_set)

    return hypothesis_set


# Given a list of hypotheses, get the average function, bias, var, and two versions of E[E_out]
def calcExperimentOutputs(x_values, hypothesis_set):
    averageHypothesis = findExpectedHypothesis(hypothesis_set)
    try:
        bias = getBias(x_values, averageHypothesis)
        var = getVar(x_values, hypothesis_set, averageHypothesis)
        EEout = getEEout(x_values, hypothesis_set)
    except FloatingPointError:
        print("Caught exception in calcExperimentOutputs")
        raise FloatingPointError

    return averageHypothesis, np.array([bias, var, EEout])


# ############ GET EXPERIMENT INPUT ############ GET EXPERIMENT INPUT ############# GET EXPERIMENT INPUT ############ #


# Get data set
def getDataSet():
    x_one = np.random.uniform(-1, 1)
    x_two = np.random.uniform(-1, 1)
    return np.array([[x_one, x_one ** 2], [x_two, x_two ** 2]])


# ############## CALCULATION FUNCS ############### CALCULATION FUNCS ############### CALCULATION FUNCS ############## #


# Get the hypothesis:
def getHypothesis(data_set):
    return data_set[1][0] + data_set[0][0], -(data_set[0][0]*data_set[1][0])


# Return the average function, the mean of many hypotheses
def findExpectedHypothesis(manyTrialsResults):
    return np.mean(manyTrialsResults, 0)


# Evaluate a g(x) = ax + b at x
def evaluateLineAtX(x, coefficients):
    return coefficients[0] * x + coefficients[1]


# Take the bias over a new data set
def getBias(x_values, averageFunction):
    evaluation = evaluateLineAtX(x_values, averageFunction)
    with np.errstate(over=overflow_err_state):
        try:
            deviation = evaluation - np.square(x_values)
            bias = np.mean(np.square(deviation))
        except FloatingPointError:
            print("Catching overflow in getBias")
            raise FloatingPointError
    return bias


# Find the sample variance over test data
def getVar(x_values, hypothesis_set, averageFunction):
    num_hyps = len(hypothesis_set)
    deviation = hypothesis_set - averageFunction
    error_matrix = np.tensordot(deviation[:, 0], x_values, 0) + deviation[:, 1:]
    with np.errstate(over=overflow_err_state):
        try:
            error_matrix = np.square(error_matrix)
        except FloatingPointError:
            print("Catching overflow in getVar")
    mean_over_x = np.mean(error_matrix, 1)
    return np.sum(mean_over_x) / (num_hyps - 1)


# E[E_out] as mean approximation of expectation over data sets and test data
def getEEout(x_values, hypothesis_set):
    evaluation = np.tensordot(hypothesis_set[:, 0], x_values, 0) + hypothesis_set[:, 1:]
    with np.errstate(over=overflow_err_state):
        try:
            deviation = evaluation - np.square(x_values)
            error_matrix = np.square(deviation)
        except FloatingPointError:
            print("Catching overflow in getEEout")
    return np.mean(error_matrix)
