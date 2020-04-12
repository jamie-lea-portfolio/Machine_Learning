import matplotlib.pyplot as plt
import numpy as np
import experiment_funcs as expf


def resultPrinterGeneral(result_row, filename):

    num_trials = result_row[0][0]
    test_size = result_row[0][1]
    avg_func = getLineEquationAsString(result_row[1])
    bias = result_row[2][0]
    var = result_row[2][1]
    eeout = result_row[2][2]

    print("# of Trials: ", num_trials, "; test data set size: ", test_size, file=filename)
    print("-------------------------------------------------------------", file=filename)
    print("g_hat(x)  = ", avg_func, file=filename)
    print("bias      = ", bias, file=filename)
    print("var       = ", var, file=filename)
    print("E[E_out]  = ", eeout, file=filename)
    print("", file=filename)
    print("bias + var = ", bias + var, file=filename)
    print("E[E_out] - (bias + var) = ", eeout - bias - var, file=filename)
    print("", file=filename)


def printLineEquation(coefficients, filename):
    print(coefficients[0], "x + ", coefficients[1], file=filename)


def getLineEquationAsString(coefficients):
    return "%sx + %s" % tuple(coefficients)


output_prefix = "results/final/"


def plotQuestionAnswer(results, title):
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-0.25, 1)
    target_x_range = np.linspace(-1, 1)
    target_y = np.square(target_x_range)
    ax.plot(target_x_range, target_y, c='k')

    num_hyps = len(results[1])
    alpha_bias = 0.75
    hyp_alpha = alpha_bias/num_hyps
    hyp_col_bias = 1
    hyp_col_shift = hyp_col_bias/num_hyps
    for hyp in results[1]:
        hyp_y = expf.evaluateLineAtX(target_x_range, hyp)
        hyp_alpha = 0.25
        ax.plot(target_x_range, hyp_y, alpha=hyp_alpha, c=(1 - hyp_col_shift, 0, hyp_col_shift))
        if hyp_alpha >= 1:
            hyp_alpha = 1
        else:
            hyp_alpha += alpha_bias/num_hyps
        if hyp_col_shift >= 1:
            hyp_col_shift = 1
        else:
            hyp_col_shift += alpha_bias/num_hyps

    fig.show()
    fig.savefig(output_prefix + title + '.png', transparent=False, dpi=80, bbox_inches="tight")
    plt.close(fig)


def initPlot_trials(sup_title, x_label, y_label, x_ticks, y_ticks, x_scale, y_scale):
    trials_fig, trials_ax = plt.subplots()
    trials_fig.set_size_inches(6, 3)

    super_title = sup_title
    trials_ax.set_title(sup_title)

    trials_ax.set_xlabel(x_label)
    trials_ax.set_ylabel(y_label)

    trials_ax.set_xscale(x_scale)
    trials_ax.set_yscale(y_scale)

    return trials_fig, trials_ax


def trials_plotter(results):

    # print(results)
    title = str(results[0][0][1]) + " test data points"
    file_title = str(results[0][0][1]) + "_data_points"
    x_trials_range = results[0][:, 0]
    y_bias_range = results[2][:, 0]
    y_var_range = results[2][:, 1]
    y_ticks = np.concatenate((results[2][:, 0], results[2][:, 1]))
    fig_bias_var, ax_bias_var = initPlot_trials(title, "trials", "bias (dashed), var (dotted)", x_trials_range, y_ticks, "log", "linear")

    ax_bias_var.plot(x_trials_range, y_bias_range, ls='dashed', c='b')
    ax_bias_var.plot(x_trials_range, y_var_range, ls='dotted', c='r')

    fig_bias_var.show()
    fig_bias_var.savefig(output_prefix + file_title + '_bias-var' + '.png', transparent=False, dpi=80, bbox_inches="tight")
    plt.close(fig_bias_var)

    # #########

    y_bias_var_range = y_bias_range + y_var_range
    y_eeout_range = results[2][:, 2]

    y_diff_range = y_eeout_range - y_bias_var_range
    fig_bias_var_diff, ax_bias_var_diff = initPlot_trials(title, "trials", "E[E_out] - bias + var ", x_trials_range, y_diff_range, "log", "linear")

    ax_bias_var_diff.plot(x_trials_range, y_diff_range, c='k')

    fig_bias_var_diff.show()
    fig_bias_var_diff.savefig(output_prefix + file_title + '_diff' + '.png', transparent=False, dpi=80, bbox_inches="tight")
    plt.close(fig_bias_var_diff)


def tests_plotter(results):

    # print(results)
    title = str(results[0][0][0]) + " trials"
    file_title = str(results[0][0][0]) + "_trials"
    x_trials_range = results[0][:, 1]
    y_bias_range = results[2][:, 0]
    y_var_range = results[2][:, 1]
    y_ticks = np.concatenate((results[2][:, 0], results[2][:, 1]))
    fig_bias_var, ax_bias_var = initPlot_trials(title, "test data points", "bias (dashed), var (dotted)", x_trials_range, y_ticks, "log", "linear")

    # ax_bias_var.set_yscale("symlog", linthresh=.0000000000000001)
    ax_bias_var.plot(x_trials_range, y_bias_range, ls='dashed', c='b')
    ax_bias_var.plot(x_trials_range, y_var_range, ls='dotted', c='r')
    fig_bias_var.show()
    fig_bias_var.savefig(output_prefix + file_title + '_bias-var' + '.png', transparent=False, dpi=80, bbox_inches="tight")
    plt.close(fig_bias_var)

    # #########

    y_bias_var_range = y_bias_range + y_var_range
    y_eeout_range = results[2][:, 2]

    y_diff_range = y_eeout_range - y_bias_var_range
    fig_bias_var_diff, ax_bias_var_diff = initPlot_trials(title, "test data points", "E[E_out] - bias + var ", x_trials_range, y_diff_range, "log", "linear")

    ax_bias_var_diff.plot(x_trials_range, y_diff_range, c='k')

    fig_bias_var_diff.show()
    fig_bias_var_diff.savefig(output_prefix + file_title + '_diff' + '.png', transparent=False, dpi=80, bbox_inches="tight")
    plt.close(fig_bias_var_diff)

