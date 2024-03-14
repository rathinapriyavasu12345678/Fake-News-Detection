import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable


def plot_results():
    matplotlib.use('TkAgg')
    eval1 = np.load('Eval_all_Journ.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']
    Graph_Terms = [0, 3, 4, 7]
    Algorithm = ['TERMS', 'ref_1', 'ref_2', 'ref_', 'ref_4']
    for i in range(eval1.shape[0]):
        value1 = eval1[i, 4, :, 4:]

        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], value1[j, :])
        print('--------------------------------------------------  Comparison',
              '--------------------------------------------------')
        print(Table)


def stats(val):
    v = np.zeros(5)
    v[0] = max(val)
    v[1] = min(val)
    v[2] = np.mean(val)
    v[3] = np.median(val)
    v[4] = np.std(val)
    return v


def plot_results_conv():
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'HHO-OMCNLSTM [36]', 'DHOA-OMCNLSTM [37]', 'SSA-OMCNLSTM [34]', 'ARO-OMCNLSTM [27]',
                 'AARO-OMCNLSTM']
    for i in range(Fitness.shape[0]):
        Terms = ['Worst', 'Best', 'Mean', 'Median', 'Std']
        Conv_Graph = np.zeros((5, 5))
        for j in range(5):  # for 5 algms
            Conv_Graph[j, :] = stats(Fitness[i, j, :])
        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
        print('------------------------------ Statistical Report Dataset', i + 1,
              '------------------------------')
        print(Table)


if __name__ == '__main__':
    plot_results()
    plot_results_conv()
