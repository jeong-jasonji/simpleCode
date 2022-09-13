import os
import shap
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy import interp
from itertools import cycle
from random import randint, sample

import matplotlib.pyplot as plt


def extract_labels_probabilities(output):
    """
    calculates the one-hot encoded labels from the csv prediction outputs
    """
    # consolidate labels
    numerical_labels = []
    onehot_labels = []
    probs = []
    num_cls = len(output.true.unique())
    for i in range(len(output)):
        # get the true label
        label = output['true'].iloc[i]
        numerical_labels.append(label)
        # make a onehot_label label
        onehot_label = [0 for i in range(num_cls)]
        # update onehot label
        onehot_label[label] = 1
        onehot_labels.append(onehot_label)
        # make a prob list
        prob_list = [output['score_{}'.format(j)].iloc[i] for j in range(num_cls)]
        probs.append(prob_list)
    onehot_labels = np.array(onehot_labels)
    probs = np.array(probs)
    numerical_labels = np.array(numerical_labels)
    return numerical_labels, onehot_labels, probs

def calculate_optimal_thresholds(tpr, fpr, thresholds, labels, probs):
    """
    calculates various optimal thresholds:
        gemans: optimizing gmeans value
        Yoden: Yoden's J statistic
        precision/recall: optimizing f1 score
    """
    # calculate the g-mean for each threshold
    gmeans = np.sqrt(tpr * (1 - fpr))
    # locate the index of the largest g-mean
    ix_g = np.argmax(gmeans)
    gmeans_thresh = thresholds[ix_g]
    # get the Yoden J stat
    J = tpr - fpr
    ix_y = np.argmax(J)
    yoden_thresh = thresholds[ix_y]
    # get the best threshold with precision recall curve
    precision, recall, pr_thresh = precision_recall_curve(labels, probs)
    fscore = (2 * precision * recall) / (precision + recall)
    # locate the index of the largest f score
    ix_pr = np.argmax(fscore)
    precision_recall_thresh = pr_thresh[ix_pr]

    return gmeans_thresh, yoden_thresh, precision_recall_thresh

def make_roc(labels, probs):
    """
    Inputs:
        labels: one-hot encoded labels (columns being the length of class values, rows being individual samples)
        probs: prediction probability values of each class
        classwise: either get the classwise or macro fpr, tpr to make the ROC curve
    Returns:
        tpr, fpr, thresholds, roc_auc, n_classes
    *Supports upto 7 classes for now
    """
    # convert into numpy arrays
    labels = np.array(labels)
    probs = np.array(probs)
    n_classes = labels.shape[1]
    # add class label/class dict checking
    if probs.shape != labels.shape:
        print('the given label shape [{}] and given probability shape [{}] doesn\'t match'.format(labels.shape, probs.shape))
        return

    # make fpr, tpr, threshold, and roc_auc values for each class
    fpr = dict()
    tpr = dict()
    thresholds = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], thresholds[i] = roc_curve(labels[:, i], probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    return tpr, fpr, thresholds, roc_auc

def plot_AUC(tpr, fpr, roc_auc, cls_dict=None, classwise=False):
    n_classes = len(tpr.keys()) - 1  # -1 for not counting macro
    lw = 2
    if classwise:
        # go through each classes
        for i in range(n_classes):
            cls = cls_dict[i]
            plt.plot(fpr[i], tpr[i], color=plt.cm.tab10(i), lw=lw, label='{0} (AUC = {1:0.2f})'.format(cls, roc_auc[i]))
            plt.title('Multi-Class AUCROC')
    else:
        plt.plot(fpr['macro'], tpr['macro'], color='darkorange', lw=lw, label='AUC = {:0.2f}'.format(roc_auc['macro']))
        plt.title('Macro AUCROC')

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)  # 50%
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

def generate_multi_macro_AUCs(df_dict):
    """
    df_dict: dictionary of {test_name:csv path}
    """
    lw = 2
    plt.figure(dpi=200)
    for i, k in zip(df_dict.keys(), range(len(df_dict))):
        df = pd.read_csv(df_dict[i])
        numerical_labels, onehot_labels, probs = extract_labels_probabilities(df)
        tpr, fpr, thresholds, roc_auc = make_roc(onehot_labels, probs)
        plt.plot(fpr['macro'], tpr['macro'], color=plt.cm.tab10(k), lw=lw, label='{0} (AUC = {1:0.2f})'.format(i, roc_auc['macro']))
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)  # 50%
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.axis('square')
    plt.grid()
    plt.show()


def exam_probabilities(df):
    pt_ids, pred, pred_0_avg_lst, pred_0_std_lst, pred_1_avg_lst, pred_1_std_lst, pred_2_avg_lst, pred_2_std_lst = [], [], [], [], [], [], [], []
    plt.figure(figsize=(5, 10), dpi=300)
    for i in df['pt_id'].unique():
        df_pt = df[df['pt_id'] == i]
        true_label = df_pt['true'].unique().item()
        pred_0_avg, pred_0_std = df_pt['score_0'].mean(), df_pt['score_0'].std()
        pred_1_avg, pred_1_std = df_pt['score_1'].mean(), df_pt['score_1'].std()
        pred_2_avg, pred_2_std = df_pt['score_2'].mean(), df_pt['score_2'].std()
        # print('Patient: {}, True: {}, score_0: {:.3f}+/-{:.3f}, score_1: {:.3f}+/-{:.3f}, score_2: {:.3f}+/-{:.3f}'.format('13127308', true_label, pred_0_avg, pred_0_std, pred_1_avg, pred_1_std, pred_2_avg, pred_2_std))
        # append lists to keep track in plot
        if 'tensor' in i:
            i = i.replace('tensor(', '').replace(')', '')
        pt_ids.append('{}_{}_{}'.format(i, true_label, np.argmax([pred_0_avg, pred_1_avg, pred_2_avg])))
        pred_0_avg_lst.append(pred_0_avg)
        pred_0_std_lst.append(pred_0_std)
        pred_1_avg_lst.append(pred_1_avg)
        pred_1_std_lst.append(pred_1_std)
        pred_2_avg_lst.append(pred_2_avg)
        pred_2_std_lst.append(pred_2_std)
    plt.errorbar(pred_0_avg_lst, pt_ids, xerr=pred_0_std_lst, label='AL_NP score', ls='none', fmt='x', markersize=5,
                 capsize=5)
    plt.errorbar(pred_1_avg_lst, pt_ids, xerr=pred_1_std_lst, label='CP_NP score', ls='none', fmt='x', markersize=5,
                 capsize=5)
    plt.errorbar(pred_2_avg_lst, pt_ids, xerr=pred_2_std_lst, label='NL_NP score', ls='none', fmt='x', markersize=5,
                 capsize=5)
    plt.xlabel('Probabilities')
    plt.ylabel('ID_True_Pred')
    plt.legend()
    plt.show()


def report_boostrap_CI(df, data_frac, examwise=False, return_df=False, verbose=False):
    """
    a more generalized bootstrap confidence interval calculation function
    """
    # get the class length
    n_classes = len(df.true.unique())

    # split into true data classes
    df_cls = {}
    max_size = {}
    min_size = {}
    for i in range(n_classes):
        df_cls[i] = df[df['true'] == i]
        max_size[i] = len(df_cls[i])
        if isinstance(data_frac, (float)):
            min_size[i] = int(max_size[i] * data_frac)
        else:
            min_size[i] = data_frac
    if verbose:
        print('build precision, recall, f1score dictionaries...')
    # create a dictionary of precision, recall, and f1 scores, and prediction probability
    if not examwise:
        avg_prob = {}
    avg_precision = {}
    avg_recall = {}
    avg_fscore = {}

    overall_precision = []
    overall_recall = []
    overall_fscore = []

    cls_strings = [str(i) for i in range(n_classes)]
    # add different sets to that dictionary
    for name in cls_strings:
        if not examwise:
            avg_prob[name] = []
        avg_precision[name] = []
        avg_recall[name] = []
        avg_fscore[name] = []

    # over 1000 iterations
    for i in range(1000):
        # for each class randomly pick a sample size and select those samples
        samples = {}
        for k in range(n_classes):
            sample_size = randint(min_size[k], max_size[k])
            samples[k] = df_cls[k].sample(sample_size)
        # consolidate the true and predictions
        true = []
        pred = []
        for k in range(n_classes):
            true.extend(samples[k]['true'].to_list())
            pred.extend(samples[k]['pred'].to_list())

        # need to get the correct probabilities
        if not examwise:
            pred_prob = {}
            for name in cls_strings:
                pred_prob[name] = samples[int(name)]['score_{}'.format(name)].to_list()

        # get the prediction classification report from the predictions and truth values
        dct = classification_report(true, pred, output_dict=True, zero_division=0)

        for j in cls_strings:
            if not examwise:
                avg_prob[j].append(np.mean(pred_prob[j]))
            avg_precision[j].append(dct[j]['precision'])
            avg_recall[j].append(dct[j]['recall'])
            avg_fscore[j].append(dct[j]['f1-score'])

        overall_precision.append(dct['weighted avg']['precision'])
        overall_recall.append(dct['weighted avg']['recall'])
        overall_fscore.append(dct['weighted avg']['f1-score'])

    # report final one
    if verbose:
        print('Final Report (1000 iterations)')
        for j in cls_strings:
            print('Class: {}'.format(j))
            if not examwise:
                print(
                    'Precision:{:.3f} +/- {:.3f}\nRecall:{:.3f} +/- {:.3f}\nFScore:{:.3f} +/- {:.3f}\nPrediction Probabilty:{:.3f} +/- {:.3f}\n'.format(
                        np.mean(avg_precision[j]), np.std(avg_precision[j]),
                        np.mean(avg_recall[j]), np.std(avg_recall[j]),
                        np.mean(avg_fscore[j]), np.std(avg_fscore[j]),
                        np.mean(avg_prob[j]), np.std(avg_prob[j]),
                    ))
            else:
                print('Precision:{:.3f} +/- {:.3f}\nRecall:{:.3f} +/- {:.3f}\nFScore:{:.3f} +/- {:.3f}\n'.format(
                    np.mean(avg_precision[j]), np.std(avg_precision[j]),
                    np.mean(avg_recall[j]), np.std(avg_recall[j]),
                    np.mean(avg_fscore[j]), np.std(avg_fscore[j]),
                ))
        print('Overall')
        print('Precision:{:.3f} +/- {:.3f}\nRecall:{:.3f} +/- {:.3f}\nFScore:{:.3f} +/- {:.3f}\n'.format(
            np.mean(overall_precision), np.std(overall_precision),
            np.mean(overall_recall), np.std(overall_recall),
            np.mean(overall_fscore), np.std(overall_fscore),
        ))
    if return_df:
        final_report_df = []
        for j in cls_strings:
            precision_range = '[{:.3f}-{:.3f}]'.format(np.mean(avg_precision[j]), np.std(avg_precision[j]))
            recall_range = '[{:.3f}-{:.3f}]'.format(np.mean(avg_recall[j]), np.std(avg_recall[j]))
            f1score_range = '[{:.3f}-{:.3f}]'.format(np.mean(avg_fscore[j]), np.std(avg_fscore[j]))
            line = [precision_range, recall_range, f1score_range]
            final_report_df.append(line)
        final_report_df = pd.DataFrame(final_report_df)

        return final_report_df