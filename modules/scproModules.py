import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score


def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), 
                           labelsize=18, fontsize=15, numsize=20, cmap=None):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.
    
    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix. 
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.
        
    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names, )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", annot_kws={"size": numsize}, cmap=cmap )
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
#     ax.set_ylim(len(confusion_matrix)-0.5, -0.5)
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=labelsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=labelsize)
    plt.ylabel('TRUE', fontsize=fontsize)
    plt.xlabel('PREDICTED', fontsize=fontsize)
    return fig



def split_binary(df, column, top_split=1, no_event=0, event=1):
    """Replaces numeric values in a DataFrame column with binary values [0,1].
    Arguments
    ----------
    df: pandas.DataFrame
    column: string
        Column name in the data frame with the values to be binarized
    top_split: integer
        Number of values counting backwards from the end of the values list to be given the value 0
        From a list of values [1,2,3,4,5], a top_split=2 will return [1,1,1,0,0]

    Returns
    -------
    pandas.DataFrame
        DataFrame with the values in the specified column given values of 0 or 1
    """
    binary_split_df = df.copy()
    original_value_list = list(set(binary_split_df[column].values))
    ones = len(original_value_list) - top_split
    binary_value_list = [no_event for i in range(ones)]
    for _ in range(top_split):
        binary_value_list.append(event)
    binary_split_df = binary_split_df.replace({column:original_value_list}, {column:binary_value_list})
    return binary_split_df



def find_precision_recall_threshold(estimator, x_test, y_test, target='precision'):
    """Find the probability threshold at which precision and recall cross for the optimal ratio of the two
    Arguments
    ----------
    estimator: sklearn estimator
        Fitted estimator
    y_test: array
        Test array for target classifications
    x_test: array
        Test array for X predictor values
    target: string
        Low metric that must cross the higher metric. Either "precision" or "recall". Default target='precision'

    Returns
    ----------
    Prints out the precision score, recall score, index of target threshold and target threshold number 
    np.array
        New y_pred array of predictions based upon the new threshold
    """
    predict = estimator.predict_proba(x_test)
    precision, recall, thresholds = precision_recall_curve(y_test, predict[:,1])

    precision_recall_list = list(zip(precision, recall))
    if target == 'precision':
        count = 0
        for p, r in precision_recall_list:
            if p > r:
                print("Precision Score: ", p)
                print("Recall Score: ", r)
                print("Index of Precision Score: ", count)
                print("Target Threshold: ", thresholds[count])
                break
            count += 1
    elif target == 'recall':
        count = 0
        for p, r in precision_recall_list:
            if p < r:
                print("Precision Score: ", p)
                print("Recall Score: ", r)
                print("Index of Precision Score: ", count)
                print("Target Threshold: ", thresholds[count])
                break
            count += 1


def create_new_ypred(estimator, x_test, threshold):
    """Create new y_pred based upon custom threshold
    Arguments
    ----------
    estimator: sklearn estimator
        Fitted estimator

    x_test: np.array
        Test array 
    
    threshold: float
        Positive probability must be >= this quantity

    Returns
    ----------
    np.array
        New y_pred array of predictions based upon the new threshold
    """
    predict = estimator.predict_proba(x_test)
    return np.array([1 if x >= threshold else 0 for x in predict[:,1]])


def print_scores(y_test, y_pred, pos_label=1):
    """Print the recall, F1, precision and accuracy scores of the test vs prediction arrays
    Arguments
    ----------
    y_test: array
        Test array
    y_pred: array
        Prediction array

    Returns
    ----------
    Prints out the recall, F1, precision and accuracy scores
    """
    print("F1:", f1_score(y_test, y_pred, pos_label=pos_label))
    print("Precision:" , precision_score(y_test, y_pred, pos_label=pos_label))
    print("Recall:" , recall_score(y_test, y_pred, pos_label=pos_label))
    print("Accuracy:" , accuracy_score(y_test, y_pred))


def score_df(y_test, 
             y_pred, 
             y_proba=np.array(False),  
             scores_list = ['f1', 'precision', 'recall', 'accuracy', 'average_precision'],
             pos_label=1):
    """Creates a df of the recall, F1, precision and accuracy scores of the test vs prediction arrays
    Arguments
    ----------
    scores: list, default=['f1', 'recall', 'precision', 'average_precision', 'accuracy']
        List of string names of scores.     
    y_test: array-like
        Ground truth (correct) target values.
    y_pred: array-like
        Estimated targets as returned by a classifier.
    y_proba: array-like of shape (n_samples), default=None
        List of probabilities for the positive class.
    pos_label: str/int
        Class to report.

    Returns
    ----------
    df: DataFrame 
        Pandas DataFrame containing the specified scores.
    """
    scores_dict = {
    'f1': f1_score(y_test, y_pred, pos_label=pos_label),    
    'recall': recall_score(y_test, y_pred, pos_label=pos_label),
    'precision': precision_score(y_test, y_pred, pos_label=pos_label),
    'average_precision': average_precision_score(y_test, y_pred, pos_label=pos_label),
    'accuracy': accuracy_score(y_test, y_pred)
     }
        
    if y_proba.any():
        scores_dict['roc_auc'] = roc_auc_score(y_test, y_proba[:,1])
        scores_list.append('roc_auc')

    score_values = [scores_dict[score] for score in scores_list]

    df = pd.DataFrame(score_values, index=scores_list, columns=['Scores'])

    # Remove 'roc_auc' if appended to scores_list. Seems to stay in scope within notebooks.
    if y_proba.any():
        scores_list.pop()

    return df

def plot_grid_scores(grid_param, scores_list, labels, linewidth=2.5, figsize=(15,10), fontsize='x-large', loc='best'):
    """Creates a plot of the metric scores of a grid search over values of a specific parameter of the search
     Arguments
    ----------
    grid_param: array-like
        Array of parameter values.     
    scores_list: list
        List of metric value arrays containing the values of the metric over different parameter values.
    labels: list
        List of labels of the metrics.
    linewidth: float
        Float indicating the width of each line on the plot.
    figsize: tuple
        2 integer tuple indicating the size of the plot
    fontsize: int or float or {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}
        The font size of the legend. If the value is numeric the size will be the absolute font size in points. 
        String values are relative to the current default font size. This argument is only used if prop is not 
        specified.
    linestyle: str {'-', '--', '-.', ':', '', (offset, on-off-seq), ...}
        Set the linestyle of the line
    loc: str or pair of floats, default: rcParams["legend.loc"] 
    (default: 'best') ('best' for axes, 'upper right' for figures)
        Location of the legend.
        
        The strings 'upper left', 'upper right', 'lower left', 'lower right' place the legend at the 
        corresponding corner of the axes/figure.
       
        The strings 'upper center', 'lower center', 'center left', 'center right' place the legend at 
        the center of the corresponding edge of the axes/figure.

        The string 'center' places the legend at the center of the axes/figure.

        The string 'best' places the legend at the location, among the nine locations defined so far, 
        with the minimum overlap with other drawn artists. This option can be quite slow for plots with 
        large amounts of data; your plotting speed may benefit from providing a specific location.

    Returns
    ----------
    Matplotlib line plot 
        Line plot containing a line for each of the scoring metrics over each value of the parameter
    """
    default_cycler = (cycler(color=['blue', 'orange', 'green', 'red', 'magenta', 'gray']) +
                      cycler(linestyle=['-', '-', '-', '-','-', '-']))
    plt.rc('axes', prop_cycle=default_cycler)
    plt.figure(figsize=(15,10))
    plt.title('Score Grid', fontsize=fontsize)
    plt.grid(True, fillstyle='right', c='gray', alpha=0.5)
    for score, label in zip(scores_list, labels):
        plt.plot(grid_param, score, label=label, linewidth=linewidth)
    plt.xlabel('Parameter', fontsize=fontsize)
    plt.ylabel('Scores', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(fontsize=fontsize, loc=loc);


def expected_baseline(p, size, test, iterations=1000):
    """Generate baseline scores for common binary classification metrics, based upon 
    the expected probability of each class. Generates scores for F1, precision, recall,
    and average precision. The scores are the mean x number of scoring iterations.

    Arguments
    ----------
    p: float
        Probability of positive class
    size: int
        Size of the test and prediction array
    test: array-like
        Test array with ground truth labels
    iterations: int
        Number of prediction arrays and scores to generate

    Returns
    ----------
    df: DataFrame
        Pandas DataFrame containing the mean scores.
    """
    f1_scores = []
    precision_scores = []
    recall_scores = []
    average_precision_scores = []
    accuracy_scores = []

    for _ in range(iterations):
        pred = np.random.choice([1,0], size=len(test), p=[p,1-p])
        f1_scores.append(f1_score(test, pred))
        precision_scores.append(precision_score(test, pred))
        recall_scores.append(recall_score(test,pred))
        average_precision_scores.append(average_precision_score(test, pred))
        accuracy_scores.append(accuracy_score(test,pred))

    scores_dict = {
        'F1': np.mean(f1_scores),
        'Precision': np.mean(precision_scores),
        'Recall': np.mean(recall_scores),
        'Average Precision': np.mean(average_precision_scores),
        'Accuracy': np.mean(accuracy_scores)
    }
    
    return pd.DataFrame(scores_dict, index=['Scores']).T


def plot_knn_metrics(metrics_df, metric_list, metric_labels, figsize=(15,10), label_font_size='xx-large', tick_font_size='x-large', legend_loc=(0.85,0.75)):
    """Print a plot of metrics for each number of KNN neighbors from a GridSearchCV
    Arguments
    ----------
    metrics_df: pd.DataFrame
        Dataframe of all metrics produced after running GridSearchCV
    metric_list: list
        List of pd.Series from metrics_df to be included in the plot
    metric_labels: list
        List of string labels for each metric in the plot, which will be displayed in the legend.
    figsize: tuple
        Size of plot. Default = (15,10)
    label_font_size: string
        String denoting the size of the axis labels. Labels are matplotlib-allowed sizes.
    tick_font_size: string
        String denoting the size of the axis ticks. Labels are matplotlib-allowed sizes.
    legend_loc: tuple
        Tuple of two floats indicating the position of the legend. Valid values are 0-1.

    Returns
    ----------
    Matplotlib.pyplot plot of all indicated metrics for each number of KNN neighbors
    """
    plt.figure(figsize=figsize)
    metric_label_list = list(zip(metric_list, metric_labels))
    for metric, label in metric_label_list:
        plt.plot(metrics_df, metric, linewidth=2, label=label)
    plt.xlabel('Neighbors', fontsize=label_font_size)
    plt.ylabel('Scores', fontsize=label_font_size)
    plt.xticks(fontsize=tick_font_size)
    plt.yticks(fontsize=tick_font_size)
    plt.legend(loc=legend_loc, fontsize=tick_font_size)
    plt.grid(True);