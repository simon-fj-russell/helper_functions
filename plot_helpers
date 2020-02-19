import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

   
def plot_confusion_matrix(y_test, y_pred_prob):
    """Pretty prints a confusion matrix as a figure

    Args:y_test data & y predicted probability 
    Rerturns: Just plots the confusion matrix.
    """
    # Confusion_matrix, this a 2x2 array of the number of TN, FP, FN, TP
    cf_matrix = confusion_matrix(y_test, y_pred_prob > 0.5)
    
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.show()
    
    
from sklearn.metrics import roc_curve, roc_auc_score
def get_auc(y, y_pred_probabilities, class_labels, column =1, plot = True):
    """Plots ROC AUC
    """
    fpr, tpr, _ = roc_curve(y == column, y_pred_probabilities[:,column],drop_intermediate = False)
    roc_auc = roc_auc_score(y_true=y, y_score=y_pred_probabilities[:,1])
    print ("AUC: ", roc_auc)
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


def plot_svc_decision_function(model, ax=None, plot_support=True):
    """ Convenience function to plot results and buffers, extracted from Python Data Science Handbook
    Used for SVM notebook
    """
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    
    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    
    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, edgecolors='black', facecolors='None');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)



def Plot_Point(X,y,Target,Neighbors, ax=None):
    """ Plots sample observation, Targets and some Neighors
    Used for K-NN
    """
    
    if (ax==None):
        fig,ax = plt.subplots()

    scatter=plt.scatter(X[:, 0], X[:, 1], c=y, s=50 , cmap='rainbow');
    # produce a legend with the unique colors from the scatter
    legend1 = ax.legend(*scatter.legend_elements(),
                        loc="upper right", title="Classes")
    ax.scatter(Target[0][0],Target[0][1], s=100, linewidth=1, edgecolors='black', facecolors='black');
    ax.scatter(Neighbors[:,0],Neighbors[:,1], s=200, linewidth=1, edgecolors='black', facecolors='None');
