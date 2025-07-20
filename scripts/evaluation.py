# python file for evaluation metrics (e.g., F1, confusion matrix)
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix 

# plots confusion matrix using seaborn heatmap
# y_true = true data
# y_pred = model prediction
def plot_cf(y_true, y_pred, labels=None):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6,4))

    # first param "cm" = confusion matrix in array format
    # annot = True -> show numbers in each heatmap cell
    # fmt = 'd' -> show numbers as integers
    ax = sns.heatmap(cm, annot=True, fmt='d')

    # set x-axis label and ticks
    ax.set_xlabel("Predicted", fontsize=14, labelpad=20)
    
    if labels:
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)

    # plot title
    ax.set_title("Confusion Matrix for Student Depression Dataset", fontsize=14, pad=20)
    plt.show()
    return cm

# resources used as guidance
# - https://proclusacademy.com/blog/practical/confusion-matrix-accuracy-sklearn-seaborn/
# - https://proclusacademy.com/blog/explainer/confusion-matrix-accuracy-classification-models/