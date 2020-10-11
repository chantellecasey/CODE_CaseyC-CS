import matplotlib.pyplot as plt


def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color=(0.1, 0.7, 1), label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Sentiment Classification')
    plt.legend()
    plt.show()
