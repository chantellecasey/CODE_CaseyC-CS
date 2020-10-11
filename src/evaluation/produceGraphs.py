import pandas as pd
import matplotlib.pyplot as plot


def produceSimpleBarCharts(results, x_axis_title, y_axis_title, title):
    results.plot.bar(x="label", y="count", rot=70, color=(0.1, 0.7, 1), edgecolor='black')
    plot.suptitle(title, fontsize=12)
    plot.xlabel(x_axis_title, fontsize=10)
    plot.ylabel(y_axis_title, fontsize=10)
    plot.show(block=True)


'''Count of classified examples - bar charts'''
# SA Results
sa_results = pd.DataFrame({'label': ['Positive', 'Negative'], 'count': [66483, 152120]})
produceSimpleBarCharts(sa_results, 'Sentiment', 'Count of labelled examples', 'Distribution of classified examples '
                                                                              'per class - Sentiment Classification')

# RF Results
rf_results = pd.DataFrame({'label': ['Positive', 'Neutral', 'Negative'], 'count': [14898, 1184, 30431]})
produceSimpleBarCharts(rf_results, 'Experience Classification', 'Count of labelled examples',
                       'Distribution of classified examples '
                       'per class - Random Forest Model')
# KNN Results
knn_results = pd.DataFrame({'label': ['Positive', 'Neutral', 'Negative'], 'count': [14898, 1184, 30431]})
produceSimpleBarCharts(knn_results, 'Experience Classification', 'Count of labelled examples',
                       'Distribution of classified examples '
                       'per class - kNN Model')


'''Performance by class - Experience Classification'''
performance = {"Positive": [0.63, 0.8, 0.56], "Neutral": [0.03, 0.05, 0.01], "Negative": [0.43, 0.25, 0.31]}
performance_index = ["Precision", "Recall", "F1 Score"]
dataFrame = pd.DataFrame(data=performance, index=performance_index)
dataFrame.plot.bar(rot=15, title="Performance by Class - kNN Model")
plot.show(block=True)
