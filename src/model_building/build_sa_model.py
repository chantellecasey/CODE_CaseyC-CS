import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from src.helper_scripts.roc_curve import *


def build_and_analyse_sa_model(p):
    X = p['all_features']
    Y = p['sentiment']

    # Split data into test data and training data, with a 25%, 75% split respectively
    x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=0)

    # TFIDF Vectorizer used to assign weights to words in documents
    # fit both training and test data to count vectorizer
    tfidf = TfidfVectorizer()
    x_train_transform = tfidf.fit_transform(x_train)
    x_test_transform = tfidf.transform(x_test)
    # use the Logistic Regression as this is a binary classification
    # fit training data model to logistic regression model
    model = LogisticRegression()
    model.fit(x_train_transform, y_train)
    y_test_predictions = model.predict(x_test_transform)

    print('--------- Sentiment Classifier Results ------------>')
    print('Classification Report:')
    print(classification_report(y_test, y_test_predictions))
    print('Accuracy Score: ' + str(accuracy_score(y_test, y_test_predictions)))
    # Area under ROC curve
    probabilities = model.predict_proba(x_test_transform)
    probabilities = probabilities[:, 1]

    auc = roc_auc_score(y_test, probabilities)
    print('AUC: %.2f' % auc)
    fpr, tpr, thresholds = roc_curve(y_test, probabilities, pos_label='positive')
    plot_roc_curve(fpr, tpr)

    # Add results of test and train data to original data frame, return new p with sentiment
    all_x = x_test.append(x_train)
    all_y = y_test.append(y_train)
    results = pd.DataFrame(all_y)
    results['all_features'] = all_x
    p['sentiment'] = all_y
    return p



