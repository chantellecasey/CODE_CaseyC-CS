from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from src.helper_scripts.roc_curve import *
import pandas as pd


def build_and_analyse_rf_model(data, param):
    # Fit random forest model
    Y = data['experience_classification']
    '''Predictive Model Build'''
    '''Predictive Model Build'''
    X = data.drop(columns=['experience_classification'])
    # Split data in test and training sets
    x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=0)

    # Feature selection
    feature_selection = SelectFromModel(RandomForestClassifier(n_estimators=param))
    feature_selection.fit(x_train, y_train)
    selected_features = feature_selection.get_support()
    print("Selected features: ")
    print(x_train.columns[selected_features])
    print("Number of selected features: " + str(len(selected_features)))

    # Build model
    classifier = RandomForestClassifier(n_estimators=param)
    x_train = x_train[x_train.columns[(feature_selection.get_support())]]
    x_test = x_test[x_test.columns[(feature_selection.get_support())]]

    # Fit classifier
    classifier.fit(x_train, y_train)
    y_test_predictions = classifier.predict(x_test)
    print('<--------- Random Forest Results ------------>')
    print('Classification Report:')
    print(classification_report(y_test, y_test_predictions))
    print('Accuracy Score: ' + str(accuracy_score(y_test, y_test_predictions)))

    # Area under ROC curve
    if len(y_test) == 0:
        probabilities = classifier.predict_proba(x_test)
        probabilities = probabilities[:, 1]

        auc = roc_auc_score(y_test, probabilities)
        print('AUC: %.2f' % auc)
        fpr, tpr, thresholds = roc_curve(y_test, probabilities, pos_label='positive')
        plot_roc_curve(fpr, tpr)

    # Add results of test and train data to original data frame, return new data
    all_x = x_test.append(x_train)
    all_y = y_test.append(y_train)
    results = pd.DataFrame(all_x)
    results['experience_classification'] = all_y
    return results
