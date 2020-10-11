from experimentation.dataPrepTest import *
from experimentation.sentimentAnalysisModel import *
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_selection import SelectFromModel

# Start time of program
start = datetime.now()
print("Start: {}".format(start))

''' Date curation and preparation'''

product_reviews = getDF('AMAZON_FASHION.json.gz')
product_reviews = product_reviews.drop_duplicates(
    ['asin', 'reviewerID', 'reviewText', 'reviewerName', 'summary', 'overall', 'unixReviewTime'], keep='first')
product_reviews = cleanse(product_reviews)
product_reviews = tokenize_features(product_reviews)
product_reviews = product_reviews.sample(n=100000)
print("Number of records after cleanse for sentiment analysis: {}".format(len(product_reviews)))

'''Build sentiment classification model'''
product_reviews['sentiment'] = product_reviews['overall'].apply(sentiment)
product_reviews['all_features'] = product_reviews.apply(combined_features, axis=1)

print('--------- Sentiment Analysis Model Results --------')
sentimentAnalysisResults = build_and_analyse_sa_model(product_reviews)


def calculateReviewerExperienceFeatures(features):
    # Calculate count aggregates grouped by reviewer ID for each type of review sentiment
    reviewSumsTotal = features.groupby('reviewerID').agg(
        reviewer_total_count=pd.NamedAgg(column='sentiment', aggfunc='count'))
    reviewSumsPositive = features[features['sentiment'] == 'positive'].groupby('reviewerID').agg(
        reviewer_positive_count=pd.NamedAgg(column='sentiment', aggfunc='count'))
    reviewSumsNegative = features[features['sentiment'] == 'negative'].groupby('reviewerID').agg(
        reviewer_negative_count=pd.NamedAgg(column='sentiment', aggfunc='count'))
    reviewSumsNeutral = features[features['sentiment'] == 'neutral'].groupby('reviewerID').agg(
        reviewer_neutral_count=pd.NamedAgg(column='sentiment', aggfunc='count'))

    # Merge these with other features
    features = pd.merge(features, reviewSumsPositive, on=['reviewerID'], how='outer', suffixes=['_and', '_or'])
    features = pd.merge(features, reviewSumsNegative, on=['reviewerID'], how='outer', suffixes=['_and', '_or'])
    features = pd.merge(features, reviewSumsNeutral, on=['reviewerID'], how='outer', suffixes=['_and', '_or'])
    features = pd.merge(features, reviewSumsTotal, on=['reviewerID'], how='outer', suffixes=['_and', '_or'])

    # Turn n/a's into 0's
    features.fillna({'reviewer_positive_count': 0}, inplace=True)
    features.fillna({'reviewer_negative_count': 0}, inplace=True)
    features.fillna({'reviewer_neutral_count': 0}, inplace=True)

    # Calculate negative experience percentage
    features['reviewer_neg_experience_percentage'] = features['reviewer_negative_count'] / features[
        'reviewer_total_count']

    # Calculate positive experience percentage
    features['reviewer_pos_experience_percentage'] = features['reviewer_positive_count'] / features[
        'reviewer_total_count']

    # Turn n/a's into 0's
    features.fillna({'reviewer_neg_experience_percentage': 0}, inplace=True)
    features.fillna({'reviewer_pos_experience_percentage': 0}, inplace=True)

    # drop columns that are not needed
    features = features.drop(
        columns=['all_features', 'reviewer_positive_count', 'reviewer_negative_count', 'reviewer_neutral_count',
                 'reviewer_total_count'])

    features['experience_classification'] = ""
    # Set values for experience classification
    for i in range(0, len(features) - 1):
        experience_classification_index = features.columns.get_loc('experience_classification')
        neg_experience_index = features.columns.get_loc('reviewer_neg_experience_percentage')
        pos_experience_index = features.columns.get_loc('reviewer_pos_experience_percentage')
        neg_experience_value = features.iat[i, neg_experience_index]
        pos_experience_value = features.iat[i, pos_experience_index]
        if neg_experience_value > pos_experience_value:
            classification = -1
        elif neg_experience_value < pos_experience_value:
            classification = 1
        else:
            classification = 0
        features.iat[i, experience_classification_index] = classification
        i += 1
    return features


resultsWithReviewExperienceClassification = calculateReviewerExperienceFeatures(sentimentAnalysisResults)

print("Time: {}".format(datetime.now()))

'''Add product metadata to add new features for predictive model'''
metadata = getDF('meta_AMAZON_FASHION.json.gz')
# metadata.to_csv('amazon_product_data.csv')
# metadata['price'] = metadata['price'].apply(price_cleanse)

'''Combine product meta data with review data'''
fashion_all_features = pd.merge(resultsWithReviewExperienceClassification, metadata, on=['asin'], how='inner',
                                suffixes=['_and', '_or'])
fashion_all_features = fashion_all_features.dropna(subset=['brand'])
fashion_all_features = fashion_all_features.dropna(subset=['vote'])
all_features = fashion_all_features.sample(n=5000)

print("Time: {}".format(datetime.now()))

'''Predictive Model comparison prep'''
# all_features = all_features.sample(n=100000)
all_features = all_features.drop(
    columns=['style', 'details', 'category', 'rank', 'main_cat', 'price', 'asin', 'fit', 'reviewerID', 'reviewTime',
             'reviewerName', 'reviewText', 'summary', 'image_and', 'tech1', 'description', 'title', 'also_buy',
             'image_or', 'tech2', 'feature', 'also_view', 'similar_item', 'date'])

all_features['vote'] = all_features['vote'].apply(float)
all_features['experience_classification'] = all_features['experience_classification'].apply(float)
model_features = pd.get_dummies(all_features)

# Fit random forest model
Y = model_features['experience_classification']

'''Predictive Model Build'''
'''Predictive Model Build'''
X = model_features.drop(columns=['experience_classification'])
# Split data in test and training sets
x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=0)

# Feature selection
feature_selection = SelectFromModel(RandomForestClassifier(n_estimators=100))
feature_selection.fit(x_train, y_train)
selected_features = feature_selection.get_support()
print("Selected features: ")
print(x_train.columns[selected_features])
print("Number of selected features: " + str(len(selected_features)))

# Build model
classifier = RandomForestClassifier(n_estimators=100)
x_train = x_train[x_train.columns[(feature_selection.get_support())]]
x_test = x_test[x_test.columns[(feature_selection.get_support())]]

'''COnvert to tfidf
cv = CountVectorizer()
# x_train_transform = cv.fit_transform(x_train)
# x_test_transform = cv.transform(x_test)'''

# Create random forest classifier
print("Number of records after cleanse for experience model: {}".format(len(X)))

classifier.fit(x_train, y_train)
y_test_predictions = classifier.predict(x_test)
print('--------- Random Forest Results ------------>')
print('Classification Report:')
print(classification_report(y_test, y_test_predictions))
print('Accuracy Score: ' + str(accuracy_score(y_test, y_test_predictions)))

''''# kNN Model
knn_model = KNeighborsClassifier(n_neighbors=7)
# Train the model using the training sets
knn_model.fit(x_train, y_train)
y_test_predictions = knn_model.predict(x_test)
print('--------- knn Results ------------>')
print('Classification Report:')
print(classification_report(y_test, y_test_predictions))
print('Accuracy Score: ' + str(accuracy_score(y_test, y_test_predictions)))'''

# end time of program
end = datetime.now()
print('Duration: {}'.format(end - start))
