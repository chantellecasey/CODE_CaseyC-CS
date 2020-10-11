import pandas as pd


# denote positive sentiments as 1 and negative sentiments as 0
def sentiment(n):
    return "positive" if n >= 4 else "negative"


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
    for i in range(0, len(features)):
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
