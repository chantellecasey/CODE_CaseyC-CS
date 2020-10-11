
from data_preparation import *
from sentimentAnalysisModel import *
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Start time of program
start = datetime.now()

all_features = pd.read_csv('fashion_all_features.csv')
# Data preparation
# labels
all_features = all_features.dropna(subset=['brand', 'title'])
# Fit random forest model
Y = all_features[['brand', 'title']]
# X = all_features.drop(columns=['brand', 'title'])
X = all_features['sentiment']
# Split data in test and training sets
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.33, random_state=66)

# COnvert to tfidf
tfidf = TfidfVectorizer()
x_train_transform = tfidf.fit_transform(x_train)
x_test_transform = tfidf.transform(x_test)

# Create random forest classifier
# How do I know how many estimators
clf=RandomForestClassifier(n_estimators=100)
clf.fit(x_train_transform, y_train)
y_test_predictions = clf.predict(x_test_transform)
print(classification_report(y_test, y_test_predictions))

# end time of program
end = datetime.now()
print('Duration: {}'.format(end - start))