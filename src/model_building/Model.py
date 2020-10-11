from src.data_preparation.data_preparation import *
from src.model_building.build_knn_model import build_and_analyse_knn_model
from src.model_building.build_rf_model import *
from src.model_building.categorise import *
from src.model_building.build_sa_model import *


class Model:
    def __init__(self, model_type, param, datafile):
        self.rawFileName = datafile
        self.model_type = model_type
        self.rawDF = getDF(datafile) if model_type == "SA" else datafile
        self.processedDF = None
        self.param = param

    def preProcess(self, dataset_type):
        print("Preprocessing Data")
        print("Raw Data file: ", self.rawFileName)
        print("For Model: ", self.model_type)
        print("Param: ", self.param)
        print("Number of records before cleanse: {}".format(len(self.rawDF)))
        processed_data = cleanse(self.rawDF, dataset_type)
        # processed_data = processed_data.sample(n=test_and_train_data_size)
        if self.model_type == 'SA':
            processed_data['all_features'] = processed_data.apply(combined_features, axis=1)
            processed_data = tokenize_features(processed_data)
        self.processedDF = processed_data
        print("Number of records after cleanse: {}".format(len(processed_data)))

    def label(self):
        if self.model_type == "SA":
            self.processedDF['sentiment'] = self.processedDF['overall'].apply(sentiment)
        elif self.model_type == "RF" or self.model_type == "KNN":
            self.processedDF = calculateReviewerExperienceFeatures(self.processedDF)

    def build(self):
        if self.model_type == "SA":
            return build_and_analyse_sa_model(self.processedDF)
        elif self.model_type == "RF":
            self.processedDF = self.processedDF.drop(
                columns=['style', 'details', 'category', 'rank', 'main_cat', 'price', 'asin', 'fit', 'reviewerID',
                         'reviewTime', 'reviewerName', 'reviewText', 'summary', 'image_and', 'tech1', 'description',
                         'title', 'also_buy', 'image_or', 'tech2', 'feature', 'also_view', 'similar_item', 'date'])
            self.processedDF['experience_classification'] = self.processedDF['experience_classification'].apply(float)
            featuresDF = pd.get_dummies(self.processedDF)
            return build_and_analyse_rf_model(featuresDF, self.param)
        elif self.model_type == "KNN":
            self.processedDF = self.processedDF.drop(
                columns=['style', 'details', 'category', 'main_cat', 'asin', 'fit', 'reviewerID',
                         'reviewTime', 'reviewerName', 'reviewText', 'summary', 'image_and', 'tech1', 'description',
                         'title', 'also_buy', 'image_or', 'tech2', 'feature', 'also_view', 'similar_item', 'date'])
            self.processedDF['experience_classification'] = self.processedDF['experience_classification'].apply(float)
            featuresDF = pd.get_dummies(self.processedDF)
            return build_and_analyse_knn_model(featuresDF, self.param)

