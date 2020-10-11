from datetime import datetime
from src.model_building.Model import *

if __name__ == '__main__':
    # Start time of program
    start = datetime.now()
    print("Start: {}".format(start))

    print("<------------------------- Sentiment Analysis Model Start ----------------------------->")
    print(" ")
    SA_model_instance = Model("SA", None, "data/raw/AMAZON_FASHION.json.gz")
    SA_model_instance.preProcess("review")
    SA_model_instance.label()
    sa_results = SA_model_instance.build()
    sa_results.to_csv("sa_results.csv")
    print(" ")
    print("<------------------------- Sentiment Analysis Model End ----------------------------->")

    # Load and merge product data
    product_data = getDF('data/raw/meta_AMAZON_FASHION.json.gz')
    merged_data = pd.merge(sa_results, product_data, on=['asin'], how='inner',
                           suffixes=['_and', '_or'])
    merged_data.to_csv("merged_data.csv")

    print("<------------------------- Random Forest Model Start ----------------------------->")
    print(" ")
    # Build RF Models
    params = [15, 30, 45, 60, 75, 90]
    for n in params:
        print('Started - RF Model with param: ', str(n))
        model_start = datetime.now()
        RF_model_instance = Model("RF", n, merged_data)
        RF_model_instance.preProcess("product")
        print("Number of records after preProcess: {}".format(len(RF_model_instance.processedDF)))
        RF_model_instance.label()
        print("Number of records after label: {}".format(len(RF_model_instance.processedDF)))
        RF_model_instance.build()
        print("Number of records after build : {}".format(len(RF_model_instance.processedDF)))

        model_end = datetime.now()
        print('Finished - RF Model with param: ', str(n))
        print('Model Duration: {}'.format(model_end - model_start))
        print("--------------")
    print(" ")
    print("<------------------------- Random Forest Model End ----------------------------->")

    print("<------------------------- kNN Model Start ----------------------------->")
    print(" ")
    # Build kNN Models
    params = [3, 5, 7, 9, 11]
    for n in params:
        print('Started - kNN Model with param: ', str(n))
        model_start = datetime.now()
        KNN_model_instance = Model("KNN", n, merged_data)
        KNN_model_instance.preProcess("product")
        KNN_model_instance.label()
        KNN_model_instance.build()
        model_end = datetime.now()
        print('Finished - kNN Model with param: ', str(n))
        print('Model Duration: {}'.format(model_end - model_start))
        print("--------------")
    print(" ")
    print("<------------------------- kNN Model End ----------------------------->")

    # end time of program
    end = datetime.now()
    print('Duration: {}'.format(end - start))
