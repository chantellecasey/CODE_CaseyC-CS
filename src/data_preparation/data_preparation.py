from numpy.core.defchararray import lower
import json
import pandas as pd
import gzip
from src.helper_scripts.tokenizer import *

''' Import json gz file and convert to data frame'''


def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield json.loads(l)


def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')


# Cleanse the product data for testing and training
# Make sure names and reviews are strings


def cleanse(data, dataset_type):
    if dataset_type == "review":
        cols = ['reviewText', 'summary']
        data = data.drop_duplicates(
            ['asin', 'reviewerID', 'reviewText', 'reviewerName', 'summary', 'overall', 'unixReviewTime'],
            keep='first')
        data = data.dropna(subset=['reviewText', 'summary'])
        for col in cols:
            for i in range(0, len(data) - 1):
                if type(data.iloc[i][col]) != str:
                    data.iloc[i][col] = str(data.iloc[i][col])
            data[col].apply(lower)

    elif dataset_type == "product":
        data = data.drop_duplicates(['asin'], keep='first')
        data = data.dropna(subset=['brand', 'vote', 'price', 'rank'])
        data['vote'] = data['vote'].apply(float)
        data['rank'] = data['rank'].apply(extract_rank)
        data['price'] = data["price"].apply(price_to_float)
        # data = reduce_brands(data)
    else:
        print("dataset type not defined")
    return data


# combine columns name and review
def combined_features(row):
    return row['reviewText'] + ' ' + row['summary']


'''Input a price string with a dollar and output the price as a float'''


def price_to_float(price_str):
    if " - " in price_str and "$" in price_str:
        lower_limit = price_str[1: price_str.index('-') - 1]
        lower_limit = float(lower_limit.replace(',', ''))
        upper_limit = price_str[price_str.index('-') + 3:]
        upper_limit = float(upper_limit.replace(',', ''))
        price_float = (upper_limit + lower_limit) / 2
    elif "$" in price_str:
        price_str = price_str.replace(',', '')
        price_float = float(price_str.replace('$', ''))
    else:
        price_float = 0
    return price_float


'''extract rank number from rank column'''


def extract_rank(string):
    if "in" in string:
        rank = string[0: string.index('in') - 1]
        rank = rank.replace(',', '')
        return float(rank)
    else:
        return 0


''''Tokenize each column specified of the data frame specified
    Function that loops through each product ad tokenizes the name and review
    Using Christopher Potts tokenizer script'''


def tokenize_features(p, cols=[]):
    # tokenize each column
    for col in cols:
        column = p[col]
        col_index = p.columns.get_loc(col)
        for i in range(0, len(column) - 1):
            tokenizer = Tokenizer()
            tokenized_column = list(tokenizer.tokenize(column.iloc[i]))
            new_str = ""
            for j in range(0, len(tokenized_column) - 1):
                if i != len(tokenized_column) - 1:
                    new_str += tokenized_column[j] + " "
                else:
                    new_str += tokenized_column[i]
            p.iat[i, col_index] = new_str
    return p


def reduce_brands(data):
    brands = data.groupby(['brand']).agg(
        brand_count=pd.NamedAgg(column='brand', aggfunc='count'))
    brands = brands.sort_values(by="brand_count", ascending=False)
    top_brands = brands.head(20)
    brand_indexes = top_brands.index.values
    print(brand_indexes)
    data = data.loc[data['brand'].isin(brand_indexes)]
    print('After top 20 brands cleanse: ', len(data))
    return data
