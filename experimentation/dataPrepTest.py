from numpy.core.defchararray import lower
import json
import pandas as pd
import gzip
from tokenizer import *

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
def cleanse(p, cols=[]):
    # drop nan values
    p = p.dropna(subset=['reviewText', 'summary'])
    # p = p[p.overall != 3]
    # make sure all records in review columns are strings and change to strings
    # change all data to lower case, and conve
    for col in cols:
        for i in range(0, len(p) - 1):
            if type(p.iloc[i][col]) != str:
                p.iloc[i][col] = str(p.iloc[i][col])
        p[col].apply(lower)
    return p


''' Input a price string with a dollar and output the price as a float'''


def price_cleanse(price_str):
    if price_str != '' and price_str.index('-') > 0:
        lower_limit = price_str.substring(0, price_str.index('-') - 1)
    else:
        price_float = float(price_str.replace('$', ''))
    return price_float


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
