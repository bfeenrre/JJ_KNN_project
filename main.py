#!/usr/bin/env python
import os
from pathlib import Path
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from knn import KNN

# make sure we're working in the directory this file lives in,
# for imports and for simplicity with relative paths
os.chdir(Path(__file__).parent.resolve())

# our code
from utils import handle, main, weekdays, months, today
## from knn import KNN

@handle("clean")
def clean():
    # load raw data
    dataset = pd.read_csv(r'C:\Users\bfeen\OneDrive\Desktop\coding_projects\JJ_ML_project\data\items-2022-03-03-2023-01-31.csv', low_memory=False)
    
    # get columns that're being used for the model
    df_raw = pd.DataFrame(dataset, columns = ['Date', 'Item', 'Qty'])
    
    # ignore "voided" item labels
    df_non_void = df_raw[~df_raw['Item'].str.contains('void', case=False)]

    # set data-selection mask - currently only selecting two-rivers products
    get = ((df_non_void['Item'].str.contains('sandwich', case=False))
    | (df_non_void['Item'].str.contains('bagel', case=False)) 
    | (df_non_void['Item'].str.contains('wrap', case=False)))

    # get desired data using mask
    df = df_non_void[get]

    list_days = df['Date'].unique()
    num_days = list_days.shape[0]

    list_items = df['Item'].unique()
    num_items = list_items.shape[0]
    print("Cleaning and exporting daily sales data for the following items...")
    print(list_items)

    # create new, faster, summarized dataset: row is day, column is item
    raw_cleaned = np.zeros(num_days * num_items).reshape(num_days, num_items)
    
    for i in range(num_days):
        for j in range(num_items):
            sift = df[(df['Item'] == list_items[j]) & (df['Date'] == list_days[i])]
            raw_cleaned[i, j] = sift.shape[0]

    cleaned_df = pd.DataFrame(raw_cleaned, columns=list_items)
    cleaned_df.insert(0, 'Date', list_days)
    fname = Path("..", "data", "cleaned_day_item_array.csv")
    cleaned_df.to_csv(fname)

@handle("predict")
def predict():
    fname = Path("..", "data", "cleaned_day_item_array.csv")
    data_all = pd.read_csv(fname)
    df_all = pd.DataFrame(data_all)

    dates = df_all.index
    print(dates)

    list_items = df_all.columns
    num_items = list_items.shape[0]
    print("Predicting today's sales for the following items...")
    print(list_items)

    # something broken in here - I need to sleep though

    pred = np.zeros(num_items)
    for item in list_items:
        data = df_all.loc[:, ['Date', item]]
        n = data.shape[0]
        data_vec = np.array(data)
        
        weekday_arr = weekdays(data_vec[:, 0]).reshape(n, 1)
        month_arr = months(data_vec[:, 0]).reshape(n, 1)
        quantities = np.asarray(data_vec[:, 1]).reshape(n, 1)

        date_data = np.append(month_arr, weekday_arr, axis = 1)
        learnable_data = np.append(date_data, quantities, axis = 1)
        if item == 'Sandwich, Caprese':
            print(learnable_data)
            print(quantities.dtype)

        X = learnable_data[:, 0:2]
        y = learnable_data[:, 2]
        if item == 'Sandwich, Caprese':
            print(X)
            print(y)

        #plt.scatter(X[:, 0], X[:, 1], c = y, cmap = 'viridis')
        #plt.colorbar()
        plt.show()

        #model = KNN(k = 10)
        #model.fit(X, y)
        #pred[np.where(list_items == item)] = model.predict(today())
    
    print(pred)

if __name__ == "__main__":
    main()
