#!/usr/bin/env python
import os
from pathlib import Path
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from knn import KNN
import time
from data_manager import DataManager, update_clean_data
from data_manager import food_items, bakery_items, bottle_drink_items

# make sure we're working in the directory this file lives in,
# for imports and for simplicity with relative paths
os.chdir(Path(__file__).parent.resolve())

# our code
from utils import handle, main

## GOALS FOR THIS PROJECT FOR THE FUTURE
#   - implement bagging with my ensembles to reduce variance of my data
#   - add more dimensions to my feature space
#       - distance to a reading break/school holiday
#       - distance to christmas
#       - distance to new years
#       - proximity to finals season
#   - quantitatively find the best (or at least a near-optimal) weighting scheme for my feature space somehow

#   - QUANTITATIVELY TEST AGAINST JJ'S GOOGLE SHEETS SYSTEM - TEST IF KNN IS REALLY AN IMPROVEMENT

# status struct definition
class Status:
    quit = 0 
    timer = 0
    model_mode = "knn"
    items = food_items
    data_mng = DataManager()
    model = None

# menu constants
valid_commands = ["-o", "-q", "-t", "-m", "update_clean_data"]
main_options = "options:\n    - '-q' does what you think it does\n    - '-t' timer on/off\n    - '-m' choose what type of model you want to use\n    - 'update_clean_data' load raw data from csv, clean it, use it to calculate summary of daily sales by item with augmented date data, and store summary as new, learnable csv\n    - coming soon..."
mode_options = "options:\n    - 'knn' single knn model\n    - 'nknn' ensemble knn model\n    - 'r' return to main menu"

# global status variable
status = Status()

@handle("run")
def run():
    # init
    print(main_options)
    while(status.quit == False):
        command = input().lower().replace(" ", "")
        if command in valid_commands:
            if command in valid_commands[0:4]:
                update_status(command)
            else:
                if status.timer == 1:
                    t_s = time.time()
                run_func(command)
                if status.timer == 1:
                    t_e = time.time()
                    print("run time: " + str(t_e - t_s))
            if (status.quit == False):
                print("done! please enter another command...")
        else:
             print("please enter a valid command [type '-o' for list of valid commands]\n")
    print("see ya!")

def update_status(command):
    if (command == "-q"):
        status.quit = True
    if (command == "-t"):
        status.timer ^= 1
    if (command == "-o"):
        print("\n" + main_options)
    if (command == "-m"):
        valid_modes = ["knn", "nknn"]
        print("current model mode is... " + status.model_mode)
        print(mode_options)
        while(True):
            mode_sel = input().lower().replace(" ", "")
            if (mode_sel == "q"):
                return
            if mode_sel in valid_modes:
                status.model_mode = mode_sel
                return
            else:
                print("invalid command...")
                print(mode_options)

def run_func(function):
    if function == "update_clean_data":
        update_clean_data()

if __name__ == "__main__":
    main()



################################## FEELING BOGGED DOWN BY MAIN IMPLEMENTATION STYLE - WANNA TRY SOME ENSEMBLE TESTING W/ BOOTSTRAPPING

@handle("test_bagged__knn")
def test_bagged_optimzed_knn():
    update_clean_data()

    data = DataManager()
    data.load_data()

    ensemble_size = 10
    bootstrap_factor = 10
    ks = range(0, 50)

    items_to_test = bakery_items + food_items + bottle_drink_items
    nitems = len(items_to_test)

    for item in items_to_test:
        X_train, y_train = data.get_learnable_data(item, True, bootstrap_factor * ensemble_size)
        n, d = X_train.shape[0]
        X_valid, y_valid = data.get_learnable_data(item, True, (bootstrap_factor / 2) * ensemble_size)
        preds = np.zeros(ensemble_size, items_to_test)
        
        for i in range(ensemble_size):
            X_t = X_train[(i * (n/ensemble_size)):((i+1) * (n/ensemble_size))]
            y_t = y_train[0:(n/ensemble_size)]
            X_v = 
            model = KNN(ks).fit(X_t, y_t).find_best_k()
            preds = model.predict(X_v)
        ## @TODO quantitatively test mode vs median vs mean on this dataset
    
    print("printing predictions for ")


################################## UNUSED BY MAIN/ UN-REFACTORED SO FAR (DO NOT DELETE UNTIL REFACTORED SOMEWHERE ELSE)


@handle("predict")
def predict():
    df_all = load_data()
    list_items = df_all.columns[2:]
    num_items = list_items.shape[0]

    print("Predicting today's sales for the following items...")
    print(list(list_items))
    num_items = list_items.shape[0]

    pred_tot = np.zeros(num_items)

    ensemble_size = 15

    for j in range(ensemble_size):
        pred = np.zeros(num_items)

        ks = np.round(find_ks()).astype(int)

        for i in range(num_items):

            data = get_preprocessed_data(list_items[i], df_all)

            X = data[:, 0:2]
            y = np.round(data[:, 2])

            model = KNN(k=ks[i])
            model.fit(X, y)
            today_ = today().reshape(1, -1)
                
            pred[i] = model.predict(today_)
    
        pred_tot += pred
    print(np.round(pred_tot / ensemble_size))

@handle('test_granular')
def test_granular():
    df_all = status.data_mng.df_nv_clean
    list_items = df_all.columns[2:]
    num_items = list_items.shape[0]

    print("Running granular cross-validation for the following items...")
    print(list(list_items))

    for item in list_items:

        data = get_preprocessed_data(item, df_all)
        np.random.shuffle(data)

        X = data[:, 0:2]
        y = data[:, 2]

        ks = range(1, 21, 1)
        folds = 20
        cv_accs = test(X, y, ks, folds, 'abs')

        plt.clf()
        plt.plot(ks, cv_accs)
        plt.xlabel('k')
        plt.ylabel('err_abs')
        fname = Path("..", "figs", "granular", item.lower().replace(' ', '').replace(',', '_') + "_cv_accs.pdf")
        plt.savefig(fname)
        print(f"figure saved as {fname}")

@handle('test_aggregate')
def test_aggregate():
    df_all = load_data()
    list_items = df_all.columns[2:]
    num_items = list_items.shape[0]

    ks = range(1, 21, 1)
    folds = 20

    print("Running aggregate cross-validation for the following items...")
    print(list(list_items))
    print("Testing k's in: ")
    print(ks)
    print("Folds: ")
    print(folds)

    itemized_errors = np.zeros((num_items, len(ks)))

    for item in list_items:

        data = get_preprocessed_data(item, df_all)
        np.random.shuffle(data)

        X = data[:, 0:2]
        y = data[:, 2]

        itemized_errors[np.where(list_items == item)] = test(X, y, ks, folds, 'rel')

    aggregate_cv_accs = np.mean(itemized_errors, axis = 0)
    
    plt.plot(ks, aggregate_cv_accs, label = 'item')
    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.legend(loc = 'upper right')
    fname = Path("..", "figs", "aggregate_cv_accs.pdf")
    plt.savefig(fname)
    print(f"figure saved as {fname}")

@handle('granular_summary')
def granular_summary():
    df_all = load_data()
    list_items = df_all.columns[2:]
    num_items = list_items.shape[0]

    ks = range(1, 21, 1)
    folds = 20

    print("Running aggregate cross-validation for the following items...")
    print(list(list_items))
    print("Testing k's in: ")
    print(ks)
    print("Folds: ")
    print(folds)

    for item in list_items:

        data = get_preprocessed_data(item, df_all)
        np.random.shuffle(data)

        X = data[:, 0:2]
        y = data[:, 2]

        cv_accs = test(X, y, ks, folds, 'rel')
        plt.plot(ks, cv_accs, label = item.lower().replace(' ', '').replace(',', '_'))
    
    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.legend(loc = 'upper right')
    fname = Path("..", "figs", "summary_granular_cv_accs.pdf")
    plt.savefig(fname)
    print(f"figure saved as {fname}")

@handle('test_jj')
def test_jj():
    df_all = load_data()
    list_items = df_all.columns[2:]
    num_items = list_items.shape[0]

    print("Testing JJ's current model's accuracy for the following items: ")
    print(list_items)

    accs = np.zeros(num_items)
    j = 0
    for item in list_items:

        data = get_preprocessed_data(item, df_all)

        X = data[:, 0:2]
        y = data[:, 2]

        num_preds = y.shape[0] - 4
        y_pred = np.zeros(num_preds)
        for i in range(4, num_preds + 4, 1):
            y_pred[i - 4] = 0.4*(y[i-1]) + 0.3*(y[i-2]) + 0.2*(y[i-3]) + 0.1*(y[i-4])

        accs[j] = np.mean(np.abs((y[4:] - y_pred)/(y[4:] + 1)))
        j = j + 1

    print("Accuracies: ")
    print(accs)

def run_k_test(df_all):
    list_items = df_all.columns[2:]
    num_items = list_items.shape[0]

    best_ks = np.zeros(num_items)

    for i in range(num_items):

        data = get_preprocessed_data(list_items[i], df_all)
        np.random.shuffle(data)

        X = data[:, 0:2]
        y = data[:, 2]

        ks = range(1, 20, 1)
        folds = 25
        cv_accs = test(X, y, ks, folds, 'abs')
        best_ks[i]= ks[np.argmin(cv_accs)]
    return best_ks
        
@handle('find_best_ks')
def find_ks():
    df_all = load_data()
    list_items = df_all.columns[2:]
    num_items = list_items.shape[0]

    num_tests = 5

    #rint("Using cross-validation to find the best k for the following items...")
    #print(list(list_items))

    best_ks = np.zeros(num_items)

    for j in range(num_tests):
        best_ks += run_k_test(df_all)

    return best_ks/num_tests

@handle('test_best_knn')
def test_best_knn():
    df_all = load_data()
    list_items = df_all.columns[2:]
    num_items = list_items.shape[0]

    print("Testing my model's accuracy for the following items: ")
    print(list_items)

    ks_by_item = find_ks()

    accs = np.zeros(num_items)
    for j in range(num_items):


        data = get_preprocessed_data(list_items[j], df_all)
        np.random.shuffle(data)
        n = int(data.shape[0])

        X_train = data[:, 0:2]
        y_train = data[:, 2]
        test_indx = np.random.randint(n, size = int(n/2))
        test_data = data[test_indx]
        X_test = test_data[:, 0:2]
        y_test = test_data[:, 2]

        model = KNeighborsClassifier(n_neighbors=int(ks_by_item[j]))
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accs[j] = np.mean(np.abs(((y_pred - y_test) + 1)/(y_pred + 1)))

    print("Accuracies: ")
    print(accs)
