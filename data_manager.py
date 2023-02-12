import pandas as pd
import numpy as np
from pathlib import Path

import date_utils as date_utils

############################################ item breakdowns
bakery_items = ['Apple Cruffin', 'Danish', 'Scone, Cheese', 'Muffin, Vegan', 'Muffin, Oat', 'Scone, Berry', 'Muffin, Cheese', 'Croissant, Cheese', 'Vegan Granola 1 lb.', 'Regular Granola 1 lb.', 'Honey 2KG', 'Cookie', 
'Cruffin, Apple', 'Muffin', 'Scone', 'Danish, Rhubarb', 'Banana Bread', 'Croissant, Butter',  'Croissant, Chocolate', 'Croissant, Ham & Cheese', 'Cinnamon Bun', 'Croissant, Everything', 'Power Puck', 'Muffin, Gluten Friend', 'Croissant, Almond']
food_items = ['Sandwich, Caprese', 'Wrap, Chorizo', 'Wrap, Chicken', 'Bagel, Breakfast Meat', 'Yoghurt & Granola', 'Wrap, Ranchero', 'Wrap, Vegan Umami', 'Wrap, Vegan Lunch', 'Bagel, Breakfast Veg', 'Sandwich, Chicken', 'Vegan "Yoghurt" & Granola']
bar_drink_items = ['London Fog', 'Latte', 'Mocha', 'Americano', 'Dirty Chai', 'Iced Latte', 'Decaf', 'Chai', 'Trad. Cappuccino', 'Coffee', 'Espresso', 'Tea', "Kids' Hot Chocolate", 'Matcha Latte', 'Cappuccino', 'Iced Matcha Latte', 'Caramel Macchiato', 'Cold Brew', 
'Flat White', 'Turmeric Latte', 'Hot Choc', 'Iced Americano', 'Iced Mocha', 'Americano Misto', 'Cortado', 'Espresso Macchiato', 'Café Au Lait', 'Iced Chai', 'Shot In Dark', 'Mocha Shiver', 'Fresca Medici', 'Pink Beet Latte', 'Iced Espresso', 'Iced Matcha Tea', 
"Kids' Steamed Milk", 'Iced Tea', 'Steamed Milk', 'Vanilla Shiver','Tea Misto', 'Matcha Shiver', 'Espresso Con Panna', 'Caramel Shiver', 'Matcha Tea', 'Iced Caramel Macchiato', 'Iced Pink Beet Latte', 'SF Van. Shiver', 'Iced Turmeric Latte', 'Iced Lemon Tea', 
'Pink Beet Shiver', 'Nitro Cold Brew']
bottle_drink_items = ['San Pellegrino Pop', 'LOOP Juice', 'Glory Juice, Orange Juice', 'Kombucha', 'Flow Water', 'Glory Juice, Numbered', 'Eska Water', 'Glory Juice']
wholesale_items = ['Oatley Carton', 'Cold Brew Carton', 'Authentic Chai 1L', 'Pacific Soy Carton', 'Blue Diamond Almond']
coffee_items = ['JJ', 'Decaf Beans', 'Refisa', 'Railtown', 'Eastside', 'Puerta Verde', 'Ngaratua', 'Gatugi', 'La Providencia', 'El Tanque', 'Gititu', 'El Diamante', 'Kerinci', 'Zelaya Espresso', 'La Colina', 'La Indonesia Natural Gold Reserve', 'Chirimoya', 'La Florida', 
'Mirado', 'Mayan Harvest', 'Murundo Espresso', 'La Naranja', 'Kochere', 'Kinini Village', 'Terra Honey', 'Kaizen Espresso', 'Santana',  'Zelaya', 'Allona Natural', 'Volcán Azul', 'Christmas Blend']
miir_items = ['Miir Camp Cup, 8 oz.', 'MIIR Tumbler, 16 oz.', 'MIIR Tumbler, 12 oz.', 'MIIR Tumbler, 8 oz.', 'Miir Camp Cup, Red, 12 oz.', 'Miir Bean Container', 'MIIR Camp Cup, 16 oz.', '16oz. MIIR Tumbler, White', '12oz. MIIR Traveller, Black', 
'12oz. MIIR Tumbler, White', '16oz. MIIR Traveller, Black', '12 oz. Miir Camp Cup, Red']

add_on_items = ['…Xtra Shot(S)', '…DOUBLE CUP', '*To Go Cup', 'Bottle Deposit', '…Breve', '…Add Steamed Milk', '...Oat Milk', 'Extra Scoop Granola']
ignored_items = ['Mugshare Deposit', 'Eggnog Latte', 'Mugshare Return', 'Paper Handle Bag', '10 Tea Bags', '*To Go Cup', 'Custom Amount', '4 Cup Colombia Press', '100 Tea Bags', 'Last Chance Discount Muffin', 'Extra Tea Bag', 'Keepcup Glass', '1 Tea Bag', 'Keepcup Plastic', 'Ukraine Donation', 'OLD GIFT CARDS']


############################################### 'public' functions/classes for use in main

def update_clean_data():
    df = get_cleaned_augmented_df_from_raw_csv()
    fname = Path('..', 'data', 'augmented_daily_sales_by_item.csv')
    df.to_csv(fname)
    print(f'>>    file saved as {fname}')

class DataManager():
    df = None

    # loads cleaned, augmented, daily sales by item df from csv
    #       requires that ../data/augmented_daily_sales_by_item.csv exists already. if it doesn't, you'll need to run 'update_clean_data' from the main menu to create it (unfortunately)
    def load_data(self):
        fname = Path('..', 'data', 'augmented_daily_sales_by_item.csv')
        df = pd.read_csv(fname)

    ## TODO: implement a function that allows me to create a bootstrapped dataset of any size from augmented_daily_sales_by_item, and OPTIONALLY export it to its own csv

############################################## helpers for public functions/classes

__ = ">    "
def clean(df_raw):
    # filter out any voided sales
    df_raw = df_raw[~(df_raw['Item'].str.contains('void', case=False))]
    # group all drinks of same type together (ignore labels like ' (To Go)', ' (Own Cup)', etc...)
    df_raw = df_raw.replace(to_replace=' \(.{0,20}\)?', value="", regex=True)
    # remove all ignored items
    df_raw = df_raw[~df_raw['Item'].isin(ignored_items)]
    # remove all add ons
    df_raw = df_raw[~df_raw['Item'].isin(add_on_items)]
    # filter out cambro sales - not enough data on these to use for model; sold too infrequently
    df_raw = df_raw[~df_raw['Item'].str.contains('cambro', case=False) & ~df_raw['Item'].str.contains('carrier', case=False)]

    bakery_ind = df_raw['Item'].isin(bakery_items)
    food_ind = df_raw['Item'].isin(food_items)
    bar_drink_ind = df_raw['Item'].isin(bar_drink_items)
    bottle_drink_ind = df_raw['Item'].isin(bottle_drink_items)    
    wholesale_ind = df_raw['Item'].isin(wholesale_items)
    coffee_ind = df_raw['Item'].isin(coffee_items)
    miir_ind = df_raw['Item'].str.contains('miir', case=False)

    get = (bakery_ind | miir_ind | food_ind | bar_drink_ind | bottle_drink_ind | wholesale_ind | coffee_ind)
    df_clean = df_raw[get]

    uncategorized = list(df_raw[~get]['Item'].unique())
    if len(uncategorized) != 0:
        print('WARNING: please update item lists in data_manager.py, as the following items were not explicitly categorized during data cleaning: ')
        print(list(uncategorized))

    return df_clean

def sum_daily(df_clean):
    items = list(df_clean['Item'].unique())
    days = list(df_clean['Date'].unique())

    ndays = len(days)
    nitems = len(items)
    print(f'{__}summing sales for {nitems} items for {ndays} days... (this may take a minute)')

    day_item_arr = np.zeros((ndays, nitems))
    for i in range(ndays):
        date_a = df_clean[(df_clean['Date'] == days[i])]
        for j in range(nitems):
            day_item_arr[i, j] = len(date_a[(date_a['Item'] == items[j])])
        # print(f'{__}done with sums for {days[i]}')

    ret =  pd.DataFrame(day_item_arr, columns=items)
    ret.insert(0, 'Date', days)
    return ret

def get_cleaned_augmented_df_from_raw_csv():
        fname = Path("..", "data", "items-2022-03-03-2023-01-31.csv")
        print(f'{__}loading raw sales data from {fname}')
        raw_data_csv = pd.read_csv(fname, low_memory=False)

        # get columns that are being used for model, namely Date, Item, and Qty, and turn them into a DataFrame for ease of use
        df_raw = pd.DataFrame(raw_data_csv, columns = ['Date', 'Item', 'Qty'])

        # clean data (remove add-ons, ignored items, and cambro/carrier sales; simplify drink labels)
        print(f'{__}cleaning data...')
        df_clean = clean(df_raw)

        # use clean data to create dataframe that stores daily sales by item, as well as augmented date data for modeling
        print(f'{__}summing data across days and getting augmented date data...')
        df_daily_sums = sum_daily(df_clean)

        date_data =  get_augmented_date_a(pd.DataFrame(df_daily_sums['Date'], columns = ['Date']))

        df_aug = pd.concat([date_data, df_daily_sums.loc[:, df_daily_sums.columns != 'Date']], axis=1)

        return df_aug
