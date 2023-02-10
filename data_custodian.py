import pandas as pd

class DataCustodian:
    cleaned_data_df = None

    def __init__(self) -> None:
        pass

    def load_and_clean_data(self):
        raw_data_csv = pd.read_csv(r'C:\Users\bfeen\OneDrive\Desktop\coding_projects\JJ_ML_project\data\items-2022-03-03-2023-01-31.csv', low_memory=False)

        # get columns that are being used for model, namely Date, Item, and Qty, and turn them into a DataFrame for ease of use
        df_raw = pd.DataFrame(raw_data_csv, columns = ['Date', 'Item', 'Qty'])

        # filter out any voided sales
        df_non_void = df_raw[~df_raw['Item'].str.contains('void', case=False)]