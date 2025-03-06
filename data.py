import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from skrub import GapEncoder, TextEncoder
from feature_engine.selection import DropConstantFeatures
from feature_engine.imputation import MeanMedianImputer
from feature_engine.scaling import MeanNormalizationScaler
from sklearn.model_selection import train_test_split
from feature_engine.timeseries.forecasting import LagFeatures
import numpy as np

def loadDataBase(dataset, encoder=None):
    print(dataset.columns)  
    base = pd.DataFrame()
    base.loc[:,'onpromotion'] = dataset["onpromotion"]

    oil = pd.read_csv('C:\\Users\\Danie\\Desktop\\work\\kaggle\\store-sales\\data\\oil.csv')
    oil = pd.merge(dataset, oil, on='date', join='left')
    base.loc[:,'dcoilwtico'] = oil['dcoilwtico']
    
    dataset['store_nbr_date'] = dataset['store_nbr'].astype(str) + dataset['date'].astype(str)
    dataset['date2'] = pd.to_datetime(dataset['date'], format="%Y-%m-%d")
    base.loc[:,'month'] = dataset['date2'].dt.month
    base.loc[:,'day'] = dataset['date2'].dt.day
    base.loc[:,'year'] = dataset['date2'].dt.year
    base.loc[:,'is_weekend'] = dataset['date2'].dt.weekday >= 5
    base.loc[:,'weekday'] = dataset['date2'].dt.weekday
    baseDay = pd.to_datetime('2012-01-01', format="%Y-%m-%d")
    base.loc[:, 'day_count'] = (dataset['date2'] - baseDay).dt.days

    base.loc[:,'family__isFood'] = dataset['family'].map({
        'AUTOMOTIVE': 0,
        'BABY CARE': 0,
        'BEAUTY': 0,
        'BEVERAGES': 1,
        'BOOKS': 0,
        'BREAD/BAKERY': 1,
        'CELEBRATION': 0,
        'CLEANING': 0,
        'DAIRY': 1,
        'DELI': 1,
        'EGGS': 1,
        'FROZEN FOODS': 1,
        'GROCERY I' : 1,
        'GROCERY II' : 1,
        'HARDWARE': 0,
        'HOME AND KITCHEN I': 0,
        'HOME AND KITCHEN II': 0,
        'HOME APPLIANCES': 0,
        'HOME CARE': 0,
        'LADIESWEAR': 0,
        'LAWN AND GARDEN': 0,
        'LINGERIE': 0,
        'LIQUOR,WINE,BEER': 1,
        'MAGAZINES': 0,
        'MEATS': 1,
        'PERSONAL CARE': 0,
        'PET SUPPLIES': 0,
        'PLAYERS AND ELECTRONICS': 0,
        'POULTRY': 1,
        'PREPARED FOODS': 1,
        'PRODUCE': 1,
        'SCHOOL AND OFFICE SUPPLIES': 0,
        'SEAFOOD': 1
    })

    base.loc[:,'family__Home&Living'] = dataset['family'].map({
        'AUTOMOTIVE': 0,
        'BABY CARE': 0,
        'BEAUTY': 0,
        'BEVERAGES': 0,
        'BOOKS': 0,
        'BREAD/BAKERY': 0,
        'CELEBRATION': 0,
        'CLEANING': 1,
        'DAIRY': 0,
        'DELI': 0,
        'EGGS': 0,
        'FROZEN FOODS': 0,
        'GROCERY I' : 0,
        'GROCERY II' : 0,
        'HARDWARE': 1,
        'HOME AND KITCHEN I': 1,
        'HOME AND KITCHEN II': 1,
        'HOME APPLIANCES': 1,
        'HOME CARE': 1,
        'LADIESWEAR': 0,
        'LAWN AND GARDEN': 1,
        'LINGERIE': 0,
        'LIQUOR,WINE,BEER': 0,
        'MAGAZINES': 0,
        'MEATS': 0,
        'PERSONAL CARE': 0,
        'PET SUPPLIES': 0,
        'PLAYERS AND ELECTRONICS': 0,
        'POULTRY': 0,
        'PREPARED FOODS': 0,
        'PRODUCE': 0,
        'SCHOOL AND OFFICE SUPPLIES': 0,
        'SEAFOOD': 0
    })

    base.loc[:,'family__PersonalCare'] = dataset['family'].map({
        'AUTOMOTIVE': 0,
        'BABY CARE': 0,
        'BEAUTY': 1,
        'BEVERAGES': 0,
        'BOOKS': 0,
        'BREAD/BAKERY': 0,
        'CELEBRATION': 0,
        'CLEANING': 0,
        'DAIRY': 0,
        'DELI': 0,
        'EGGS': 0,
        'FROZEN FOODS': 0,
        'GROCERY I' : 0,
        'GROCERY II' : 0,
        'HARDWARE': 0,
        'HOME AND KITCHEN I': 0,
        'HOME AND KITCHEN II': 0,
        'HOME APPLIANCES': 0,
        'HOME CARE': 0,
        'LADIESWEAR': 1,
        'LAWN AND GARDEN': 0,
        'LINGERIE': 1,
        'LIQUOR,WINE,BEER': 0,
        'MAGAZINES': 0,
        'MEATS': 0,
        'PERSONAL CARE': 1,
        'PET SUPPLIES': 0,
        'PLAYERS AND ELECTRONICS': 0,
        'POULTRY': 0,
        'PREPARED FOODS': 0,
        'PRODUCE': 0,
        'SCHOOL AND OFFICE SUPPLIES': 0,
        'SEAFOOD': 0
    })

    base.loc[:,'family__Entertainment'] = dataset['family'].map({
        'AUTOMOTIVE': 0,
        'BABY CARE': 0,
        'BEAUTY': 0,
        'BEVERAGES': 0,
        'BOOKS': 1,
        'BREAD/BAKERY': 0,
        'CELEBRATION': 1,
        'CLEANING': 0,
        'DAIRY': 0,
        'DELI': 0,
        'EGGS': 0,
        'FROZEN FOODS': 0,
        'GROCERY I' : 0,
        'GROCERY II' : 0,
        'HARDWARE': 0,
        'HOME AND KITCHEN I': 0,
        'HOME AND KITCHEN II': 0,
        'HOME APPLIANCES': 0,
        'HOME CARE': 0,
        'LADIESWEAR': 0,
        'LAWN AND GARDEN': 0,
        'LINGERIE': 0,
        'LIQUOR,WINE,BEER': 1,
        'MAGAZINES': 1,
        'MEATS': 0,
        'PERSONAL CARE': 0,
        'PET SUPPLIES': 0,
        'PLAYERS AND ELECTRONICS': 1,
        'POULTRY': 0,
        'PREPARED FOODS': 0,
        'PRODUCE': 0,
        'SCHOOL AND OFFICE SUPPLIES': 0,
        'SEAFOOD': 0
    })

    base.loc[:,'family__Stationery'] = dataset['family'].map({
        'AUTOMOTIVE': 0,
        'BABY CARE': 0,
        'BEAUTY': 0,
        'BEVERAGES': 0,
        'BOOKS': 1,
        'BREAD/BAKERY': 0,
        'CELEBRATION': 0,
        'CLEANING': 0,
        'DAIRY': 0,
        'DELI': 0,
        'EGGS': 0,
        'FROZEN FOODS': 0,
        'GROCERY I' : 0,
        'GROCERY II' : 0,
        'HARDWARE': 0,
        'HOME AND KITCHEN I': 0,
        'HOME AND KITCHEN II': 0,
        'HOME APPLIANCES': 0,
        'HOME CARE': 0,
        'LADIESWEAR': 0,
        'LAWN AND GARDEN': 0,
        'LINGERIE': 0,
        'LIQUOR,WINE,BEER': 0,
        'MAGAZINES': 1,
        'MEATS': 0,
        'PERSONAL CARE': 0,
        'PET SUPPLIES': 0,
        'PLAYERS AND ELECTRONICS': 0,
        'POULTRY': 0,
        'PREPARED FOODS': 0,
        'PRODUCE': 0,
        'SCHOOL AND OFFICE SUPPLIES': 1,
        'SEAFOOD': 0
    })

    base.loc[:,'family'] = dataset.family
    base = pd.get_dummies(base, columns=["family"], prefix="family")

    stores = pd.read_csv('C:\\Users\\Danie\\Desktop\\work\\kaggle\\store-sales\\data\\stores.csv')
    stores = pd.merge(dataset, stores, on='store_nbr')
    base = pd.concat([base, 
                      pd.get_dummies(stores[['type', 'cluster']], columns=['type', 'cluster'])],
                      axis=1)

    transactions = pd.read_csv('C:\\Users\\Danie\\Desktop\\work\\kaggle\\store-sales\\data\\transactions.csv')
    transactions.loc[:,'store_nbr_date'] = transactions['store_nbr'].astype(str) + transactions['date'].astype(str)
    transactions = pd.merge(dataset, transactions, on=['store_nbr_date'])
    base.loc[:,'transactions'] = transactions['transactions']

    holiday = pd.read_csv('C:\\Users\\Danie\\Desktop\\work\\kaggle\\store-sales\\data\\holidays_events.csv')
    holiday = pd.merge(stores, holiday, on='date')
    base.loc[:,'holiday_Holiday'] = (holiday['type_y'] == 'Holiday') & ((holiday['locale'] == 'National') | 
        ((holiday['locale'] == 'Local') & (holiday['city'] == holiday['locale_name'])) | 
        ((holiday['locale'] == 'Regional') & (holiday['state'] == holiday['locale_name'])))
    base.loc[:,'holiday_Event'] = (holiday['type_y'] == 'Event') & ((holiday['locale'] == 'National') | 
        ((holiday['locale'] == 'Local') & (holiday['city'] == holiday['locale_name'])) | 
        ((holiday['locale'] == 'Regional') & (holiday['state'] == holiday['locale_name'])))
    base.loc[:,'holiday_Additional'] = (holiday['type_y'] == 'Additional') & ((holiday['locale'] == 'National') | 
        ((holiday['locale'] == 'Local') & (holiday['city'] == holiday['locale_name'])) | 
        ((holiday['locale'] == 'Regional') & (holiday['state'] == holiday['locale_name'])))
    base.loc[:,'holiday_IsTransferred'] = (holiday['transferred'] == 'True') & ((holiday['locale'] == 'National') | 
        ((holiday['locale'] == 'Local') & (holiday['city'] == holiday['locale_name'])) | 
        ((holiday['locale'] == 'Regional') & (holiday['state'] == holiday['locale_name'])))
    base.loc[:,'holiday_Other'] = ((holiday['locale'] == 'National') | 
        ((holiday['locale'] == 'Local') & (holiday['city'] == holiday['locale_name'])) | 
        ((holiday['locale'] == 'Regional') & (holiday['state'] == holiday['locale_name'])))

    base.loc[:,'onpromotion_WholeStore'] = dataset.groupby(['date', 'store_nbr'])['onpromotion'].transform('sum')
    base.loc[:,'onpromotion_AllStore'] =  dataset.groupby(['date'])['onpromotion'].transform('sum')
    base.loc[:,'onpromotion_AllCluster'] =  stores.groupby(['date', 'cluster'])['onpromotion'].transform('sum')
    
    if encoder is None:
        encoder = TextEncoder(model_name='sentence-transformers/all-MiniLM-L6-v2', n_components=5)
        encoder.fit(pd.read_csv('C:\\Users\\Danie\\Desktop\\work\\kaggle\\store-sales\\data\\holidays_events.csv').description)
    transformed = encoder.transform(holiday['description']).reset_index(drop=True)
    base = pd.concat([base, transformed],axis=1)

    base = base + 0
    return base, encoder

def preporess_data():
    print('started')
    train_dataset = pd.read_csv('C:\\Users\\Danie\\Desktop\\work\\kaggle\\store-sales\\data\\train.csv')
    train, encoder = loadDataBase(train_dataset)
    print('step 1')
    test, encoder = loadDataBase(pd.read_csv('C:\\Users\\Danie\\Desktop\\work\\kaggle\\store-sales\\data\\test.csv'), encoder)
    print('step 2')

    # temp = DropCorrelatedFeatures()
    # temp.fit(train)
    # train = temp.transform(train)
    # test = temp.transform(test)

    temp = MeanMedianImputer()
    temp.fit(train)
    train = temp.transform(train)
    test = temp.transform(test)
    train = train.fillna(0)
    test = test.fillna(0)

    temp = DropConstantFeatures()
    temp.fit(train)
    train = temp.transform(train)
    test = temp.transform(test)


    # TODO add scaler
    scaler = MeanNormalizationScaler()
    scaler.fit(train)
    train = scaler.transform(train)
    test = scaler.transform(test)

    print('step 3')
    train.loc[:,'sales'] = train_dataset.sales
    train.to_csv('C:\\Users\\Danie\\Desktop\\work\\kaggle\\store-sales\\data\\train_preporess.zip', compression={'method': 'zip', 'compresslevel': 9})
    print('step 4')
    test.to_csv('C:\\Users\\Danie\\Desktop\\work\\kaggle\\store-sales\\data\\test_preporess.zip', compression={'method': 'zip', 'compresslevel': 9})
    print('all done')

def loadTrain():
    
    data = pd.read_csv('C:\\Users\\Danie\\Desktop\\work\\kaggle\\store-sales\\data\\train_preporess.zip');
    y = data.sales
    x = data.drop(columns=['sales'])



    y_train = y[x.day_count <= x.day_count[9 * (len(x) // 10)]]
    x_train = x[x.day_count <= x.day_count[9 * (len(x) // 10)]]

    y_test = y[x.day_count > x.day_count[9 * (len(x) // 10)]]
    x_test = x[x.day_count > x.day_count[9 * (len(x) // 10)]]


    return x_train, x_test, y_train, y_test

def breakIntoParts(data):
    out = []
    for store in data.store_nbr.unique():
        for family in data.family.unique():
            part = data[(data.store_nbr == store) and (data.family == family)]
            out.append(part)
    return out

def loadDataSections(df,encoder = None):
    store_dfs = {}
    oil = pd.read_csv('C:\\Users\\Danie\\Desktop\\work\\kaggle\\store-sales\\data\\oil.csv')
    stores = pd.read_csv('C:\\Users\\Danie\\Desktop\\work\\kaggle\\store-sales\\data\\stores.csv')
    transactions = pd.read_csv('C:\\Users\\Danie\\Desktop\\work\\kaggle\\store-sales\\data\\transactions.csv')

    total_promotions_all = df.groupby('date')['onpromotion'].sum().reset_index()
    total_promotions_all.rename(columns={'onpromotion': 'total_promotions_all'}, inplace=True)

    df_with_cluster_city = df.merge(stores[['store_nbr', 'state', 'city', 'cluster']], on='store_nbr', how='left')
    city_promotions = df_with_cluster_city.groupby(['date', 'state', 'city'])['onpromotion'].mean().reset_index()
    city_promotions.rename(columns={'onpromotion': 'total_promotions_city'}, inplace=True)

    cluster_promotions = df_with_cluster_city.groupby(['date', 'cluster'])['onpromotion'].mean().reset_index()
    cluster_promotions.rename(columns={'onpromotion': 'total_promotions_cluster'}, inplace=True)

    all_families = df['family'].unique()

    holidays = pd.read_csv('C:\\Users\\Danie\\Desktop\\work\\kaggle\\store-sales\\data\\holidays_events.csv')
    if encoder == None:
        encoder = TextEncoder(model_name='sentence-transformers/all-MiniLM-L6-v2', n_components=5) # go for a low dim because 
        encoder.fit(holidays.description)
    row_count = None
    for store in df['store_nbr'].unique():
        # Filter rows for the current store
        store_df = df[df['store_nbr'] == store]
        store_df['family'] = pd.Categorical(store_df['family'], categories=all_families)

        # Pivot the table: 
        #   - index: 'date'
        #   - columns: 'family'
        #   - values: both 'onpromotion' and 'sales'
        # Here, we use an aggregation function 'first' assuming there is only one record per date and family.
        pivoted = store_df.pivot_table(
            index='date',
            columns='family',
            values=['onpromotion'],
            aggfunc='first'
        )
        pivoted = pivoted.fillna(0)
        
        if row_count != None and row_count != pivoted.shape[1]:
            raise Exception('wrong shape: {row_count} =/= {pivoted.shape[1]}')
        row_count = pivoted.shape[1]
        # Flatten the multi-level columns.
        # This will create column names like "beverages_onpromotion" and "beverages_sales"
        pivoted.columns = [f"{fam}_{col}" for col, fam in pivoted.columns]
        
        pivoted['total_onpromotion'] = pivoted.filter(like='_onpromotion').sum(axis=1)

        # Reset index to turn 'date' back into a column
        pivoted = pivoted.reset_index()

        pivoted = pd.merge(pivoted, total_promotions_all, on='date', how='left')

        pivoted['date2'] = pd.to_datetime(pivoted['date'], format="%Y-%m-%d")
        pivoted.loc[:,'month'] = pivoted['date2'].dt.month
        pivoted.loc[:,'day'] = pivoted['date2'].dt.day
        pivoted.loc[:,'year'] = pivoted['date2'].dt.year
        pivoted.loc[:,'is_weekend'] = pivoted['date2'].dt.weekday >= 5
        pivoted.loc[:,'weekday'] = pivoted['date2'].dt.weekday
        baseDay = pd.to_datetime('2012-01-01', format="%Y-%m-%d")
        pivoted.loc[:, 'day_count'] = (pivoted['date2'] - baseDay).dt.days
        pivoted = pivoted.drop('date2', axis=1)


        storeType = stores.loc[stores['store_nbr'] == store, 'type']
        for char in ['A','B','C','D','E']:
            pivoted.loc[:, 'store_type_' + char] = storeType == char



        
        temp = pd.merge(pivoted, oil, on='date', how='left')
        pivoted.loc[:,'dcoilwtico'] = temp['dcoilwtico']
        
        temp = transactions[transactions.store_nbr == store]
        temp = pd.merge(pivoted, temp, on='date', how='left')
        pivoted.loc[:,'transactions'] = temp['transactions']

        storeCity = stores.loc[stores['store_nbr'] == store, 'city'].iloc[0]
        storeState = stores.loc[stores['store_nbr'] == store, 'state'].iloc[0]
        temp = holidays[(holidays['locale'] == 'National') |
                            ((holidays['locale'] == 'Regional') & (holidays['locale_name'] == storeState)) |
                            ((holidays['locale'] == 'Local') & (holidays['locale_name'] == storeCity))
                            ]
        
        pivoted.loc[:,'is_Any_event'] = pivoted.date.isin(temp.date)
        pivoted.loc[:,'is_Holiday_event'] = pivoted.date.isin(temp[temp.type == 'Holiday'].date)
        pivoted.loc[:,'is_Event_event'] = pivoted.date.isin(temp[temp.type == 'Event'].date)
        pivoted.loc[:,'is_Additional_event'] = pivoted.date.isin(temp[temp.type == 'Additional'].date)
        
        temp = temp.groupby('date')['description'].agg(lambda x: ', '.join(x)).reset_index()
        temp = pivoted.merge(temp, on='date', how='left')
        pivoted = pd.concat([pivoted, encoder.transform(temp.description)],axis=1) 

        city_promotions_subset = city_promotions[(city_promotions.city == storeCity) & (city_promotions.state == storeState)]
        temp = pd.merge(pivoted, city_promotions_subset, on='date', how='left')
        pivoted.loc[:,'total_promotions_city'] = temp.total_promotions_city

        storeCluster = stores.loc[stores['store_nbr'] == store, 'cluster'].iloc[0]
        cluster_promotions_subset = cluster_promotions[cluster_promotions.cluster == storeCluster]
        temp = pd.merge(pivoted, cluster_promotions_subset, on='date', how='left')
        pivoted.loc[:,'total_promotions_cluster'] = temp.total_promotions_cluster

        # Store the pivoted DataFrame in the dictionary using the store number as key
        pivoted = pivoted.drop('date', axis=1)

        pivoted = pivoted.apply(pd.to_numeric)

        store_dfs[store] = pivoted

    return store_dfs, encoder

def loadDataSectionsY(df):
    all_families = df['family'].unique()  
    store_dfs = {}
    for store in df['store_nbr'].unique():
        store_df = df[df['store_nbr'] == store]
        store_df['family'] = pd.Categorical(store_df['family'], categories=all_families)
        pivoted = store_df.pivot_table(
            index='date',
            columns='family',
            values='sales',
            aggfunc='first'
        )

        pivoted = pivoted.reindex(columns=all_families)

        pivoted = pivoted.reset_index(drop=True)
        print(pivoted)
        pivoted = pivoted.apply(pd.to_numeric)
        print(pivoted)
        store_dfs[store] = pivoted
    return store_dfs

def combine(dataframes: dict):
    return pd.concat([dataframes[store] for store in dataframes.keys()])

def transform_dict(dataframes: dict, transformer):
    out = dict()
    for store in dataframes.keys():
        out[store] = transformer.transform(dataframes[store])
    return out

import pickle

def preporess_data_sections():
    print('start')
    train_df = pd.read_csv('C:\\Users\\Danie\\Desktop\\work\\kaggle\\store-sales\\data\\train.csv')
    train, encoder = loadDataSections(train_df)
    test, _ = loadDataSections(pd.read_csv('C:\\Users\\Danie\\Desktop\\work\\kaggle\\store-sales\\data\\test.csv'),encoder)
    train_combined = combine(train)
    
    print('step 1')
    temp = MeanMedianImputer()
    train_combined = temp.fit_transform(train_combined)
    train = transform_dict(train, temp)
    test = transform_dict(test, temp)
    for store in train.keys():
        train[store] = train[store].apply(pd.to_numeric, errors='coerce')
        test[store] = test[store].apply(pd.to_numeric, errors='coerce')
        bool_cols = train[store].select_dtypes(include='bool').columns
        train[store][bool_cols] = train[store][bool_cols].astype(int)
        test[store][bool_cols] = test[store][bool_cols].astype(int)

        train[store] = train[store].fillna(0)
        test[store] = test[store].fillna(0)

        if train[store].select_dtypes(include=[np.number]).shape != train[store].shape:
            print(train[store].columns[train[store].isna().any()].tolist())
            print(f'failed for {store} x')
    
    train_combined = train_combined.apply(pd.to_numeric, errors='coerce')
    train_combined = train_combined.fillna(0)
    
    print('step 2')
    temp = DropConstantFeatures()
    train_combined = temp.fit_transform(train_combined)
    train = transform_dict(train, temp)
    test = transform_dict(test, temp)

    print('step 3')
    temp = MeanNormalizationScaler()
    train_combined = temp.fit_transform(train_combined)
    train = transform_dict(train, temp)
    test = transform_dict(test, temp)
    print('step 4')
    
    y = loadDataSectionsY(train_df)


    print('step 5')
    
    print(train_combined.shape)
    pickle.dump(test, open('C:\\Users\\Danie\\Desktop\\work\\kaggle\\store-sales\\data\\test.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump((train, y), open('C:\\Users\\Danie\\Desktop\\work\\kaggle\\store-sales\\data\\train.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    

def split_dict(dfs: dict, split=0.9):
    train = {}
    test = {}
    for store in dfs.keys():
        df = dfs[store]
        split_index = int(len(df) * split)

        train_set = df.iloc[:split_index]  # First 90% for training
        test_set = df.iloc[split_index:]   # Last 10% for testing
        train[store] = train_set
        test[store] = test_set
    return train, test



def load_train_sections():
    print('load')
    x, y = pickle.load(open('C:\\Users\\Danie\\Desktop\\work\\kaggle\\store-sales\\data\\train.pkl', 'rb'))
    
    
    print('step 2')
    x_train, x_test = split_dict(x)
    y_train, y_test = split_dict(y)

    

    return x_train, y_train, x_test, y_test



def load_train_sections_combined(lags=None):
    x, y = pickle.load(open('C:\\Users\\Danie\\Desktop\\work\\kaggle\\store-sales\\data\\train.pkl', 'rb'))
    
    if lags != None:
        lf = LagFeatures(periods=lags)
        for store in x.keys():
            x[store] = lf.fit_transform(x[store])
        
        inputer = MeanMedianImputer()
        inputer.fit(combine(x))
        for store in x.keys():
            x[store] = inputer.transform(x[store])

        dc = DropConstantFeatures()
        dc.fit(combine(x))
        for store in x.keys():
            x[store] = dc.transform(x[store])
    
    x_train, x_test = split_dict(x)
    y_train, y_test = split_dict(y)

    x_train = combine(x_train)
    x_test = combine(x_test)

    y_train = combine(y_train)
    y_test = combine(y_test)

    return x_train, x_test, y_train, y_test


import matplotlib.pyplot as plt

def getHistagram():
    data = pd.read_csv('C:\\Users\\Danie\\Desktop\\work\\kaggle\\store-sales\\data\\holidays_events.csv')

    # Plot a histogram for a specific column (e.g., 'LotArea')
    column_name = "type"  # Replace with your desired column

    value_counts = data[column_name].value_counts()
    filtered_values = value_counts[value_counts < 150].index
    filtered_data = data[data[column_name].isin(filtered_values)]

    plt.hist(data[column_name].dropna(), bins=20, color='blue', edgecolor='black')


    # Add labels and title
    plt.xlabel(column_name)
    plt.ylabel('Frequency')
    plt.title(f'Histogram of {column_name}')

    # Show the plot
    plt.show()

if __name__ == '__main__':
    # TODO add detrend into this
    #preporess_data_sections()
    preporess_data_sections()
