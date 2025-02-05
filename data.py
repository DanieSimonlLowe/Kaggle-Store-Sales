import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from skrub import GapEncoder
from feature_engine.selection import DropCorrelatedFeatures, DropConstantFeatures
from feature_engine.imputation import MeanMedianImputer
from feature_engine.scaling import MeanNormalizationScaler
from sklearn.model_selection import train_test_split

def loadDataBase(dataset, encoder=None):
    print(dataset.columns)  
    base = pd.DataFrame()
    base.loc[:,'onpromotion'] = dataset["onpromotion"]

    oil = pd.read_csv('C:\\Users\\Danie\\Desktop\\work\\kaggle\\store-sales\\data\\oil.csv')
    oil = pd.merge(dataset, oil, on='date')
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
        encoder = GapEncoder(n_components=5)
        encoder.fit(pd.read_csv('C:\\Users\\Danie\\Desktop\\work\\kaggle\\store-sales\\data\\holidays_events.csv').description)
    transformed = encoder.transform(holiday['description']).reset_index(drop=True)
    base = pd.concat([base, transformed],axis=1)

    base = base + 0
    return base, encoder

def preporess_date():
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
    preporess_date()
