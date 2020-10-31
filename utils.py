import pandas as pd
import locale
import numpy as np
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

'''
Transform Text columns to numeric label encoding
'''
def generate_labels():
    le = {'Name': preprocessing.LabelEncoder(),
          'City': preprocessing.LabelEncoder(),
          'State': preprocessing.LabelEncoder(),
          'Bank': preprocessing.LabelEncoder(),
          'BankState': preprocessing.LabelEncoder()}

    columns = ['Name', 'City', 'State', 'Bank', 'BankState']
    train_df = get_data(type='train', dropna=False, get_dummy=False)
    test_df = get_data(type='test', dropna=False, get_dummy=False)
    for column in columns:
        train_unique = train_df[column].unique()
        test_unique = test_df[column].unique()
        total = sorted(set(train_unique) | set(test_unique))
        le[column].fit(total)
        # df[column] = le[column].transform(df[column])
    return le


def generate_scaler(le, transformer):
    base_dropna = get_data(le=le,type='train', dropna=True, get_dummy=True, feature_split=False, values_only=True,drop_columns=[])
    base_dropna_x = base_dropna.drop(columns='ChargeOff')
    scale_columns = base_dropna_x.columns

    tf = Pipeline(steps=[
            ('transformer', transformer)])

    scaler = ColumnTransformer(
            remainder='passthrough', #passthough features not listed
            transformers=[
                ('tf', tf , scale_columns)
            ])

    scaler.fit(base_dropna_x)
    
    return scaler
                    

def get_data(scaler=None, le={}, type='train', dropna=True, get_dummy=True, feature_split=False, values_only=False, drop_columns=[]):
    
    # Load data
    if type == 'train':
        df_train = pd.read_csv("dataset/Xtrain.csv", dtype={'Zip': 'object', 'NAICS': 'object', 'NewExist': 'object',
                                                            'FranchiseCode': 'object', 'UrbanRural': 'object'}, parse_dates=['ApprovalDate', 'DisbursementDate'])
        y_train = pd.read_csv("dataset/Ytrain.csv")
        df = pd.concat([df_train, y_train['ChargeOff']], axis=1)
    else:
        df = pd.read_csv("dataset/Xtest.csv", dtype={'Zip': 'object', 'NAICS': 'object', 'NewExist': 'object',
                                                     'FranchiseCode': 'object', 'UrbanRural': 'object'}, parse_dates=['ApprovalDate', 'DisbursementDate'])

        
    df = data_preprocessing(df)
    
    # Handle NA values
    if dropna and type == 'train':
        if get_dummy:
            df = one_hot_encoding_common(df)
            df = df.dropna()
        else:
            df = df.dropna()
    else:
        df.fillna(
            {'NewExist': '-1', 'LowDoc': '-1', 'DisbursementDate': pd.Timestamp(2017, 1, 1),
             'ApprovalFY': 2017, 'Bank': '-1', 'BankState': '-1', 'Name': '-1', 'City': '-1', 'State': '-1'}, inplace=True)
        if get_dummy:
            df = one_hot_encoding_common(df)
            df = df.drop(columns=['NewExist_-1', 'LowDoc_-1'])

    # Transform continuous value to categorical
    if (feature_split):
        df = feature_transformation(df)
        
    # Transform Date, Text columns to numerical values
    if values_only:
        
        # Date to Epoch
        df['ApprovalDate'] = (df['ApprovalDate'] -
                              pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
        df['DisbursementDate'] = (
            df['DisbursementDate'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
        
        # Apply label encoding to text column
        columns = ['Name', 'City', 'State', 'Bank', 'BankState']
        for column in columns:
            df[column] = le[column].transform(df[column])
            
    # Drop columns according to passed parameters 
    df = df.drop(columns=drop_columns)
    
    # Perform data normalization
    if scaler is not None:
        if type == 'train':
            x = df.drop(columns='ChargeOff').copy()
            x_scaled = scaler.transform(x)
            x_normalized = pd.DataFrame(x_scaled, columns=x.columns)
            df = pd.concat([x_normalized.reset_index(drop=True), df['ChargeOff'].reset_index(drop=True)], axis=1)
        else:
            x_scaled = scaler.transform(df.copy())
            df = pd.DataFrame(x_scaled, columns=df.columns)
            
    return df


def data_preprocessing(df):
    # Drop BalanceGross because all values are zero
    df = df.drop(columns=['BalanceGross'])

    # Process Date
    df.loc[(df['ApprovalDate'].dt.year >= 2020),
           'ApprovalDate'] = df['ApprovalDate'] - pd.DateOffset(years=100)
    df.loc[(df['DisbursementDate'].dt.year >= 2020),
           'DisbursementDate'] = df['DisbursementDate'] - pd.DateOffset(years=100)

    # Process categorical columns for NaN
    df.loc[(df['NewExist'] != "1.0") & (
        df['NewExist'] != "2.0"), 'NewExist'] = np.NaN
    df['NewExist'] = df['NewExist'].str.strip('.0')

    df.loc[(df['FranchiseCode'] == '0') | (
        df['FranchiseCode'] == '1'), 'FranchiseCode'] = '0'
    df.loc[(df['RevLineCr'] != 'Y'), 'RevLineCr'] = 'N'
    df.loc[(df['LowDoc'] != 'Y') & (df['LowDoc'] != 'N'), 'LowDoc'] = np.NaN

    # Process Numeric
    df['DisbursementGross'] = pd.to_numeric(df['DisbursementGross'].map(
        lambda x: locale.atof(x.strip('$').replace(',', '').replace(' ', ''))), downcast='float')
    df['GrAppv'] = pd.to_numeric(df['GrAppv'].map(lambda x: locale.atof(
        x.strip('$').replace(',', '').replace(' ', ''))), downcast='float')
    df['SBA_Appv'] = pd.to_numeric(df['SBA_Appv'].map(lambda x: locale.atof(
        x.strip('$').replace(',', '').replace(' ', ''))), downcast='float')
    df['ApprovalFY'] = df.ApprovalFY.str.replace(r"\D", '')
    df['ApprovalFY'] = pd.to_numeric(df['ApprovalFY'], downcast='integer')
    df['NAICS'] = df.NAICS.str.replace(r"\D", '')
    df['NAICS'] = pd.to_numeric(df['NAICS'], downcast='integer')
    df['Zip'] = df.Zip.str.replace(r"\D", '')
    df['Zip'] = pd.to_numeric(df['Zip'], downcast='integer')
    df['FranchiseCode'] = df.FranchiseCode.str.replace(r"\D", '')
    df['FranchiseCode'] = pd.to_numeric(
        df['FranchiseCode'], downcast='integer')

    return df


def one_hot_encoding_common(df):
    NewExist_dummy = pd.get_dummies(df['NewExist'], prefix='NewExist')
    UrbanRural_dummy = pd.get_dummies(df['UrbanRural'], prefix='UrbanRural')
    RevLineCr_dummy = pd.get_dummies(df['RevLineCr'], prefix='RevLineCr')
    LowDoc_dummy = pd.get_dummies(df['LowDoc'], prefix='LowDoc')

    df = pd.concat([df, NewExist_dummy, UrbanRural_dummy,
                    RevLineCr_dummy, LowDoc_dummy], axis=1)
    df = df.drop(['NewExist', 'UrbanRural',
                  'RevLineCr', 'LowDoc', "Id"], axis=1)

    return df


def feature_transformation(df_in):
    df_feature = df_in.copy()
    inf = float('inf')
    df_feature.NoEmp = pd.cut(
        df_feature.NoEmp, bins=[-inf, 9.99, 49.99, 249.99, inf], labels=['Micro', 'Small', 'Medium', 'Large'])
    df_feature.Term = pd.cut(df_feature.Term, bins=[-inf, 0.99*12, 2.99*12, 24.99*12, inf], labels=[
                             'Short', 'Intermediate', 'Long', 'Extra Long'])
    return pd.get_dummies(df_feature, columns=['NoEmp', 'Term'])


if __name__ == '__main__':
    get_data(values_only=True, type='test')
