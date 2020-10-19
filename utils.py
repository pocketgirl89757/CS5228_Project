import pandas as pd
import locale
import numpy as np


def get_data(type='train', dropna=True, get_dummy=True, feature_split=False, values_only=False):
    if type == 'train':
        df_train = pd.read_csv("dataset/Xtrain.csv", dtype={'Zip': 'object', 'NAICS': 'object', 'NewExist': 'object',
                                                            'FranchiseCode': 'object', 'UrbanRural': 'object'}, parse_dates=['ApprovalDate', 'DisbursementDate'])
        y_train = pd.read_csv("dataset/Ytrain.csv")
        df = pd.concat([df_train, y_train['ChargeOff']], axis=1)
    else:
        df = pd.read_csv("dataset/Xtest.csv", dtype={'Zip': 'object', 'NAICS': 'object', 'NewExist': 'object',
                                                     'FranchiseCode': 'object', 'UrbanRural': 'object'}, parse_dates=['ApprovalDate', 'DisbursementDate'])

    df = data_preprocessing(df)
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

    if (feature_split):
        df = feature_transformation(df)
    if values_only:
        df['ApprovalDate'] = (df['ApprovalDate'] -
                              pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
        df['DisbursementDate'] = (
            df['DisbursementDate'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
        # TODO: add label encocder
        df = df.drop(columns=['Name', 'City', 'State', 'Bank', 'BankState'])

    return df


def data_preprocessing(df):
    # Drop column
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
    get_data()
    # print(dataset.train_val_df.describe())
    # print(dataset.test_df.describe())
    # print(dataset.all_df.describe())
