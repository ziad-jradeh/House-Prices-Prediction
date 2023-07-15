import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import cross_val_score

# get_description() is a function to get the description of any column name or a value.
# It takes the column/value name, or any list of them:
from data_description import get_description




def feat_eng(df, standardize = False, normalize = False):
    '''
    Feature Engineering function.
    standardize: bool, default False, If True, the dataset is scaled with RobustScaler.
    Normalize: bool, default False, If True, the dataset distribution is normalized using log().
    Returns: The dataset as a DataFrame.
    '''
    
    
    data = df.copy()
    
    # New features:
    data['MoSold'] = data['MoSold'].replace([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])  
    data['HouseAge'] = data['YrSold'] - data['YearBuilt']
    data[data['HouseAge'] < 0] = 0
    data['LastRemodeled'] = data['YrSold'] - data['YearRemodAdd']
    data['IsRemodeled'] = np.where(data['YearRemodAdd'] != data['YearBuilt'], 1, 0)
    data['TotalBathrooms'] = data['FullBath'] + data['BsmtFullBath'] + (data['HalfBath'] + data['BsmtHalfBath']) / 2
    
    # Converting YrSold and MSSubClass into categorical features
    data['YrSold'] = data['YrSold'].astype(str)
    data['MSSubClass'] = data['MSSubClass'].astype(str)
    
    ## Converting ordinal categorical features into numerical:
    # ordinal = ['ExterQual','BsmtQual','KitchenQual','FireplaceQu','GarageQual','ExterCond', 'BsmtCond','GarageCond','HeatingQC']
    # for col in ordinal:
    #     data[col] = data[col].replace(['No_Fireplace', 'No_Garage', 'No_Bsmt', 'No_Pool'], 'None')
    #     data[col+'_num'] = data[col].replace(['Ex','Gd','TA','Fa','Po','None'], [5,4,3,2,1,0])


    exclude = ['ExterQual','BsmtQual','KitchenQual','FireplaceQu','GarageQual',
            'ExterCond', 'BsmtCond','GarageCond','HeatingQC',
            'FullBath', 'BsmtFullBath', 'HalfBath', 'BsmtHalfBath', 
            'YearRemodAdd', 'YearBuilt', 'YrSold'
            ]
    
    # data = data.drop(columns=exclude)
    
    categorical = [x for x in data.columns if data[x].dtype == 'object']
    numerical = [x for x in data.columns if (data[x].dtype != 'object') and (x != 'SalePrice')]
    
    # Normalizing the distribution of the numerical features if the skewness is larger than 0.5.
    if normalize == True:
        skewness = data[numerical].skew()
        skewd_feats = skewness[abs(skewness) > 0.5].index
        
        for col in skewd_feats:
            data[col] = np.log(data[col] + 1)
        
        data.loc[:1460, 'SalePrice'] = np.log(data['SalePrice'][:1460] + 1)
            
    
    # Standardizing the numerical features.
    if standardize == True:
        from sklearn.preprocessing import RobustScaler
        data[numerical] = RobustScaler().fit_transform(data[numerical])
        
        
    # One-hot encoding the categorical features.
    data = pd.get_dummies(data, drop_first=True)
    
    return data



def model_cv(model, X, y, n_folds = 5):
    '''
    A cross validation function to calculate the rmse and r2 scores of a model,
    using cross_val_score() function with n_folds number of folds.
    '''
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=n_folds))
    r2 = cross_val_score(model, X, y, scoring="r2", cv=n_folds)
    return rmse.mean(), r2.mean()


def fit_models(models, X, y):
    '''
    A function to fit a list of models.
    Returns: A list of fitted models.
    '''
    return [model.fit(X, y) for model in models]

def avg_predict(models, X,):
    '''
    A function to calculate the average of the predictions of a list of models.
    '''
    y_pred = np.mean([model.predict(X) for model in models], axis=0)
    return y_pred