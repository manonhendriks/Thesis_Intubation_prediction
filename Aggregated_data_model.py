#%% General packages ###
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import missingno as msno
import seaborn as sns
import math
import pydot
from sklearn.pipeline import Pipeline
import scipy.stats as stats
from scipy.stats import shapiro

### Processing packages ###
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import category_encoders as ce
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import PowerTransformer

### Classifier packages ###
from sklearn.linear_model import LogisticRegression, Lasso, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif, chi2
from xgboost import XGBClassifier
from xgboost import plot_tree
from sklearn.inspection import permutation_importance
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsRegressor

### Model Selection ###
from sklearn.model_selection import RandomizedSearchCV, train_test_split, learning_curve, ShuffleSplit, StratifiedKFold, GridSearchCV, KFold, cross_val_score 
from sklearn.metrics import auc, classification_report, confusion_matrix, precision_score, recall_score, accuracy_score, f1_score, roc_auc_score, roc_curve, balanced_accuracy_score

### Display DataFrames ###
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

#%% Load in data
df = pd.read_excel()
df.drop_duplicates(subset=['patientnr'], keep='first', inplace=True)
#print(sum(df.duplicated(subset=['patientnr']))) (test if there are double IDs)

#%% Data preprocessing (removing NaN, make label column) ###
# Give patients with an intubatieduur target label 1
# Determine deaths within admission to also give the intubation label
conditions = [
    (df['intubatieduur'] >= 1),
    (df['intubatieduur'] == 0),
    (df['IntubatieExtra'] == 1)
    ]
values = [1, 0, 1]
df['geintubeerd'] = np.select(conditions, values)
df = df.drop('intubatieduur',axis=1)
df = df.drop('IntubatieExtra',axis=1)

# Encode categorical 'herkomst' data using OneHotEncoder encoder 
ce_OHE = ce.OneHotEncoder(cols=['herkomst'])
df_herkomst = ce_OHE.fit_transform(df['herkomst'])
df1 = df.join(df_herkomst, how='left', lsuffix='herkomst')
df1 = df1.drop('herkomst',axis=1)
## herkomst_1 = 'eigen woonomgeving' herkomst_2 = 'uit eigen ziekenhuis' herkomst_3 = ander ziekenhuis herkomst_4 = SEH
# Categorical gender data using Label encoder 0 = Male, 1 = Female
df1['geslacht'] = df1['geslacht'].astype('category')
df1['Gen_new'] = df1['geslacht'].cat.codes
df1 = df1.drop('geslacht',axis=1)
print(df1['Gen_new'])
# OpnamedatumIC day shift/ night shift + drop time variables
df1 = df1.drop(['opnamedatum','einddatumic','ontslagdatum','optiflowstart','optifloweind','optiflowduur','intubatiestart','intubatieeind','overlijdensdatum'],axis=1)
df1['opnameictijd'] = pd.to_datetime(df1['OpnamedatumIC']).dt.time
df1['hour'] = pd.to_datetime(df1['opnameictijd'], format='%H:%M:%S').dt.hour
df1['admission day/night'] = df1['hour'].apply(lambda x: 'night' if 18 < x <= 23 or 0 <= x <= 7 else 'day')
df1 = df1.drop(['OpnamedatumIC','opnameictijd','hour'],axis=1)

df1['admission day/night'] = df1['admission day/night'].astype('category')
df1['Admission during day or night'] = df1['admission day/night'].cat.codes
df1 = df1.drop('admission day/night',axis=1)
## 0 = day, 1 = night

### Patient characteristics 
#df1.groupby('geintubeerd', as_index=False)['leeftijd'].agg(['mean','std'])
#df1.groupby('geintubeerd', as_index=False)['lengte'].agg(['mean','std'])
#df1.groupby('geintubeerd', as_index=False)['gewicht'].agg(['mean','std'])
#grouped = df1.groupby(['geintubeerd', 'geslacht']).size().reset_index(name='count')
#print(grouped)
#grouped_origin = df1.groupby(['geintubeerd', 'herkomst']).size().reset_index(name='count')
#print(grouped_origin)
#grouped_daynight = df1.groupby(['geintubeerd', 'admission day/night']).size().reset_index(name='count')
#print(grouped_daynight)
#grouped1 = df1.groupby('geintubeerd')
#count_died = grouped1['overlijdensdatum'].count()
#print(count_died)

df1 = df1.drop('patientnr',axis=1)
# How are the missing data distributed, rows and columns with their number of NaN values 
print(df1.head(5))
print(msno.bar(df1))
df_NaN = df1.isna().sum(axis=0)  
sum_df_NaN = (df_NaN == 0).sum(axis=0)
df_Nan_pt = df1.isna().sum(axis=1) 

# Data characteristics, number of (intubated) admissions and missing data
print(f'The number of patients: {len(df)}')
print(f'The number of features: {len(df1.columns)}')
print(f'The number of features with NaN values: {(len(df_NaN) - sum_df_NaN)}')
print(f'The number of patients with NaN values: {len(df_Nan_pt)}')
intubated_patients = (list(df1['geintubeerd'] == 1)).count(True)
print(f'The number of intubated patients (including deaths within admission):{intubated_patients}') # waar komen de pt die overleden zijn bij? geintubeerd overleden bij intubatie groep of niet geintubeerd overleden--> waarbij?

# Set threshold for maximum number of missing data per feature 
df2 = df1.dropna(axis=1, how='any', thresh=210, subset=None, inplace=False)
print(df2.head(20))
print(msno.bar(df2))
df1_list = list(df1.columns)
print(df1_list)
df2_list = list(df2.columns)
print(df2_list)
dropped_NaNColumns = [x for x in df1_list if not x in df2_list or df2_list.remove(x)]
print(len(dropped_NaNColumns))

#%% Multiple imputation with KNN k=3
### Data processing filling NaN values (outside cross validation) ### 

print(df2.head(5))
df2 = df2.drop(['Respiratory rate delta','Perifere O2 saturatie delta'],axis=1)
df2 = df2.drop(['Arteriële O2 saturatie delta','Bicarbonaat delta'],axis=1)
df2 = df2.drop(['CO2partiële druk delta','O2partiële druk delta'],axis=1)
df2 = df2.drop(['pH delta','Bloeddruk systolisch delta'],axis=1)
df2 = df2.drop(['Bloeddruk diastolisch delta','Hartritme delta'],axis=1)
df2 = df2.drop(['Temperatuur axillair/inguinaal delta','Kreatinine.1'],axis=1)

mice_imputer = IterativeImputer(estimator=KNeighborsRegressor(n_neighbors=3), max_iter=150, random_state=0)
imputed_data = mice_imputer.fit_transform(df2)

# create multiple imputations
m = 3
imputed_data_list = []
for i in range(m):
    imputed_data = mice_imputer.transform(df2)
    imputed_data_list.append(imputed_data)

# convert the list of imputed data arrays to a list of dataframes
imputed_dfs = [pd.DataFrame(imputed_data, columns=df2.columns) for imputed_data in imputed_data_list]

      

# create histogram of 'magnesium' column in original df2
plt.hist(df2['Hemoglobine'], bins=20, alpha=0.9, label='Original')

# create histogram of 'magnesium' column in each imputed DataFrame
for i, df in enumerate(imputed_dfs):
    plt.hist(df['Hemoglobine'], bins=20, alpha=0.1, label=f'Imputed {i+1}')

plt.legend()
plt.show()

# compare the first two imputed DataFrames for duplicity
is_equal = imputed_dfs[0].equals(imputed_dfs[1])

if is_equal:
    print("The first two imputed DataFrames are identical.")
else:
    print("The first two imputed DataFrames are not identical.")

df4 = imputed_dfs[0]
print(df4.head(5))
print(len(df4.columns))

#%% Test if imputated data is normally imputated

### Test if data is normally distributed
for col in df2.columns:
    stat, p = shapiro(df2[col])
    alpha = 0.05
    if p > alpha:
        print(f"{col}: Sample looks Gaussian (fail to reject H0)")
    else:
        print(f"{col}: Sample does not look Gaussian (reject H0)")

### Test if data is normally distributed
for col in df4.columns:
    stat, p = shapiro(df4[col])
    alpha = 0.05
    if p > alpha:
        print(f"{col}: Sample looks Gaussian (fail to reject H0)")
    else:
        print(f"{col}: Sample does not look Gaussian (reject H0)")
        
#%% Divide data set in training and testing data   
### Extract validation data set from data set, 100% of data will be used in cross validation, external validation set will be used ###
df5 = df4.drop(['geintubeerd'],axis=1)
x_train, x_test, y_train, y_test = train_test_split(df5, df4['geintubeerd'], test_size=0.2, stratify=df4['geintubeerd'])

#%% Nested-cross validation
def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    
    if axes is None:
        _, axes = plt.subplots(1, 1, figsize=(5, 5))

    axes.set_title(title)
    if ylim is not None:
        axes.set_ylim(*ylim)
    axes.set_xlabel("Training examples")
    axes.set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)

    # Plot learning curve
    axes.grid()
    axes.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color="r",
              label="Training score")
    axes.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', color="g",
              label="Cross-validation score")
    axes.legend(loc="best")

    return plt



# Define the feature transformation and selection steps
transformers = [PowerTransformer(), RobustScaler()]
selectors = [PCA(), SelectKBest(score_func=f_classif, k=50)]


# Define the hyperparameters to be tuned for each model
gb_params = {
    'n_estimators': [100, 200, 500],
    'learning_rate': [0.01, 0.1, 1],
    'max_depth': [3, 5, 10]
}

lr_params = {
    'C': [0.1, 1, 10],
    'penalty': ['l1', 'l2']
}

rf_params = {
    'n_estimators': [100, 200, 500],
    'max_depth': [3, 5, 10],
    'min_samples_split': [5, 10, 20]
}


# Define the models to be tuned
models = {
    'Gradient Boosting': (GradientBoostingClassifier(), gb_params),
    'Logistic Regression': (LogisticRegression(solver='liblinear', max_iter=10000), lr_params),
    'Random Forest': (RandomForestClassifier(), rf_params)
}

# Perform nested cross-validation
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_val_scores = {}
auc_scores = {model_type: [] for model_type in models}
best_model = None

for fold, (train_idx, test_idx) in enumerate(outer_cv.split(df5, df4['geintubeerd'])):
    print(f"OUTFold {fold+1}")
    x_train_outer, x_test_outer = df5.iloc[train_idx], df5.iloc[test_idx]
    y_train_outer, y_test_outer = df4['geintubeerd'].iloc[train_idx], df4['geintubeerd'].iloc[test_idx]

    for transformer in [None] + transformers:
            for selector in [None] + selectors:
                for name, (model, params) in models.items():
                    print(f'Transformer: {transformer}; Selector: {selector}; Model: {name}')
                    if transformer is not None:
                        x_train_trans = transformer.fit_transform(x_train_outer)
                        x_test_trans = transformer.transform(x_test_outer)
                    else:
                        x_train_trans = x_train_outer
                        x_test_trans = x_test_outer

                    if selector is not None:
                        x_train_trans = x_train_trans
                        x_test_trans = x_test_trans
                
                        x_train_sel = selector.fit_transform(x_train_trans, y_train_outer)
                        x_test_sel = selector.transform(x_test_trans)
                    else:
                        x_train_sel = x_train_trans
                        x_test_sel = x_test_trans
            
                    x_train_sel = pd.DataFrame(x_train_sel)  # convert to DataFrame
                    x_train_sel = x_train_sel.reset_index(drop=True)  # reset index
            
                    y_train = pd.DataFrame(y_train_outer)
                    y_train = y_train.reset_index(drop=True)
            
                    y_train = np.ravel(y_train) #make it 1d array
                    
                    inner_val_scores[name] = []
                    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                        
                    for fold, (train_idx, val_idx) in enumerate(inner_cv.split(x_train_sel, y_train)):
                        print(f"INFold {fold+1}")
                        x_train_fold = x_train_sel.iloc[train_idx]
                        y_train_fold = y_train[train_idx]
                        x_val_fold = x_train_sel.iloc[val_idx]
                        y_val_fold = y_train[val_idx]
                   
                        y_val_fold = np.ravel(y_val_fold)
            
                        # Create a grid search object and fit on training data
                        grid_search = GridSearchCV(model, params, cv=inner_cv, scoring='roc_auc', n_jobs=-1)
                        grid_search.fit(x_train_fold, y_train_fold)
               
                        # Evaluate the best model on the test set
                        best_model = grid_search.best_estimator_
                        y_pred = best_model.predict_proba(x_val_fold)[:, 1]
                        val_score = roc_auc_score(y_val_fold, y_pred)
                        inner_val_scores[name].append(val_score)
                        
                        
                        # Plot the learning curve
                        if best_model is not None and any(inner_val_scores[name] for name in inner_val_scores):
                            model_name = f"{transformer}-{selector}-{name}"
                            title = f"Learning Curve ({model_name})"
                            if transformer is not None:
                                title += f" with {type(transformer).__name__}"
                            if selector is not None:
                                title += f" and {type(selector).__name__}"             
                            
                            plt.title(title)
                            plot_learning_curve(best_model, title, x_train_sel, y_train, cv=inner_cv, n_jobs=-1)
                            plt.show()
          
                    # Report the best hyperparameters and score
                    # print(f'Outer CV Fold {fold+1}')
                    print(f'Best hyperparameters: {grid_search.best_params_}')
                    print(f'Best CV score: {grid_search.best_score_:.3f}')
                    print(f'{name} validation scores:', inner_val_scores[name])
                    print()
                    
                    if name == 'Gradient Boosting':
                        feature_importances = grid_search.best_estimator_.feature_importances_
                        top_feature_indices = np.argsort(feature_importances)[::-1][:10]
                        top_features = [df5.columns[i] for i in top_feature_indices]
                        print(f"Top 10 features for Gradient Boosting: {', '.join(top_features)}")
                        gb_AUC_scores = []
                        print()
                        gb_test_preds = best_model.predict(x_test_sel)
                        gb_test_score = roc_auc_score(y_test_outer, gb_test_preds)
                        gb_AUC_scores.append(gb_test_score)
                        print('AUC score GBM:', gb_AUC_scores)
                        
                        gb_test_report = classification_report(y_test_outer, gb_test_preds)
                        print('Classification_report for Gradient Boosting:')
                        print(gb_test_report)
                        print()
                        
                    elif name == 'Logistic Regression':
                        coefficients = grid_search.best_estimator_.coef_[0]
                        top_feature_indices = np.argsort(abs(coefficients))[::-1][:10]
                        top_features = [df5.columns[i] for i in top_feature_indices]
                        print(f"Top 10 features for Logistic Regression: {', '.join(top_features)}")
                        print()
                        lr_AUC_scores = []
                        lr_test_preds = best_model.predict(x_test_sel)
                        lr_test_score = roc_auc_score(y_test_outer, lr_test_preds)
                        lr_AUC_scores.append(lr_test_score)
                        print('AUC score LRM:', lr_AUC_scores)
                        print()
                        
                        lr_test_report = classification_report(y_test_outer, lr_test_preds)
                        print('Classification_report for Logistic Regression:')
                        print(lr_test_report)
                        print()
                        
                    elif name == 'Random Forest':
                        feature_importances = grid_search.best_estimator_.feature_importances_
                        top_feature_indices = np.argsort(feature_importances)[::-1][:10]
                        top_features = [df5.columns[i] for i in top_feature_indices]
                        print(f"Top 10 features for Random Forest: {', '.join(top_features)}")
                        print()
                           
                        rf_AUC_scores = []
                       
                        rf_test_preds = best_model.predict(x_test_sel)
                        rf_test_score = roc_auc_score(y_test_outer, rf_test_preds)
                        rf_AUC_scores.append(rf_test_score)
                        print('AUC score RF:', rf_test_score)
                        print()
                        rf_test_report = classification_report(y_test_outer, rf_test_preds)
                        print('Classification_report for Random Forest:')
                        print(rf_test_report)
                        print()
                    
                    auc_scores[name].append(grid_search.best_score_)
                    
# Compute mean and standard deviation of AUC scores for each model
for name in models:
    mean_auc = np.mean(auc_scores[name])
    std_auc = np.std(auc_scores[name])
    print(f"Mean AUC score for {name}: {mean_auc:.3f} (std: {std_auc:.3f})")

#%% Extract best performing model, with training skills and apply validation data set
# Random Forest with no feature selection and no scaling performed best
# df5 is the x_train of the the whole training set 
y = df4['geintubeerd']

model = RandomForestClassifier(max_depth=3, min_samples_split=20, n_estimators=500)
model.fit(df5,y)

feature_list = df4.columns
# generate a list with all the features and their feature importance from large to small
importances = model.feature_importances_
feature_importances = sorted(zip(feature_list, importances), key=lambda x: x[1], reverse=True)
for feature, importance in feature_importances:
    print(f"{feature}: {importance}")

# select the top 10 features
top_features = feature_importances[:10][::-1]
print(top_features)

##CODE for learning curve
# specify the range of training set sizes to use
train_sizes = np.linspace(0.1, 1.0, 10)

# generate the learning curve
train_sizes, train_scores, val_scores = learning_curve(
    model, df5, y, cv=5, train_sizes=train_sizes, scoring='roc_auc')

# calculate the mean and standard deviation of the training and validation scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

# plot the learning curve
plt.plot(train_sizes, train_mean, label='Training score')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
plt.plot(train_sizes, val_mean, label='Validation score')
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
plt.xlabel('Training set size')
plt.ylabel('AUC')
plt.title('Learning curve for best aggregated data set model: RF-no feature selection-no scaling')
plt.legend()
plt.show()
#%% Load in validation data
### Load in data ###
df = pd.read_excel()
df = pd.read_excel()
df = pd.read_excel()

df_val = pd.concat([#above dfs], ignore_index=True)

DubbeleIDS = df_val[df_val.duplicated('patientnr')]
print(DubbeleIDS)
df_val.drop_duplicates(subset=['patientnr'], keep='first', inplace=True)

#%% Preprocess--> give label, encode categorical data
conditions = [
    (df_val['intubatieduur'] >= 1),
    (df_val['intubatieduur'] == 0),
    (df_val['IntubatieExtra'] == 1)
    ]
values = [1, 0, 1]
df_val['geintubeerd'] = np.select(conditions, values)
df_val = df_val.drop(['intubatieduur'],axis=1)
df_val = df_val.drop(['IntubatieExtra'],axis=1)

print(df_val.head(2))
# Encode categorical 'herkomst' data using OneHotEncoder encoder 
ce_OHE = ce.OneHotEncoder(cols=['herkomst'])
dfv1 = ce_OHE.fit_transform(df_val['herkomst'])
dfv1 = df_val.join(dfv1, how='left', lsuffix='herkomst')
dfv1 = dfv1.drop(['herkomst'],axis=1)
## herkomst_1 = 'Eigen woonomgeving' herkomst_2 = 'Ander ziekenhuis' herkomst_3 = 'Uit eigen ziekenhuis' herkomst_4 = 'SEH'

# Categorical gender data using Label encoder 0 = Male, 1 = Female
dfv1['geslacht'] = dfv1['geslacht'].astype('category')
dfv1['Gen_new'] = dfv1['geslacht'].cat.codes
dfv1 = dfv1.drop(['geslacht'],axis=1)

print(dfv1.tail(4))
print(df_val['geslacht'])
# OpnamedatumIC day shift/ night shift + drop time variables
dfv1 = dfv1.drop(['opnamedatum','einddatumic','ontslagdatum','optiflowstart','optifloweind','optiflowduur','intubatiestart','intubatieeind','overlijdensdatum'],axis=1)
dfv1['opnameictijd'] = pd.to_datetime(dfv1['OpnamedatumIC']).dt.time
dfv1['hour'] = pd.to_datetime(dfv1['opnameictijd'], format='%H:%M:%S').dt.hour
dfv1['admission day/night'] = dfv1['hour'].apply(lambda x: 'night' if 18 < x <= 23 or 0 <= x <= 7 else 'day')
dfv1 = dfv1.drop(['OpnamedatumIC','opnameictijd','hour'],axis=1)

dfv1['admission day/night'] = dfv1['admission day/night'].astype('category')
dfv1['Admission during day or night'] = dfv1['admission day/night'].cat.codes
dfv1 = dfv1.drop(['admission day/night'],axis=1)

print(msno.bar(dfv1))
dfv_NaN = dfv1.isna().sum(axis=0)  
print(dfv_NaN)


#%% Make sure only the features from aggregated df4 are in dfv1
dfv2 = pd.DataFrame(dfv1, columns=df4.columns)
print(len(dfv2.columns))
print(dfv2.isna().sum(axis=0))

print(msno.bar(dfv2))
#%% Data Characteristics
# Data characteristics, number of (intubated) admissions and missing data
print(f'The number of patients: {len(df_val)}')
print(f'The number of features: {len(df_val.columns)}')
print(f'The number of features with NaN values: {(len(df_NaN) - sum_df_NaN)}')
print(f'The number of patients with NaN values: {len(df_Nan_pt)}')
intubated_patients = (list(df_val['geintubeerd'] == 1)).count(True)
print(f'The number of intubated patients (including deaths within admission):{intubated_patients}') 

### Patient characteristics 
dfv1.groupby('geintubeerd', as_index=False)['leeftijd'].agg(['mean','std'])
dfv1.groupby('geintubeerd', as_index=False)['lengte'].agg(['mean','std'])
dfv1.groupby('geintubeerd', as_index=False)['gewicht'].agg(['mean','std'])
grouped = df_val.groupby(['geintubeerd', 'geslacht']).size().reset_index(name='count')
print(grouped)
grouped_origin = df_val.groupby(['geintubeerd', 'herkomst']).size().reset_index(name='count')
print(grouped_origin)
grouped_daynight = dfv1.groupby(['geintubeerd', 'admission day/night']).size().reset_index(name='count')
print(grouped_daynight)
grouped1 = df_val.groupby('geintubeerd')
count_died = grouped1['overlijdensdatum'].count()
print(count_died)

#%% Missing data strategy 
# columns with no data, all data points are missing --> imputed with zero (Antibiotica,Steroïden,Serum albumine)
# columns with few data points imputed with KNN k=3 

dfv2['Antibiotica'] = dfv2['Antibiotica'].replace(np.nan, 0)
dfv2['Steroïden'] = dfv2['Steroïden'].replace(np.nan, 0)
dfv2['Serum albumine'] = dfv2['Serum albumine'].replace(np.nan, 0)

mice_imputer = IterativeImputer(estimator=KNeighborsRegressor(n_neighbors=3), max_iter=150, random_state=0)
imputed_data = mice_imputer.fit_transform(dfv2)

# create multiple imputations
m = 3
imputed_data_list = []
for i in range(m):
    imputed_data = mice_imputer.transform(dfv2)
    imputed_data_list.append(imputed_data)

# convert the list of imputed data arrays to a list of dataframes
imputed_dfs = [pd.DataFrame(imputed_data, columns=dfv2.columns) for imputed_data in imputed_data_list]

      

# create histogram of 'magnesium' column in original df2
plt.hist(dfv2['Hemoglobine'], bins=20, alpha=0.9, label='Original')

# create histogram of 'magnesium' column in each imputed DataFrame
for i, df in enumerate(imputed_dfs):
    plt.hist(df['Hemoglobine'], bins=20, alpha=0.1, label=f'Imputed {i+1}')

plt.legend()
plt.show()

# compare the first two imputed DataFrames for duplicity
is_equal = imputed_dfs[0].equals(imputed_dfs[2])

if is_equal:
    print("The first two imputed DataFrames are identical.")
else:
    print("The first two imputed DataFrames are not identical.")

dfv3 = imputed_dfs[0]
print(dfv3.isna().sum(axis=0))

#%% divide validation data into x and y
x_val = dfv3.drop(['geintubeerd'],axis=1)
y_val = dfv3['geintubeerd']
print(y_val)
#%% Use validation data on best model

predicted_val = model.predict(x_val)
rf_exval_score = roc_auc_score(y_val, predicted_val)

print('AUC score RF with validation data:', rf_exval_score)
rf_exval_report = classification_report(y_val, predicted_val)
print(rf_exval_report)    
