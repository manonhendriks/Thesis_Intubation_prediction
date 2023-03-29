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
from sklearn.preprocessing import RobustScaler, MinMaxScaler
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

#%% ### Load in linear mixed effects model data ###
df = pd.read_csv(, delimiter=';')
print(df.head(2))

### Load in other features from the aggregated data set ###
df1 = pd.read_excel(')
print(df1.info())
df1 = df1.rename(columns={'patientnr':'Patientnr'})

### Combine the two dataframes into 1 new data set, remove aggregated values that are already in repeated measurements data ###
df2 = pd.merge(df1, df, on='Patientnr')
print(df2.head(2))
df2 = df2.drop(['Fio2 Mean','Fio2 Stdev','Fio2 Laatste Waarde', 'Fio2 Delta',
                'Perifere O2 saturatie Mean', 'Perifere O2 saturatie stdev',
                'Perifere O2 saturatie laatste waarde', 'Respiratory rate Mean',
                'Respiratory rate stdev', 'Respiratory rate laatste waarde',
                'Arteriële O2 saturatie Mean', 'Arteriële O2 saturatie stdev',
                'Arteriële O2 saturatie laatste waarde', 'Bicarbonaat Mean',
                'Bicarbonaat stdev', 'Bicarbonaat laatste waarde',
                'CO2partiële druk Mean', 'CO2partiële druk stdev',
                'CO2partiële druk laatste waarde', 'O2partiële druk Mean',
                'O2partiële druk stdev', 'O2partiële druk laatste waarde',
                'pH Mean', 'pH stdev', 'pH laatste waarde', 'Bloeddruk systolisch Mean',
                'Bloeddruk systolisch stdev', 'Bloeddruk systolisch laatste waarde',
                'Bloeddruk diastolisch Mean', 'Bloeddruk diastolisch stdev', 
                'Bloeddruk diastolisch laatste waarde', 'Hartritme Mean',
                'Hartritme stdev', 'Hartritme laatste waarde', 
                'Temperatuur axillair/inguinaal Mean', 
                'Temperatuur axillair/inguinaal laatste waarde', 'id'],axis=1)

#%%
df_NaN = df2.isna().sum(axis=0)  
sum_df_NaN = (df_NaN == 0).sum(axis=0)
df_Nan_pt = df2.isna().sum(axis=1)
# Data characteristics, number of (intubated) admissions and missing data
print(f'The number of patients: {len(df)}')
print(f'The number of features: {len(df2.columns)}')
print(f'The number of features with NaN values: {(len(df_NaN) - sum_df_NaN)}')
print(f'The number of patients with NaN values: {len(df_Nan_pt)}')
intubated_patients = (list(df2['geintubeerd'] == 1)).count(True)
print(f'The number of intubated patients (including deaths within admission):{intubated_patients}')

### Set the same threshold as aggregated dataset for dropping missing numbers
df3 = df2.dropna(axis=1, how='any', thresh=210, subset=None, inplace=False)
df3_NaN = df3.isna().sum(axis=0)
print(df3_NaN)
df2_list = list(df2.columns)
df3_list = list(df3.columns)

dropped_NaNColumns = [x for x in df2_list if not x in df3_list or df3_list.remove(x)]
print(dropped_NaNColumns) #53 column deleted because of missing numbers
### DELETED COLUMNS ###
# The columns that were deleted were NIBD (non invasive blood pressure) 
# measurements (all sys, dia, mean) and the std of Tempall because only 1 measurement often
#%% Multiple imputation with KNN k=3
print(df3.head(5))
df3 = df3.drop(['Patientnr'],axis=1)
mice_imputer = IterativeImputer(estimator=KNeighborsRegressor(n_neighbors=3), max_iter=150, random_state=0)
imputed_data = mice_imputer.fit_transform(df3)

# create multiple imputations
m = 3
imputed_data_list = []
for i in range(m):
    imputed_data = mice_imputer.transform(df3)
    imputed_data_list.append(imputed_data)

# convert the list of imputed data arrays to a list of dataframes
imputed_dfs = [pd.DataFrame(imputed_data, columns=df3.columns) for imputed_data in imputed_data_list]

      

# create histogram of 'magnesium' column in original df2
plt.hist(df3['Hemoglobine'], bins=20, alpha=0.9, label='Original')

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

#%%
### Extract validation data set from data set, 80% of data will be used in cross validation, 20% will be validation set ###

column_names = df4.columns
# create a DataFrame with the input data
df4 = pd.DataFrame(df4, columns=column_names)
# scale the data using MinMaxScaler
scaler = MinMaxScaler()
df4_scaled = scaler.fit_transform(df4)
# create a PowerTransformer object
pt = PowerTransformer()
# fit and transform the data using the PowerTransformer
transformed_data = pt.fit_transform(df4_scaled)
# create a new dataframe with the transformed data
df4_transformed = pd.DataFrame(transformed_data, columns=column_names)

X = df4_transformed.drop(['geintubeerd'],axis=1)
y = df4['geintubeerd']
y = np.ravel(y)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=df4['geintubeerd'])

print(y_test)

#%% Nested Cross-validation
# import sys
# sys.stdout = open('//userapps/data/SHAREDIR/Afstudeeronderzoek Intubatie voorspelling/Python code files/results_log.txt', 'w')

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
transformers = [PowerTransformer(method='yeo-johnson', standardize=True), RobustScaler()]
selectors = [PCA(), SelectKBest(f_classif, k=50)]


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

scaler = MinMaxScaler()
transformer = PowerTransformer(method='yeo-johnson', standardize=True)

x_train = pd.DataFrame(x_train)
x_train = x_train.reset_index(drop=True)

y_train = pd.DataFrame(y_train)
y_train = y_train.reset_index(drop=True)

y_train = np.ravel(y_train) #make it 1d array

for fold, (train_idx, test_idx) in enumerate(outer_cv.split(x_train, y_train)):
    print(f"OUTFold {fold+1}")
    x_train_outer, x_test_outer = x_train.iloc[train_idx], x_train.iloc[test_idx]
    y_train_outer, y_test_outer = y_train[train_idx], y_train[test_idx]
    
    for transformer in [None] + transformers:
        for selector in [None] + selectors:
            for name, (model, params) in models.items():
                print(f'Transformer: {transformer}; Selector: {selector}; Model: {name}')
                if transformer is not None:
                    x_train.replace([np.inf, -np.inf], np.finfo(np.float64).max, inplace=True)
                try:
                    x_train_trans = scaler.fit_transform(x_train_outer) # Apply MinMaxScaler
                    if transformer is not None:
                        x_train_trans = transformer.fit_transform(x_train_trans) # Apply PowerTransformer
                    x_test_trans = scaler.transform(x_test_outer)
                    if transformer is not None:
                        x_test_trans = transformer.transform(x_test_trans)
                        
                except ValueError:
                    print('Could not perform PowerTransformer, replacing NaN with 0')
                    x_train_outer.fillna(0, inplace=True)
                    x_train_outer.replace([np.inf, -np.inf], np.finfo(np.float64).max, inplace=True)
                    x_test_outer.fillna(0, inplace=True)
                    x_test_outer.replace([np.inf, -np.inf], np.finfo(np.float64).max, inplace=True)
                    if transformer is not None:
                        x_train_trans = transformer.fit_transform(x_train_outer)
                        x_test_trans = transformer.transform(x_test_outer)
                    else:
                        x_train_trans = x_train_outer
                        x_test_trans = x_test_outer
                
                else:
                    x_train_trans = scaler.fit_transform(x_train_outer) # Apply MinMaxScaler
                    x_test_trans = scaler.transform(x_test_outer)
                    if transformer is not None:
                        x_train_trans = transformer.fit_transform(x_train_trans) # Apply PowerTransformer
                        x_test_trans = transformer.transform(x_test_trans)

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
        
                y_train_IN = pd.DataFrame(y_train_outer)
                y_train_IN = y_train_IN.reset_index(drop=True)
        
                y_train_IN = np.ravel(y_train_IN) #make it 1d array
                
                inner_val_scores[name] = []
                inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                
                for fold, (train_idx, val_idx) in enumerate(inner_cv.split(x_train_sel, y_train_IN)):
                    print(f"Fold {fold+1}")
                    x_train_fold = x_train_sel.iloc[train_idx]
                    y_train_fold = y_train_IN[train_idx]
                    x_val_fold = x_train_sel.iloc[val_idx]
                    y_val_fold = y_train_IN[val_idx]
               
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
                        plot_learning_curve(best_model, title, x_train_sel, y_train_IN, cv=inner_cv, n_jobs=-1)
                        plt.show()

                # Report the best hyperparameters and score
                print(f'Best hyperparameters: {grid_search.best_params_}')
                print(f'Best CV score: {grid_search.best_score_:.3f}')
                print(f'{name} validation scores:', inner_val_scores[name])
                print()
                
                if name == 'Gradient Boosting':
                        feature_importances = grid_search.best_estimator_.feature_importances_
                        top_feature_indices = np.argsort(feature_importances)[::-1][:10]
                        top_features = [X.columns[i] for i in top_feature_indices]
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
                        top_features = [X.columns[i] for i in top_feature_indices]
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
                        top_features = [X.columns[i] for i in top_feature_indices]
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

#sys.stdout.close()

#%% Validation of best model with 20% test set
# train best model on all other data
# Best model is RF with Power Transformer scaling and no feature selection

model = RandomForestClassifier(max_depth= 10, min_samples_split= 5, n_estimators= 200)
model.fit(x_train,y_train)

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
    model, x_train, y_train, cv=5, train_sizes=train_sizes, scoring='roc_auc')

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
plt.title('Learning curve for best joint model: RF-no feature selection-Power Transformer')
plt.legend()
plt.show()

#%%
predicted_val = model.predict(x_test)
rf_exval_score = roc_auc_score(y_test, predicted_val)

print('AUC score RF with validation data:', rf_exval_score)
rf_exval_report = classification_report(y_test, predicted_val)
print(rf_exval_report)

