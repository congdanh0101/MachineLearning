# %%
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import joblib  # new lib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from statistics import mean
from sklearn.model_selection import KFold, train_test_split
import joblib
import investpy
import pandas_datareader.data as web
import seaborn as sns
# %%
#Do week3 em lam ve co phieu, nhung vi qua it du lieu de khai thac, nen em xin phep thay cho em doi sang du lieu ve chuyen bay o An Do

rawdataset = pd.read_csv('Clean_Dataset.csv')
rawdataset.drop(columns=['Unnamed: 0', 'flight'], inplace=True)
flight = rawdataset.dropna()
flight.head()
flight.info()
flight.describe()
# %% Histogram Airline
#Bieu do the hien cac hang hang khong o An Do
plt.figure(figsize=(20, 10))
plt1 = flight.airline.value_counts().plot(kind='bar', color='pink')
plt.title('Airline Histogram', fontsize=20)
plt1.set(xlabel='Airline', ylabel='Frequency of Airline')
# %% Histogram source city
#Bieu do the hien hanh khach thuong bat dau voi thanh pho nao
plt.figure(figsize=(20, 10))
plt.title('Source Histogram', fontsize=20)
plt1 = flight['source_city'].value_counts().plot(kind='bar')
plt1.set(xlabel='Source city', ylabel='Frequency of Source City')

# %% Histogram destination city
#Bieu do the hien hanh khach den thanh pho nao
plt.figure(figsize=(20, 10))
plt.title('Destination Histogram', fontsize=20)
plt1 = flight['destination_city'].value_counts().plot(kind='bar')
plt1.set(xlabel='Destination city', ylabel='Frequency of Destination City')
# %% Histogram departure time
#Bieu do the hien thoi gian bat dau cua cac chuyen bay
plt.figure(figsize=(20, 10))
plt.title('Departure time histogram', fontsize=20)
plt1 = flight['departure_time'].value_counts().plot(kind='bar')
plt1.set(xlabel='Departure time', ylabel='Frequency of Departure time')

# %% Histogram arrival time
#Bieu do the hien thoi gian den noi cua cac chuyen bay
plt.figure(figsize=(20, 10))
plt.title('Arrival time histogram', fontsize=20)
plt1 = flight['arrival_time'].value_counts().plot(kind='bar')
plt1.set(xlabel='Arrival time', ylabel='Frequency of Arrival time')

# %%
#Bieu do the hien ve hang hanh khach (o day la hang thuong gia va hang pho thong)
plt.figure(figsize=(20, 10))
plt.title('Class Histogram', fontsize=20)
sns.countplot(flight['class'], palette=('cubehelix'))
# %%
plt.figure(figsize=(20, 10))
plt.title('Class vs Price', fontsize=20)
sns.boxplot(x=flight['class'], y=flight['price'], palette=('cubehelix'))
# %%
plt.figure(figsize=(20, 10))
plt1 = flight.groupby('airline')['price'].mean(
).sort_values(ascending=False).plot(kind='bar')
plt.title('Airline Price Average', fontsize=20)
plt1.set(xlabel='Airline', ylabel='Price Average')

# %%
plt.figure(figsize=(10, 8))
plt.title('Flight Price Spread', fontsize=20)
sns.boxplot(y=flight.price)
# %%
plt.figure(figsize=(20, 10))
plt.title('Flight Price Distribution Plot')
sns.distplot(flight.price)
# %%
gb1 = flight.groupby(['source_city', 'destination_city', 'days_left'])[
    'price'].mean()
gb1

# %%
#Do destination, Source, Departure, Arrival thuong co cung du lieu nen ta se thay doi mot chut
flight['departure_time'] = 'dept' + flight['departure_time']
flight['arrival_time'] = 'arr' + flight['arrival_time']
flight['source_city'] = 'src' + flight['source_city']
flight['destination_city'] = 'dest' + flight['destination_city']
# %%
#O day ta se tach flight thanh 2 tap hop gom train va test
train_set, test_set = train_test_split(flight, test_size=0.2, random_state=42)
#Sau do ta tach label price
train_set_labels = train_set['price'].copy()
test_set_labels = test_set['price'].copy()
train_set = train_set.drop(columns='price')
test_set = test_set.drop(columns='price')

# %%

#Phan nay o trong bai giang nen em xin phep di nhanh qua phan nay
class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names

    def fit(self, dataframe, labels=None):
        return self

    def transform(self, dataframe):
        return dataframe[self.feature_names].values


class MyFeatureAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_PRICE_PER_MINUTE=True, add_PRICE_PER_DAYS_LEFT=True, add_TOTAL_ARRIVAL_TIME=True):
        self.add_PRICE_PER_MINUTE = add_PRICE_PER_MINUTE
        self.add_PRICE_PER_DAYS_LEFT = add_PRICE_PER_DAYS_LEFT
        self.add_TOTAL_ARRIVAL_TIME = add_TOTAL_ARRIVAL_TIME

    def fit(self, feature_values, labels=None):
        return self

    def transform(self, feature_values, labels=None):
        # if self.add_PRICE_PER_DAYS_LEFT:
        #     DAYS_LEFT_ID, PRICE = 0, 2
        #     PRICE_PER_DAYS_LEFT = feature_values[:,
        #                                          PRICE] / (feature_values[:, DAYS_LEFT_ID])
        #     feature_values = np.c_[feature_values, PRICE_PER_DAYS_LEFT]
        # if self.add_PRICE_PER_MINUTE:
        #     DURATION_ID, PRICE = 1, 2
        #     PRICE_PER_MINUTE = feature_values[:, PRICE] / \
        #         (feature_values[:, DURATION_ID]*60)
        #     feature_values = np.c_[feature_values, PRICE_PER_MINUTE]
        if self.add_TOTAL_ARRIVAL_TIME:
            DURATION_ID, DAYS_LEFT_ID = 1, 0
            TOTAL_ARRIVAL_TIME = feature_values[:,
                                                DURATION_ID]*24 + feature_values[:, DAYS_LEFT_ID]
            feature_values = np.c_[feature_values, TOTAL_ARRIVAL_TIME]
        return feature_values


num_feat_names = ['days_left', 'duration']
cat_feat_names = ['airline', 'source_city', 'departure_time', 'stops',
                  'arrival_time', 'destination_city', 'class']

cat_pipeline = Pipeline([
    ('selector', ColumnSelector(cat_feat_names)),
    ('imputer', SimpleImputer(missing_values=np.nan,
     strategy='constant', fill_value='NO INFO', copy=True)),
    ('cat_encoder', OneHotEncoder())
])

num_pipeline = Pipeline([
    ('selector', ColumnSelector(num_feat_names)),
    ('imputer', SimpleImputer(missing_values=np.nan, strategy='median', copy=True)),
    ('attr_adder', MyFeatureAdder(
        add_PRICE_PER_MINUTE=False, add_PRICE_PER_DAYS_LEFT=False, add_TOTAL_ARRIVAL_TIME=True)),
    ('std_scaler', StandardScaler(with_mean=True, with_std=True, copy=True))
])

full_pipeline = FeatureUnion(transformer_list=[
    ('num_pipeline', num_pipeline),
    ('cat_pipeline', cat_pipeline)
])


processed_train_set_val = full_pipeline.fit_transform(train_set)
print('\n____________ Processed feature values ____________')
print(processed_train_set_val[[0, 1, 2], :].toarray())
print(processed_train_set_val.shape)
print('We have %d numeric feature + 1 added features + 35 cols of onehotvector for categorical features.' %
      (len(num_feat_names)))
joblib.dump(full_pipeline, r'models/full_pipeline.pkl')

# %%
onehot_cols = []
for val_list in full_pipeline.transformer_list[1][1].named_steps['cat_encoder'].categories_:
    onehot_cols = onehot_cols + val_list.tolist()
columns_header = train_set.columns.tolist(
) + ["total_arrival_time"] + onehot_cols
for name in cat_feat_names:
    columns_header.remove(name)
processed_train_set = pd.DataFrame(
    processed_train_set_val.toarray(), columns=columns_header)
print('\n____________ Processed dataframe ____________')
print(processed_train_set.info())
print(processed_train_set.head())
# %%
#O day co store, load de dung luu va tai model.
#Ham r2score_and_rmse dung de tinh do chinh xac va sai so khi ma du doan gia ve may bay theo tung model khac nhau

def store_model(model, model_name=""):
    # NOTE: sklearn.joblib faster than pickle of Python
    # INFO: can store only ONE object in a file
    if model_name == "":
        model_name = type(model).__name__
    joblib.dump(model, 'models/' + model_name + '_model.pkl')


def load_model(model_name):
    # Load objects into memory
    #del model
    model = joblib.load('models/' + model_name + '_model.pkl')
    # print(model)
    return model


def r2score_and_rmse(model, train_data, labels):
    r2score = model.score(train_data, labels)
    from sklearn.metrics import mean_squared_error
    prediciton = model.predict(train_data)
    mse = mean_squared_error(labels, prediciton)
    rmse = np.sqrt(mse)
    return r2score, rmse


# %%Linear Regression
#Model Linear
model = LinearRegression()
new_training = 9
if new_training == 10:
    model.fit(processed_train_set_val, train_set_labels)
    store_model(model)
else:
    #Do em da train truoc do, nen em se load model cho nhanh chong
    model = load_model("LinearRegression")
print('\n____________ Linear Regression ____________')
print('Learned parameters: ', model.coef_, model.intercept_)
# %% Tinh do chinh xac va sai so cua model Linear
r2score, rmse = r2score_and_rmse(
    model, processed_train_set_val, train_set_labels)
print('\n____________ Linear Regression ____________')

print('\nR2 score (on training data, best=1):', r2score)
print("Root Mean Square Error: ", rmse.round(decimals=1))
# %%
print("\nInput data: \n", train_set.iloc[0:9])
print("\nPredictions: ", model.predict(
    processed_train_set_val[0:9]).round(decimals=1))
print("Labels:      ", list(train_set_labels[0:9]))
store_model(model)
# %% DecisionTree
model = DecisionTreeRegressor()
new_training = 10
if new_training == 10:
    model.fit(processed_train_set_val, train_set_labels)
    store_model(model)
else:
    model = load_model("DecisionTreeRegressor")
print('\n____________ Decision Tree Regressor ____________')
# %%
r2score, rmse = r2score_and_rmse(
    model, processed_train_set_val, train_set_labels)
print('\nR2 score (on training data, best=1):', r2score)
print("Root Mean Square Error: ", rmse.round(decimals=1))
# %%
print("\nPredictions: ", model.predict(
    processed_train_set_val[0:9]).round(decimals=1))
print("Labels:      ", list(train_set_labels[0:9]))
# %% RandomForest
model = RandomForestRegressor(n_estimators=5)
new_training = 9
if new_training == 10:
    model.fit(processed_train_set_val, train_set_labels)
    store_model(model)
else:
    model = load_model("RandomForestRegressor")
# %%
r2score, rmse = r2score_and_rmse(
    model, processed_train_set_val, train_set_labels)
print('\n____________ Random Forest Regressor ____________')
print('\nR2 score (on training data, best=1):', r2score)
print("Root Mean Square Error: ", rmse.round(decimals=1))
# %%
print("\nPredictions: ", model.predict(
    processed_train_set_val[0:9]).round(decimals=1))
print("Labels:      ", list(train_set_labels[0:9]))

# %% Polynomial
# add high-degree features to the data
poly_feat_adder = PolynomialFeatures(degree=2)
train_set_poly_added = poly_feat_adder.fit_transform(processed_train_set_val)
new_training = 10
if new_training == 10:
    model = LinearRegression()
    model.fit(train_set_poly_added, train_set_labels)
    store_model(model, model_name="PolinomialRegression")
else:
    model = load_model("PolinomialRegression")
# %%
print('\n____________ Polinomial regression ____________')
r2score, rmse = r2score_and_rmse(model, train_set_poly_added, train_set_labels)
print('\nR2 score (on training data, best=1):', r2score)
print("Root Mean Square Error: ", rmse.round(decimals=1))
# %%
print("\nPredictions: ", model.predict(
    train_set_poly_added[0:9]).round(decimals=1))
print("Labels:      ", list(train_set_labels[0:9]))
#%%
from sklearn.svm import SVC
new_training = 10
if new_training:
    model = SVC()
    model.fit(processed_train_set_val,train_set_labels)
    store_model(model)
else:
    model = load_model('SVC')
#%%
r2score, rmse = r2score_and_rmse(
    model, processed_train_set_val, train_set_labels)
print('\n____________ Random Forest Regressor ____________')
print('\nR2 score (on training data, best=1):', r2score)
print("Root Mean Square Error: ", rmse.round(decimals=1))
#%%
print("\nPredictions: ", model.predict(
    processed_train_set_val[0:9]).round(decimals=1))
print("Labels:      ", list(train_set_labels[0:9]))
# %%
print('\n____________ K-fold cross validation ____________')

run_evaluation = 0
if run_evaluation:
    from sklearn.model_selection import KFold, StratifiedKFold
    # NOTE:
    #   + If data labels are float, cross_val_score use KFold() to split cv data.
    #   + KFold randomly splits data, hence does NOT ensure data splits are the same (only StratifiedKFold may ensure that)
    # cv data generator: just a try to persist data splits (hopefully)
    cv = KFold(n_splits=5, shuffle=True, random_state=37)

    # Evaluate LinearRegression
    model_name = "LinearRegression"
    model = LinearRegression()
    nmse_scores = cross_val_score(
        model, processed_train_set_val, train_set_labels, cv=cv, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-nmse_scores)
    joblib.dump(rmse_scores, 'saved_objects/' + model_name + '_rmse.pkl')
    print("LinearRegression rmse: ", rmse_scores.round(decimals=1))
    print("Avg. rmse: ", mean(rmse_scores.round(decimals=1)), '\n')

    # Evaluate DecisionTreeRegressor
    model_name = "DecisionTreeRegressor"
    model = DecisionTreeRegressor()
    nmse_scores = cross_val_score(
        model, processed_train_set_val, train_set_labels, cv=cv, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-nmse_scores)
    joblib.dump(rmse_scores, 'saved_objects/' + model_name + '_rmse.pkl')
    print("DecisionTreeRegressor rmse: ", rmse_scores.round(decimals=1))
    print("Avg. rmse: ", mean(rmse_scores.round(decimals=1)), '\n')

    # Evaluate RandomForestRegressor
    model_name = "RandomForestRegressor"
    model = RandomForestRegressor(n_estimators=5)
    nmse_scores = cross_val_score(
        model, processed_train_set_val, train_set_labels, cv=cv, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-nmse_scores)
    joblib.dump(rmse_scores, 'saved_objects/' + model_name + '_rmse.pkl')
    print("RandomForestRegressor rmse: ", rmse_scores.round(decimals=1))
    print("Avg. rmse: ", mean(rmse_scores.round(decimals=1)), '\n')

    # Evaluate Polinomial regression
    model_name = "PolinomialRegression"
    model = LinearRegression()
    nmse_scores = cross_val_score(
        model, train_set_poly_added, train_set_labels, cv=cv, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-nmse_scores)
    joblib.dump(rmse_scores, 'saved_objects/' + model_name + '_rmse.pkl')
    print("Polinomial regression rmse: ", rmse_scores.round(decimals=1))
    print("Avg. rmse: ", mean(rmse_scores.round(decimals=1)), '\n')
else:
    # Load rmse
    model_name = "LinearRegression"
    rmse_scores = joblib.load('saved_objects/' + model_name + '_rmse.pkl')
    print("\nLinearRegression rmse: ", rmse_scores.round(decimals=1))
    print("Avg. rmse: ", mean(rmse_scores.round(decimals=1)), '\n')

    model_name = "DecisionTreeRegressor"
    rmse_scores = joblib.load('saved_objects/' + model_name + '_rmse.pkl')
    print("DecisionTreeRegressor rmse: ", rmse_scores.round(decimals=1))
    print("Avg. rmse: ", mean(rmse_scores.round(decimals=1)), '\n')

    model_name = "RandomForestRegressor"
    rmse_scores = joblib.load('saved_objects/' + model_name + '_rmse.pkl')
    print("RandomForestRegressor rmse: ", rmse_scores.round(decimals=1))
    print("Avg. rmse: ", mean(rmse_scores.round(decimals=1)), '\n')

    model_name = "PolinomialRegression"
    rmse_scores = joblib.load('saved_objects/' + model_name + '_rmse.pkl')
    print("Polinomial regression rmse: ", rmse_scores.round(decimals=1))
    print("Avg. rmse: ", mean(rmse_scores.round(decimals=1)), '\n')

# %%
#day la phan fine-tunning
print('\n____________ Fine-tune models ____________')


def print_search_result(grid_search, model_name=""):
    print("\n====== Fine-tune " + model_name + " ======")
    print('Best hyperparameter combination: ', grid_search.best_params_)
    print('Best rmse: ', np.sqrt(-grid_search.best_score_))
    # print('Best estimator: ', grid_search.best_estimator_) # NOTE: require refit=True in  SearchCV
    print('Performance of hyperparameter combinations:')
    cv_results = grid_search.cv_results_
    for (mean_score, params) in zip(cv_results["mean_test_score"], cv_results["params"]):
        print('rmse =', np.sqrt(-mean_score).round(decimals=1), params)


method = 1
# 6.1 Method 1: Grid search (try all combinations of hyperparams in param_grid)
if method == 1:
    from sklearn.model_selection import GridSearchCV
    cv = KFold(n_splits=5, shuffle=True, random_state=37)  # cv data generator

    run_new_search = 0
    if run_new_search:
        # 6.1.1 Fine-tune RandomForestRegressor
        model = RandomForestRegressor()
        param_grid = [
            # try 12 (3x4) combinations of hyperparameters (bootstrap=True: drawing samples with replacement)
            {'bootstrap': [True], 'n_estimators': [3, 15, 30], 'max_features': [2, 12, 20, 38]},
            # then try 12 (4x3) combinations with bootstrap set as False
            {'bootstrap': [False], 'n_estimators': [3, 5, 10, 20], 'max_features': [2, 6, 10]}
            ]
        # Train across 5 folds, hence a total of (12+12)*5=120 rounds of training
        grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='neg_mean_squared_error', return_train_score=True, refit=True)  # refit=True: after finding best hyperparam, it fit() the model with whole data (hope to get better result)
        grid_search.fit(processed_train_set_val, train_set_labels)
        joblib.dump(grid_search, 'saved_objects/RandomForestRegressor_gridsearch.pkl')
        print_search_result(grid_search, model_name="RandomForestRegressor")

        # 6.1.2 Fine-tune Polinomial regression
        model = Pipeline([('poly_feat_adder', PolynomialFeatures()),  # add high-degree features
                          ('lin_reg', LinearRegression())])
        param_grid = [
            # try 3 values of degree
            {'poly_feat_adder__degree': [1, 2, 3]}]  # access param of a transformer: <transformer>__<parameter> https://scikit-learn.org/stable/modules/compose.html
        # Train across 5 folds, hence a total of 3*5=15 rounds of training
        grid_search = GridSearchCV(
            model, param_grid, cv=cv, scoring='neg_mean_squared_error', return_train_score=True)
        grid_search.fit(processed_train_set_val, train_set_labels)
        joblib.dump(grid_search, 'saved_objects/PolinomialRegression_gridsearch.pkl')
        print_search_result(grid_search, model_name="PolinomialRegression")
    else:
        # Load grid_search
        grid_search_load = joblib.load('saved_objects/RandomForestRegressor_gridsearch.pkl')
        print_search_result(grid_search_load, model_name="RandomForestRegressor")
        grid_search_load = joblib.load('saved_objects/PolinomialRegression_gridsearch.pkl')
        print_search_result(grid_search_load, model_name="PolinomialRegression")

# 6.2 Method 2: [EXERCISE] Random search n_iter times
elif method == 2:
    haha = 1

else:
    haha = 2

#%%
#O day ta se chon ra model tot nhat va save lai.
search = joblib.load('saved_objects/RandomForestRegressor_gridsearch.pkl')
best_model = search.best_estimator_
print('\n____________ ANALYZE AND TEST YOUR SOLUTION ____________')
print('SOLUTION: ' , best_model)
store_model(best_model, model_name="SOLUTION")   

if type(best_model).__name__ == "RandomForestRegressor":
    # Print features and importance score  (ONLY on rand forest)
    feature_importances = best_model.feature_importances_
    onehot_cols = []
    for val_list in full_pipeline.transformer_list[1][1].named_steps['cat_encoder'].categories_: 
        onehot_cols = onehot_cols + val_list.tolist()
    feature_names = train_set.columns.tolist() + ["total_arrival_time"] + onehot_cols
    for name in cat_feat_names:
        feature_names.remove(name)
    print('\nFeatures and importance score: ')
    print(*sorted(zip( feature_names, feature_importances.round(decimals=4)), key = lambda row: row[1], reverse=True),sep='\n')
    
full_pipeline = joblib.load(r'models/full_pipeline.pkl')
processed_test_set = full_pipeline.transform(test_set)  
# 7.3.1 Compute R2 score and root mean squared error
r2score, rmse = r2score_and_rmse(best_model, processed_test_set, test_set_labels)
print('\nPerformance on test data:')
print('R2 score (on test data, best=1):', r2score)
print("Root Mean Square Error: ", rmse.round(decimals=1))
# 7.3.2 Predict labels for some test instances
print("\nTest data: \n", test_set.iloc[0:9])
print("\nPredictions: ", best_model.predict(processed_test_set[0:9]).round(decimals=1))
print("Labels:      ", list(test_set_labels[0:9]),'\n')
#%%