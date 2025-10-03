import pandas as pd
file_path = "data/assurancemaladie.csv"
df = pd.read_csv(file_path)
data_As_list = df.values.tolist()
print(df['age'].value_counts().sort_index())
df.isna()
#
df_bigdata_duplicates = df[df.duplicated()]
#delete the duplicate it data
df_bigdata_duplicates = df.drop_duplicates(inplace=True)
# the data info 
print(df.info())
#top 5 recordes of data 
print(df.head())
#the bottom 5 rec data
print(df.tail())
# how many columns and rows
print(df.shape)
#finding missing value
print(df.isnull().sum())
#presentage of missing value
print(df.isnull().sum()/df.shape[0]*100)
#the sum of the suplicated rows
print(df.duplicated().sum())
print(df.select_dtypes(include="object").columns)
for i in df.select_dtypes(include="object").columns: 
    print(df[i].value_counts())
#explorarety data analytics
print(df.describe().T)
print(df.describe(include="object"))
#histogram to understand the disctribution
mean_Value = df['charges'].mean()
#stadard deviation
std_dev_value = df['charges'].std(ddof=0)
print("mean value of chgarges : " , mean_Value)
print("stadard deviation  : " , std_dev_value)

outliers_bigger_than_3  = mean_Value + ( 3  * std_dev_value)
high_outliers = df[df['charges'] > outliers_bigger_than_3]
for i in high_outliers['charges'] : 
    zscore = ((i -  mean_Value ) / std_dev_value)
    print(f" x : {i}  , zscore : {zscore}")
print("the total number of outliers" , len((high_outliers['charges'] -  mean_Value ) / std_dev_value > 3))
#select the object columns type(categorial)
catg_cols = df.select_dtypes(include="object").columns.tolist()
print(catg_cols)
from sklearn.preprocessing import OneHotEncoder 
import numpy as np
encoder = OneHotEncoder(sparse_output = False  , handle_unknown="ignore")
encoder.fit(df[catg_cols])
print(encoder.categories_)
encoded_cols = list(encoder.get_feature_names_out(catg_cols))
# import warnings 
df[encoded_cols] = encoder.transform(df[catg_cols])
print(df.head())
df.drop(columns =catg_cols , inplace=True )
print(df.head())
print(df.select_dtypes(include="object").columns.tolist())
print(df.select_dtypes(include="number").columns.tolist())
# Q1 = df['bmi'].quantile(0.25)
# Q3 = df['bmi'].quantile(0.75)
# IQR = Q3 - Q1
# min  = Q1 - 1.5 * IQR
# max = Q3 + 1.5 * IQR
# df = df[(df['bmi'] >= min ) & (df['bmi'] <= max)]

#spliting data with sktlearn using train_test_split func
from sklearn.model_selection import train_test_split
# With shuffle=False
# X_train_ns, X_test_ns, y_train_ns, y_test_ns = train_test_split(X, y, test_size=0.2, shuffle=False)

# print("Without shuffle:", X_train_ns.head(), "->", X_test_ns.head())

# print(df.select_dtypes(include="number").columns)
# import matplotlib.pyplot as pltx
# import seaborn as sns
# from sklearn.preprocessing import MinMaxScaler

# #normalization 
# scaler = MinMaxScaler()
# df = scaler.fit_transform(df)

# print(df.describe().round(2))
# print(df.shape)
# print(X.shape , y.shape)




# with the standarization 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
columns = df.select_dtypes(include="number")

X = df[['age', 'bmi', 'children', 'sex_female', 'sex_male', 'smoker_no', 'smoker_yes', 'region_northeast', 'region_northwest', 'region_southeast', 'region_southwest']]
# X= [[columns]]
# df['log_charges'] = np.log(df['charges'])
y = df['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
df = pd.DataFrame(df)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
model = LinearRegression()
model.fit(X_train_scaled,y_train)
y_pred = model.predict(X_test)
# Convert back to original scale
# y_pred = np.exp(y_pred)
# y_test = np.exp(y_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error : {mse:.2f}")
print(f"R² Score: {r2:.2f}")
from sklearn.ensemble import RandomForestRegressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)
y_pred = rf_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
single_data = X_test.iloc[0].values.reshape(1, -1)
predicted_value = rf_regressor.predict(single_data)
print(f"Predicted Value: {predicted_value[0]:.2f}")
print(f"Actual Value: {y_test.iloc[0]:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")
from xgboost import XGBRegressor
xgb_reggressor = XGBRegressor()
xgb_reggressor.fit(X_train,y_train)

y_pred = xgb_reggressor.predict(X_test)
mse1 = mean_squared_error(y_test, y_pred)
r20 = r2_score(y_test, y_pred)
print(f"Predicted Value: {predicted_value[0]:.2f}")
print(f"Actual Value: {y_test.iloc[0]:.2f}")
print(f"XGBRegressor Mean Squared Error : {mse1:.2f}")
print(f"XGBRegressor R² Score: {r20:.2f}")
from sklearn.svm import SVR
svr = SVR(kernel='rbf', C=100, epsilon=0.1)
svr.fit(X_train,y_train)
# Predict and inverse scale the results
y_pred = svr.predict(X_test)
mse1 = mean_squared_error(y_test, y_pred)
r20 = r2_score(y_test, y_pred)
print(f"Predicted Value: {predicted_value[0]:.2f}")
print(f"Actual Value: {y_test.iloc[0]:.2f}")
print(f"XGBRegressor Mean Squared Error : {mse1:.2f}")
print(f"XGBRegressor R² Score: {r20:.2f}")
from sklearn.pipeline import Pipeline
svr_pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Step 1: Scale features
    ('svr', SVR(kernel='rbf', C=100, epsilon=0.1))  # Step 2: SVR model
])

# Fit the pipeline
svr_pipeline.fit(X_train, y_train)
# Predict using pipeline
y_pred = svr_pipeline.predict(X_test)
# Evaluate
mse1 = mean_squared_error(y_test, y_pred)
r20 = r2_score(y_test, y_pred)
# Optional: compare predicted and actual for one example
print(f"Predicted Value: {y_pred[0]:.2f}")
print(f"Actual Value: {y_test.iloc[0]:.2f}")
print(f"SVR (Pipeline) Mean Squared Error: {mse1:.2f}")
print(f"SVR (Pipeline) R² Score: {r20:.2f}")


# sns.pairplot(df)
# pltx.show()
# df_Copy.describe().round(2)
# warnings.filterwarnings("ignore")
# for i  in df.select_dtypes(include="number") :
#     sns.histplot(data=df ,x=i)
#     plt.show()
# for i  in df.select_dtypes(include="number").columns :
#     sns.boxplot(data=df ,x=i)
#     plt.show()
# plt.figure(figsize=(15, 6))
# counts, bins, _ = plt.hist(df['age'], bins=range(int(df['age'].min()), int(df['age'].max())+2),
#                            color='skyblue', edgecolor='black', align='left')
# for c, b in zip(counts, bins):
#     if c > 0:
#         plt.text(b+0.5, c+0.5, str(int(c)), ha='center', va='bottom', fontsize=10)
# plt.title("Distribution des âges", fontsize=18)
# plt.xlabel("Âge", fontsize=14)
# plt.ylabel("Nombre de personnes", fontsize=14)
# plt.xticks(range(int(df['age'].min()), int(df['age'].max())+1), rotation=45)
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.show()
# filtred_ages = df['age'].value_counts().values.tolist()
# filtred_ages_by_Smoke = df['smoker']
# print(filtred_ages)
