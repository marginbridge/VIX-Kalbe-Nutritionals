# Kalbe Nutritionals Data Scientist Project Based Internship Program
## Introduction
VIX Data Scientist Kalbe Nutritionals merupakan virtual internship experience yang difasilitasi oleh Rakamin Academy. 

**Objectives** <br>
* Memperkiraan quantity product yang terjual sehingga tim inventory dapat membuat stock persediaan harian yang cukup. <br>
* Membuat customer segment yang nantinya akan digunakan oleh tim marketing untuk memberikan personalized promotion dan sales treatment. <br>

**Dataset** <br>
[Kalbe Nutritionals sales dataset tahun 2022](https://drive.google.com/drive/folders/1_rQrauVW2OvLIe2zd54Vcwnr2EY-vnnR) <br>

**Tools** <br>
* Python <br>
* Jupyter Notebook <br>
* Tableau <br>
* Dbeaver <br>
* PostgreSQL <br>

## 1. Exploratory Data Analysis (EDA) di dbeaver
**Berapa rata-rata umur customer jika dilihat dari marital statusnya?** <br>
<p align="center">
<img width="400" alt="Screenshot 2023-11-06 014601" src="https://github.com/marginbridge/VIX-Kalbe-Nutritionals/assets/90979655/5f7c32f3-e8bb-440e-967f-2532db169000"> <br>

**Berapa rata-rata umur customer jika dilihat dari gender nya?** <br>
<p align="center"> 
<img width="400" alt="Screenshot 2023-11-06 015501" src="https://github.com/marginbridge/VIX-Kalbe-Nutritionals/assets/90979655/ce4a01cf-ce35-4bbe-aef2-efb23643dec7"> <br>

**Nama store dengan total quantity terbanyak!** <br>
<p align="center"> 
<img width="400" alt="Screenshot 2023-11-06 015707" src="https://github.com/marginbridge/VIX-Kalbe-Nutritionals/assets/90979655/744a81cc-9cf0-499a-9c93-3caf759d9651"> <br>

**Nama produk terlaris dengan total amount terbanyak!** <br>
<p align="center"> 
<img width="400" alt="Screenshot 2023-11-06 015926" src="https://github.com/marginbridge/VIX-Kalbe-Nutritionals/assets/90979655/c606ce05-a736-4684-9a86-8998139285af">

## 2. Data Visualization

## 3. Time Series Forecasting
Aim to implement a machine learning model to accurately predict the number of sales (quantity) of the total Kalbe products.
### Data preparation
```Python
df = df_merged.groupby('Date').agg({'Qty':'sum'})
```
### Check stasionarity
```Python
X = df['Qty'].values
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
```
p-value < 0.05, thus the data is stationary.
### Figure out order for ARIMA
```Python
auto_arima_model = pm.auto_arima(df['Qty'],
                                 seasonal=False,
                                 stepwise=False,
                                 trace=True,
                                 suppress_warnings=True)
auto_arima_model.summary()
```
Got Best model: ARIMA(1,0,1)
### Create training and testing datasets
```Python
train_size = int(len(df['Qty']) * 0.8)
train_data = df['Qty'][:train_size]
test_data = df['Qty'][train_size:]
print(train_data.shape,test_data.shape)
```
Visulize the differences:
```Python
plt.figure(figsize=(12,5))
sns.lineplot(data=train_data, x=train_data.index, y=train_data)
sns.lineplot(data=test_data, x=test_data.index, y=test_data)
plt.show()
```
<p align="center">
  <img src="https://github.com/marginbridge/VIX-Kalbe-Nutritionals/assets/90979655/e5ba5cfb-c5b8-46f7-b473-b3ee843c9aca" alt="Image description" width="900" height="350">
</p>

### Implement ARIMA
```Python
model = ARIMA(train_data, order=(1,0,1))
model = model.fit()
model.summary()
```
### Predict on test set
```Python
start = len(train_data)
end = len(train_data)+len(test_data)-1
pred = model.predict(start=start,end=end,typ='levels')
print(pred)
```
Check model accuracy:
```Python
rmse = sqrt(mean_squared_error(test_data,pred))
print(rmse)
```
output:
```Python
15.49482859020857
```
### Improvement: Manual ARIMA
```Python
model = ARIMA(train_data, order=(70,2,2))
model = model.fit()
model.summary()
```
```Python
start = len(train_data)
end = len(train_data)+len(test_data)-1
pred = model.predict(start=start,end=end,typ='levels')
print(pred)
```
```Python
pred.plot(label='Predicted')
test_data.plot(label='Actual')

plt.legend()
plt.show()
```
<p align="center">
  <img src="https://github.com/marginbridge/VIX-Kalbe-Nutritionals/assets/90979655/a6093ac6-b0e8-4e0d-83e4-a26c8d618091" alt="Image description" width="500" height="450">
</p>

```Python
rmse = sqrt(mean_squared_error(test_data,pred))
print(rmse)
```
Output:
```Python
17.333560693932814
```

### Forecast for all product
```Python
product_reg_df = df_merged[['Qty', 'Date', 'Product Name']]
new = product_reg_df.groupby("Product Name")

forecast_product_df = pd.DataFrame({'Date': pd.date_range(start='2023-01-01', periods=90)})

for product_name, group_data in new:
    target_var = group_data['Qty']
    model = ARIMA(target_var.values, order=(1,0,1))
    model_fit = model.fit()
    forecast = model_fit.forecast(90)
    forecast_product_df[product_name] = forecast

forecast_product_df.set_index('Date', inplace=True)
forecast_product_df.head()
```
Output:

| Date       | Cashew           | Cheese Stick    | Choco Bar     | Coffee Candy  | Crackers      | Ginger Candy  | Oat           | Potato Chip   | Thai Tea      | Yoghurt       |
| :--------- |:----------------:|:---------------:|:-------------:|:-------------:|:-------------:|:-------------:|--------------:|:-------------:|:-------------:|--------------:|
| 2023-01-01 | 2.643780         | 2.887033        | 5.719084      | 3.922062      | 3.536212      | 4.854919      | 1.989118      | 2.807238      | 3.472691      | 4.060748      |
| 2023-01-02 | 2.400617         | 2.916508        | 5.841903      | 3.982036      | 3.498383      | 4.931029      | 1.989644      | 2.803930	  | 3.521524      | 3.985260      |
| 2023-01-03 | 2.477836         | 2.935584        | 5.920862      | 4.004379      | 3.526270	  | 4.949085      | 1.989686      | 2.802217      | 3.496307      | 4.059759      |
| 2023-01-04 | 2.453314         | 2.947930        | 5.971625	  | 4.012703      | 3.505712      | 4.953368      | 1.989689      | 2.801329      | 3.509329      | 3.986237      |
| 2023-01-05 | 2.461101         | 2.955920        | 6.004260      | 4.015804      | 3.520867      | 4.954385      | 1.989690      | 2.800869      | 3.502604      | 4.058795      |

```Python
round(forecast_product_df.describe().T['mean'],0)
```
Output:
| Product Name | Quantity |
| ------------ | -------- |
| Cashew       | 2.0      |
| Cheese Stick | 3.0      |
| Choco Bar    | 6.0      |
| Coffee Candy | 4.0      |
| Crackers     | 4.0      |
| Ginger Candy | 5.0      |
| Oat          | 2.0      |
| Potato Chip  | 3.0      |
| Thai Tea     | 4.0      |
| Yoghurt      | 4.0      |
### Future Improvement
* Rolling forecast
* Other methods: Exponential Smoothing (ES), Simple ES
## 4. Customer Segmentation: Clustering
### Data preparation
```Python
df = df_merged.groupby('CustomerID').agg({'TransactionID':['count'],
                                          'Qty':['sum'],
                                          'TotalAmount':['sum']}).reset_index()
df.columns = ['CustomerID','TransactionID','Quantity','Total Amount']
df.head()
```
Feature slection for the model. Considering only 2 features (**Quantity and Total Amount**) Feature scaling:
```Python
X = df.iloc[:,[2,3]].values
```
```Python
# scaling using MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X)
scaled_data = scaler.fit_transform(X)
```
### KMeans clustering
Finding the number of clusters using **elbow method**
```Python
K = range(1,12)
wss = []
for k in K:
    kmeans=KMeans(n_clusters=k,init="k-means++")
    kmeans=kmeans.fit(X)
    wss_iter = kmeans.inertia_
    wss.append(wss_iter)
```
We store the number of clusters along with their WSS Scores in a DataFrame:
```Python
mycenters = pd.DataFrame({'Clusters' : K, 'WSS' : wss})
mycenters
```
Visualize:
```Python
sns.lineplot(x = 'Clusters', y = 'WSS', data = mycenters, marker="o")
```
<p align="center">
  <img src="https://github.com/marginbridge/VIX-Kalbe-Nutritionals/assets/90979655/d1c26d0f-93fc-4712-9a74-e51e8dffac8c" alt="Image description" width="400" height="350">
</p>
