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

## Exploratory Data Analysis (EDA) di dbeaver
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

## Data Visualization

## Time Series Forecasting
Aim to implement a machine learning model to accurately predict the number of sales (quantity) of the total Kalbe products.
### 1. Data preparation
```Python
df = df_merged.groupby('Date').agg({'Qty':'sum'})
```
### 2. Check stasionarity
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
### 3. Figure out order for ARIMA
```Python
auto_arima_model = pm.auto_arima(df['Qty'],
                                 seasonal=False,
                                 stepwise=False,
                                 trace=True,
                                 suppress_warnings=True)
auto_arima_model.summary()
```
Got Best model: ARIMA(1,0,1)
### 4. Create training and testing datasets
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
![download (1)](https://github.com/marginbridge/VIX-Kalbe-Nutritionals/assets/90979655/e5ba5cfb-c5b8-46f7-b473-b3ee843c9aca)

### 4. Implement ARIMA
```Python
model = ARIMA(train_data, order=(1,0,1))
model = model.fit()
model.summary()
```
### 5. Predict on test set
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
### 5. Improvement: Manual ARIMA
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
  <img src="https://github.com/marginbridge/VIX-Kalbe-Nutritionals/assets/90979655/a6093ac6-b0e8-4e0d-83e4-a26c8d618091" alt="Image description" width="500" height="500">
</p>

```Python
rmse = sqrt(mean_squared_error(test_data,pred))
print(rmse)
```
Output:
```Python
17.333560693932814
```
### 6. Forecast for all product
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
| 2023-01-01 | 2.643780         | 2.887033        | 5.719084      |               |               |               |               |               |               |               |
| 2023-01-02 | 2.400617         | 2.916508        | 5.841903      |               |               |               |               |               |               |               |
| 2023-01-03 | 2.477836         | 2.935584        | 5.920862      |               |               |               |               |               |               |               |
| 2023-01-04 | 2.453314         | 2.947930        | 5.971625	  |               |               |               |               |               |               |               |
| 2023-01-05 | 2.461101         | 2.955920        | 6.004260      |               |               |               |               |               |               |               |
## Customer Segmentation: Clustering



