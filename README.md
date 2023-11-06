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
<p align="center">
<img width="103" alt="image" src="https://github.com/marginbridge/VIX-Kalbe-Nutritionals/assets/90979655/2c169c40-c7af-41db-bb2e-f1098e24da9d">

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
### 3. Create training and testing datasets
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

### 4. Create training and testing datasets
## Customer Segmentation: Clustering



