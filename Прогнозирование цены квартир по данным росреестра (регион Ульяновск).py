# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 13:42:10 2022

@author: Elena Eremeeva
"""
import numpy as np
import json
import pandas as pd
import glob
from matplotlib import pyplot as plt
import seaborn as sns
# #импортируем библиотеки для построения моделей
from sklearn.linear_model import LinearRegression  
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, r2_score

from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor, plot_importance
from sklearn.neural_network import MLPRegressor
from wordcloud import WordCloud
pd.set_option('display.max_columns', None)

# создание функции перевода не числовых данных в числовые
def trans(dataframe):    
    for column in dataframe.columns:
        if dataframe[column].dtype != 'int64' and dataframe[column].dtype != 'float64':
            trans = LabelEncoder()
            trans.fit(dataframe[column])
            dataframe[column] = trans.transform(dataframe[column])    
    return dataframe
# создание функции нормализации данных
def scaller(dataframe):
    for column in dataframe.columns:
        if abs(dataframe[column].max()) > 1:
            scaller = MinMaxScaler()
            scaller.fit(dataframe[[column]])
            dataframe[column] = scaller.transform(dataframe[[column]])
    return dataframe
#Создание функции перебора моделей и метрик
def modeling(models, x_train, x_test, y_train, y_test):
    r2 = {}
    mape = {}
    for model in models:
        model.fit(x_train,y_train)
        y_pred = model.predict(x_test)
        y_pred2 = model.predict(x_train)
        print(model, ': r2_score', r2_score(y_train, y_pred2))
        print(model, ': MAPE', mean_absolute_percentage_error(y_test, y_pred)*100)
        r2.update({str(model):r2_score(y_train, y_pred2)})    
        mape.update({str(model):mean_absolute_percentage_error(y_test, y_pred)*100})
        if model == DTR:
            fig = plt.figure(dpi = 300, figsize = (20,30))
            plot_tree(model, filled=True, feature_names = x_train.columns)
            plt.show()
        elif model == XGBR:
            plot_importance(model)
            plt.show()
        elif model == LR:
            print('Коэффициент b:', model.intercept_) # b
            print('Коэффициент m:', model.coef_) # m
    plt.figure(figsize=(18, 10))
    plt.title('Коэффициент детерминации (r2_score) по моделям регрессии')
    plt.xlabel("Модель (алгоритм)")
    plt.ylabel('Коэффициент детерминации (r2_score) ')
    plt.plot(r2.keys(), r2.values(), marker='o', color='red')
    plt.show()
    plt.figure(figsize=(18, 10))
    plt.title('Средняя относительная ошибка (MAPE) по моделям регрессии')
    plt.xlabel("Модель (алгоритм)")
    plt.ylabel('Средняя относительная ошибка (MAPE)')
    plt.plot(mape.keys(), mape.values(), marker='s', color='blue')
    plt.show()
    
#Загрузка данных
price_files=glob.glob('price for pred/flat*.json')
df = pd.DataFrame()
for file in price_files:
    with open(file, 'r', encoding='utf-8') as readfile:
        data = json.load(readfile)
        df_ones = pd.DataFrame.from_dict(pd.json_normalize(data), orient='columns')       
        df = pd.concat([df, df_ones], axis = 0, ignore_index=True)    
#Исследование и предобработка данных                
print(df.info())
print(df.duplicated().sum())
print(df.describe().round(2))
print(df.isnull().sum())
df.dropna(subset = ['right.dealPrice'], inplace =True)
print(df.info())
print(df['regDate'].head())
df['regDate'] = pd.to_datetime(df.regDate)
print(df.isnull().sum())
print(df['regDate'].min())
print(df['regDate'].max())
df = df[df['regDate'] >= '2019-01-01']
print(df.isnull().sum())
#проверим уникальные значения по столбцам типа объект
for column in df.columns:
    if df[column].dtypes == 'O':
        print(column)
        print(df[column].unique())
#выберем данные, которые подходят для решения поставленных задач
df = df[df['purpose.text'] == 'жилое помещение']
df = df[df['address.regionCode'] == '73']
df = df[df['address.region'] == 'Ульяновская область']
df = df[(df['right.dealKind'] != '454001001000') & (df['right.dealKind'] != '454002003000') &
        (df['right.dealKind'] != '454001002000') & (df['right.dealKind'] != '454002001000') &
        (df['right.dealKind'] != '454009000000') & (df['right.dealKind'] != '454001004000') &
        (df['right.dealKind'] != '454001005000') & (df['right.dealKind'] != '454003000000') &
        (df['right.dealKind'] != '454005000000') & (df['right.dealKind'] != '454007000000') &
        (df['right.dealKind'] != '454012000000') & (df['right.dealKind'] != '454015000000') &
        (df['right.dealKind'] != '454006000000') & (df['right.dealKind'] != '454013000000')]   
#Передача жилья в собственность граждан (приватизация жилья), Договор ренты, Договор концессии
#Договор мены, и т.п. оставляем только договор купли-продажи или nan
df = df[df['purpose.code'] == '206002000000']   #206002000000  жилое
df = df[df['right.rightKind'] != '001005000000']    #001005000000  Оперативное управление
df = df[df['objKind'] == '002001003000']    #002001003000  Помещение
df.reset_index(inplace = True, drop=True)
print(df.shape)
for column in df.columns:
    if df[column].dtypes == 'O':
        print(column)
        print(df[column].unique())

#объединим столбцы, которые показывают аналогичные данные, чтобы не было пропусков
df['address.city_or_locality'] = df['address.city'].combine_first(df['address.locality'])
df['address.cityType_or_localityType'] = df['address.cityType'].combine_first(df['address.localityType'])
df['address.district_urban_or_localty'] = df['address.urbanDistrict'].combine_first(df['address.district'])
df.reset_index(inplace = True, drop=True)
print(df.info())

#удалим столбцы, которые не имеют данные или имеют одинаковые данные для всех
df.drop(columns = ['type', 'status', 'objKind', 'purpose.code', 
                    'purpose.text', 'cadCost.costValue', 'cadCost.upks',
                    'cadCost.registrationDate', 'cadCost.determinationDate',
                    'address.regionCode', 'address.region', 'address.sovet',
                    'address.sovetType', 'address.urbanType',
                    'right.dealKind', 'right.dealCurrency',
                    'right.rightKind', 'right.right_part', 'right.regDateStr',
                    'right.hasParts', 'right.dealPriceText'], inplace=True)
df.reset_index(inplace = True, drop=True)
print(df.info())
print(df.duplicated().sum())
df.drop_duplicates(inplace=True)
print(df.shape)
print(df.describe().round(1))
print(df.area.mode()) 
#удалим экстремальные значения по площади
df = df[df['area'] < (df['area'].mean() + 3 * df['area'].std())]
df.reset_index(inplace = True, drop=True)
sns.histplot(df.area, bins=50)
plt.title('Распределение по площади')
plt.xlabel('Площадь, кв.м')
plt.show()
print(df.describe().round(1))

#удалим экстремальные и ошибочные значения по цене сделки
print(df[df['right.dealPrice'] == 13530.00])
df = df[df['right.dealPrice'] != 13530.00]
print(df.describe().round(2))
print(df[df['right.dealPrice'] == 48156.66])
df = df[df['right.dealPrice'] != 48156.66]
print(df.describe().round(2))
print(df[df['right.dealPrice'] == 50000.00])
df = df[df['right.dealPrice'] < (df['right.dealPrice'].mean() + 3 * df['right.dealPrice'].std())]
df.reset_index(inplace = True, drop = True)
print(df.describe().round(2))
df = df[df['right.dealPrice'] < (df['right.dealPrice'].mean() + 3 * df['right.dealPrice'].std())]
df.reset_index(inplace = True, drop = True)
print(df.describe().round(2))
sns.histplot(df['right.dealPrice'], bins=10)
plt.title('Распределение цен')
plt.xlabel('Цена сделки, руб.')
plt.show()
#построим нормальное распределение по цене сделки
norm_raspred = np.random.normal(df['right.dealPrice'].mean(), df['right.dealPrice'].std(), size=100000)
sns.histplot(norm_raspred, bins = 100)
plt.title('Нормальное распределение')
plt.show()
#построим диаграмму рассеивания поплощади и цене сделки
plt.figure(figsize=(12,7))
sns.scatterplot(data=df, x='area', y='right.dealPrice')
plt.title('Зависимость цены от площади квартиры')
#Посмотрим график количества сделок по годам
date = df.groupby(df.regDate.dt.year).regDate.count()
date = pd.DataFrame(date)
print(date)
plt.figure(figsize=(12,7))
plt.title('Количество сделок по годам')
plt.plot(date.regDate, ls = '--', marker='o', color = 'red')
plt.xlabel('год')
plt.ylabel('количество сделок')
plt.show()
#Посмотрим график изменения средней цены по дате сделки
date_price = df.groupby(df.regDate)['right.dealPrice'].mean()
date_price = pd.DataFrame(date_price)
print(date_price)
plt.figure(figsize=(12,7))
plt.title('Изменение средней цены сделки по датам сделки')
sns.lineplot(data=date_price , x='regDate', y='right.dealPrice')

# Изменение цены от даты сделки
plt.figure(figsize=(12,7))
plt.title('Изменение цен от даты сделки')
sns.lineplot(data=df, x='regDate', y='right.dealPrice')
plt.show()

#Посмотрим график сделок по местам локации недвижимости (жилых помещений)
place = df.groupby(['address.city_or_locality']).regDate.count().sort_values()
place = pd.DataFrame(place)
print(place)
plt.figure(figsize=(12,7))
plt.title('Количество сделок по месту локации')
plt.plot(place.regDate.tail(), ls = ':', marker='s', color = 'green')
plt.xlabel('место локации')
plt.ylabel('количество сделок')
plt.show()

#Посмотрим график количества сделок по районам города Ульяновск
print(df['address.urbanDistrict'].unique())
#исправим ошибки в написании района города
df.loc[df['address.urbanDistrict'] == 'Заволжский  район', 'address.urbanDistrict'] = 'Заволжский'
df.loc[df['address.urbanDistrict'] == 'Заволжский район', 'address.urbanDistrict'] = 'Заволжский'
print(df['address.urbanDistrict'].unique())
place1 = df.groupby(['address.urbanDistrict']).regDate.count().sort_values()
place1 = pd.DataFrame(place1)
print(place1)
plt.figure(figsize=(12,7))
plt.title('Количество сделок по районам города Ульяновск')
plt.plot(place1.regDate.tail(4), ls = ':', marker='*', color = 'blue')
plt.xlabel('район города')
plt.ylabel('количество сделок')
plt.show()
#Посмотрим график по средней цене сделки по районам города Ульяновск
a = df[df['address.urbanDistrict'] == 'Заволжский']['right.dealPrice'].mean()
b = df[df['address.urbanDistrict'] == 'Засвияжский']['right.dealPrice'].mean()
c = df[df['address.urbanDistrict'] == 'Ленинский']['right.dealPrice'].mean()
d = df[df['address.urbanDistrict'] == 'Железнодорожный']['right.dealPrice'].mean()
place_price = [a, b, c, d]  
district = ['Заволжский', 'Засвияжский', 'Ленинский', 'Железнодорожный']  
print(place_price)
plt.figure(figsize=(12,7))
plt.title('Средняя цена сделок по районам города Ульяновск')
plt.plot(district, place_price,  ls = ':', marker='*', color = 'blue')
plt.xlabel('район города')
plt.ylabel('средняя цена сделок')
plt.show()

#облако слов адресов (улиц)(частота использования слов адресов)

comment_words = ''

for word in df['address.street']:   
     # слово переводим в строкойвый тип
     word = str(word) 
     # разделим слова
     tokens = word.split()     
     # каждое слово приведем к маленькому регистру
     for i in range(len(tokens)):
         tokens[i] = tokens[i].lower()     
     comment_words += ' '.join(tokens)+' '
 
wordcloud = WordCloud(width = 1000, height = 1000,
                background_color ='white',
                min_font_size = 10).generate(comment_words)
 
 # изобразим облако слов                      
plt.figure(figsize = (7, 7), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
plt.show()

print(df.info())
#создаем датасет (выбираем нужные колонки) которые будем использовать для построения модели
#отсечем данные, где нет адреса улицы
df.dropna(subset = 'address.street', inplace = True)
#отсечем столбцы где не достаточно данных
data = df[['cadBlockNum', 'area', 'address.city_or_locality', 'address.cityType_or_localityType',
             'address.street', 'address.streetType', 'right.dealPrice']]
#data.reset_index(inplace = True, drop=True)
print(data.info())
print(data.head())
print(data.isnull().sum())
print(data.describe().round(2))
print(data.shape)
#для моделирования будем использовать место локации город Ульяновск
data = data[data['address.city_or_locality'] == 'Ульяновск']
print(data.shape)
#Переведем не числовые данные в числа
data = trans(data)
#построим таблицу корреляции
plt.title('Тепловая карта корреляции')
sns.heatmap(data.corr(), annot = True, cbar = False)
plt.show()

#Моделирование
#Определим модели регрессии 
DTR = DecisionTreeRegressor(criterion = 'squared_error', min_samples_split = 2, min_samples_leaf = 2)
RFR = RandomForestRegressor(n_estimators=10, random_state=1, min_samples_split = 2, min_samples_leaf = 2)
XGBR = XGBRegressor(learning_rate = .4)
LR = LinearRegression()
KNR = KNeighborsRegressor(3, weights = 'distance')

#разделим данные на признаки (фичи) и цель (target)
x = data.drop(columns = ['right.dealPrice'])
y = data['right.dealPrice']
print(x.head())
#разобъем данные на тренировочную и тестовую выборки
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=0)
print('X Train : ', x_train.shape)
print('X Test  : ', x_test.shape)
print('Y Train : ', y_train.shape)
print('Y Test  : ', y_test.shape)
#создадим список моделей
lst_model_tree = [DTR, RFR, XGBR]
#промоделируем
print('Модели на основе деревьев решений с использованием всех признаков')
modeling(lst_model_tree, x_train, x_test, y_train, y_test)

#согласно гистограмме важности функций для решения задачи используем три наиболее важных.
#разделим данные на признаки (фичи) и цель (target)
x1 = x.drop(columns = ['address.city_or_locality', 'address.cityType_or_localityType',
                       'address.streetType'])
print(x.head())
#разобъем данные на тренировочную и тестовую выборки
x_train, x_test, y_train, y_test = train_test_split(x1, y, test_size=.2, random_state=0)
print('X Train : ', x_train.shape)
print('X Test  : ', x_test.shape)
print('Y Train : ', y_train.shape)
print('Y Test  : ', y_test.shape)

#промоделируем
print('Модели на основе деревьев решений с использованием трех наиболее важных признаков')
modeling(lst_model_tree, x_train, x_test, y_train, y_test)

#построим модели регрессии линейной и ближайщего соседа с нормализацией данных на всех признаках
#разделим данные на признаки (фичи) и цель (target)
x_train, x_test, y_train, y_test = train_test_split(scaller(x), y, test_size=.2, random_state=0)
print('X Train : ', x_train.shape)
print('X Test  : ', x_test.shape)
print('Y Train : ', y_train.shape)
print('Y Test  : ', y_test.shape)


lst_model = [LR, KNR]
#промоделируем
print('Модели регрессии с нормализованными данными с использованием всех признаков')
modeling(lst_model, x_train, x_test, y_train, y_test)

#построим модели с нормализацией данных на двух признаках
x2 = x.drop(columns = ['address.city_or_locality', 'address.cityType_or_localityType',
                       'address.street', 'address.streetType'])
#разделим данные на признаки (фичи) и цель (target)
x_train, x_test, y_train, y_test = train_test_split(scaller(x2), y, test_size=.2, random_state=0)
print('X Train : ', x_train.shape)
print('X Test  : ', x_test.shape)
print('Y Train : ', y_train.shape)
print('Y Test  : ', y_test.shape)

#промоделируем
print('Модели регрессии с нормализованными данными с использованием двух признаков')
modeling(lst_model, x_train, x_test, y_train, y_test)

#нормализуем все данные для нейронной сети
data = scaller(data)
x = data.drop(columns = ['right.dealPrice'])
y = data['right.dealPrice']
#разобъем данные на тренировочную и тестовую выборки c нормализованными выходными данными
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=0)
print('X Train : ', x_train.shape)
print('X Test  : ', x_test.shape)
print('Y Train : ', y_train.shape)
print('Y Test  : ', y_test.shape)
 #создадим нейронную сеть 
def MLPR (x_train, x_test, y_train, y_test):
     model = MLPRegressor(hidden_layer_sizes=(100,), activation='relu',
                           solver = 'adam', learning_rate_init=.01,
                           max_iter=2000, batch_size=3, random_state=18)
    #обучение дерева решений   
     model.fit(x_train, y_train)
        #расчет на тестовой выборке
     y_pred = model.predict(x_test)
     y_pred2 = model.predict(x_train)
     print('Нейронные сети с учетом всех признаков')
     print('r2_score:', r2_score(y_train, y_pred2))
     print('MAPE:', mean_absolute_percentage_error(y_test, y_pred)*100)
     return y_pred

MLPR (x_train, x_test, y_train, y_test)






