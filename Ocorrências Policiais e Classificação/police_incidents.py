#Análise Exploratória em Python e Aplicação de Algoritmo de Classificação
# Base de Dados de Segurança Pública da cidade de Cary (EUA)#

#link do dataset: https://data.opendatasoft.com/explore/dataset/cpd-incidents%40townofcary/information/?disjunctive.crime_category&disjunctive.crime_type&disjunctive.crimeday&disjunctive.district&disjunctive.offensecategory&disjunctive.violentproperty&disjunctive.total_incidents

#importar pacotes importantes para o trabalho#
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sb
import colorsys
plt.style.use('seaborn-talk')
import warnings
warnings.filterwarnings("ignore")

#alterar as configurações de coluna para mostrar mais colunas#
desired_width=320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns',37)

#importar o documento#
df = pd.read_csv("cpd-incidents.csv", sep = ";")


# Parte 1 - Se familiarizando com o dataset
#Verificar o começo da tabela
#print(df.head(10))

#Verificar os tipos de dados das colunas
df.info()
print(' ')
#Verificar nulos
print("Número de nulos")
print(df.isnull().sum())
print(' ')


#Resumo estatístico

print(df.describe())
print('')

#Deletando colunas com mais de 10 mil nulos
df = df.drop(columns=['chrgcnt','Neighborhd_ID','Apartment_Complex','Subdivisn_ID','activity_date','phxRecordStatus','PhxStatus'], axis=1)

#Deletando linhas com nulos
df = df.dropna()

#Deletando linhas com ano anteriores à 2000
df = df[df.year != 2000]
df = df[df.year != 1999]
df = df[df.year != 1998]
df = df[df.year != 1997]
df = df[df.year != 1994]
df = df[df.year != 1977]
df = df[df.year != 1988]

#Transformando anos em inteiros
df['year'] = df['year'].astype('int')

#Adicionando coluna de numero da semana
weekdays = [
    (df['CrimeDay'] == 'SUNDAY'),
(df['CrimeDay'] == 'MONDAY'),
(df['CrimeDay'] == 'TUESDAY'),
(df['CrimeDay'] == 'WEDNESDAY'),
(df['CrimeDay'] == 'THURSDAY'),
(df['CrimeDay'] == 'FRIDAY'),
(df['CrimeDay'] == 'SATURDAY')
]
weeknum = [1,2,3,4,5,6,7]

df['Week_Day'] = np.select(weekdays,weeknum)

#Verificar o começo da tabela 2
print(df.head(10))
print(' ')


#Visualições das frequências destacadas

# Definindo a quantidade
labels = df.Crime_Category.value_counts().index
num = len(df.Crime_Category.value_counts().index)

# Criando a lista de cores
listaHSV = [(x*1.0/num, 0.5, 0.5) for x in range(num)]
listaRGB = list(map(lambda x: colorsys.hsv_to_rgb(*x), listaHSV))

# Gráfico de Pizza 2
fatias, texto, autotextos = plt.pie(df.Crime_Category.value_counts().head(5), autopct='%1.1f%%',startangle = 90)
plt.axes().set_aspect('equal', 'datalim')
plt.legend(fatias, labels, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
plt.title("Frequência de ocorrência dos Top 5 crimes por categoria")
plt.show()

# Definindo a quantidade 2
labels1 = df.Crime_Type.value_counts().index
num1 = len(df.Crime_Type.value_counts().index)

# Gráfico de Pizza 2
fatias1, texto1, autotextos1 = plt.pie(df.Crime_Type.value_counts().head(5), autopct='%1.1f%%',startangle = 90)
plt.axes().set_aspect('equal', 'datalim')
plt.legend(fatias1, labels1,  loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
plt.title("Frequência de ocorrência dos Top 5 crimes por tipo")
plt.show()


# Definindo a quantidade 3
labels2 = df.Residential_Subdivision.value_counts().index
num2 = len(df.Residential_Subdivision.value_counts().index)

# Gráfico de Pizza 3
fatias2, texto2, autotextos2 = plt.pie(df.Residential_Subdivision.value_counts().head(5), autopct='%1.1f%%',startangle = 90)
plt.axes().set_aspect('equal', 'datalim')
plt.legend(fatias2, labels2,  loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
plt.title("Frequência de ocorrência dos Top 5 Locais")
plt.show()


#Histograma de ocorrências ao longo dos anos
y = len(df.year.value_counts().index)
df.year.hist(bins = 70)
plt.xlabel("Anos")
plt.ylabel("Número de ocorrências")
plt.title("Ocorrência de crimes ao longo dos anos")
plt.show()

#Histograma de ocorrências ao longo dos dias da semana
y1 = len(df.Week_Day.value_counts().index)
df.Week_Day.hist(bins = 70)
plt.xlabel("Dias da Semana")
plt.ylabel("Número de ocorrências")
plt.title("Ocorrência de crimes ao longo da semana (TOTAL)")
plt.show()

#Etapa 2 - Segmentação
# O objetivo é criar um dataset com as Subdivisões Residenciais como index e traçar o nivel de segurança de cada uma das localidades

df1 = df[['Residential_Subdivision','Crime_Type','year']].copy()

#Adicionando colunas de somataria para as colunas selecionadas
crime_other = [(df1['Crime_Type'] == 'LARCENY - ALL OTHER LARCENY'),(df1['Crime_Type'] != 'LARCENY - ALL OTHER LARCENY')]
crime_other_num = [1,0]
df1['Other_Larceny'] = np.select(crime_other,crime_other_num)

crime_vehicle = [(df1['Crime_Type'] == 'LARCENY - FROM MOTOR VEHICLE'),(df1['Crime_Type'] != 'LARCENY - FROM MOTOR VEHICLE')]
crime_vehicle_num = [1,0]
df1['Vehicle_Larceny'] = np.select(crime_vehicle,crime_vehicle_num)

crime_shoplifting = [(df1['Crime_Type'] == 'LARCENY - SHOPLIFTING'),(df1['Crime_Type'] != 'LARCENY - SHOPLIFTING')]
crime_shoplifting_num = [1,0]
df1['Shoplifting_Larceny'] = np.select(crime_shoplifting,crime_shoplifting_num)

crime_vandalism = [(df1['Crime_Type'] == 'VANDALISM'),(df1['Crime_Type'] != 'VANDALISM')]
crime_vandalism_num = [1,0]
df1['Vandalism'] = np.select(crime_vandalism,crime_vandalism_num)

crime_burglary = [(df1['Crime_Type'] == 'BURGLARY - FORCIBLE ENTRY'),(df1['Crime_Type'] != 'BURGLARY - FORCIBLE ENTRY')]
crime_burglary_num = [1,0]
df1['Burglary'] = np.select(crime_burglary,crime_burglary_num)

crime_physical = [(df1['Crime_Type'] == 'SIMPLE PHYSICAL ASSAULT'),(df1['Crime_Type'] != 'SIMPLE PHYSICAL ASSAULT')]
crime_physical_num = [1,0]
df1['Physical_Assault'] = np.select(crime_physical,crime_physical_num)

crime_damage_property = [(df1['Crime_Type'] == 'VANDALISM - DAMAGE TO PROPERTY'),(df1['Crime_Type'] != 'VANDALISM - DAMAGE TO PROPERTY')]
crime_damage_property_num = [1,0]
df1['Damage_Property'] = np.select(crime_damage_property,crime_damage_property_num)

crime_fraud_creditcard = [(df1['Crime_Type'] == 'FRAUD - CREDIT CARD/ATM'),(df1['Crime_Type'] != 'FRAUD - CREDIT CARD/ATM')]
crime_fraud_creditcard_num = [1,0]
df1['Fraud_Credit_Card'] = np.select(crime_fraud_creditcard,crime_fraud_creditcard_num)

crime_fraud_other = [(df1['Crime_Type'] == 'FRAUD - ALL OTHER'),(df1['Crime_Type'] != 'FRAUD - ALL OTHER')]
crime_fraud_other_num = [1,0]
df1['Fraud_Other'] = np.select(crime_fraud_other,crime_fraud_other_num)

crime_assault = [(df1['Crime_Type'] == 'ASSAULT - SIMPLE - ALL OTHER'),(df1['Crime_Type'] != 'ASSAULT - SIMPLE - ALL OTHER')]
crime_assault_num = [1,0]
df1['Assault'] = np.select(crime_assault,crime_assault_num)

df1 = df1.drop(columns=['Crime_Type','year'])

print(df1.head())

group1 = df1.groupby('Residential_Subdivision')

print(group1.head())


#Etapa 2 - Segmentação
# O objetivo é criar um dataset com as Subdivisões Residenciais como index e traçar o nivel de segurança de cada uma das localidades

#Criando um dataset base

total_crimes = df['Residential_Subdivision'].value_counts()
df_class = pd.DataFrame(total_crimes)

#Adicionando colunas para os crimes
crime1 = df.groupby('Residential_Subdivision').Crime_Type.value_counts().unstack(fill_value=0).loc[:,'LARCENY - ALL OTHER LARCENY']
crime2 = df.groupby('Residential_Subdivision').Crime_Type.value_counts().unstack(fill_value=0).loc[:,'LARCENY - FROM MOTOR VEHICLE']
crime3 = df.groupby('Residential_Subdivision').Crime_Type.value_counts().unstack(fill_value=0).loc[:,'LARCENY - SHOPLIFTING']
crime4 = df.groupby('Residential_Subdivision').Crime_Type.value_counts().unstack(fill_value=0).loc[:,'VANDALISM']
crime5 = df.groupby('Residential_Subdivision').Crime_Type.value_counts().unstack(fill_value=0).loc[:,'BURGLARY - FORCIBLE ENTRY']
crime6 = df.groupby('Residential_Subdivision').Crime_Type.value_counts().unstack(fill_value=0).loc[:,'SIMPLE PHYSICAL ASSAULT']
crime7 = df.groupby('Residential_Subdivision').Crime_Type.value_counts().unstack(fill_value=0).loc[:,'VANDALISM - DAMAGE TO PROPERTY']
crime8 = df.groupby('Residential_Subdivision').Crime_Type.value_counts().unstack(fill_value=0).loc[:,'FRAUD - CREDIT CARD/ATM']
crime9 = df.groupby('Residential_Subdivision').Crime_Type.value_counts().unstack(fill_value=0).loc[:,'FRAUD - ALL OTHER']
crime10 = df.groupby('Residential_Subdivision').Crime_Type.value_counts().unstack(fill_value=0).loc[:,'ASSAULT - SIMPLE - ALL OTHER']

crimes = [crime1,crime2,crime3,crime4,crime5,crime6,crime7,crime8,crime9,crime10]

for x in crimes:
    df_class = pd.concat([df_class, x], axis=1)

#Somando as colunas criadas e as deletando

df_crimes = df_class.drop(['Residential_Subdivision'], axis=1)

df_class['top_crimes'] = df_crimes.sum(axis=1)

colum = ['LARCENY - ALL OTHER LARCENY','LARCENY - FROM MOTOR VEHICLE','LARCENY - SHOPLIFTING','VANDALISM','BURGLARY - FORCIBLE ENTRY',
         'SIMPLE PHYSICAL ASSAULT','VANDALISM - DAMAGE TO PROPERTY','FRAUD - CREDIT CARD/ATM','FRAUD - ALL OTHER','ASSAULT - SIMPLE - ALL OTHER']

for x in colum:
    df_class = df_class.drop(columns=[x])

#Adicionando colunas para os anos para realizar comparações

anos = [2001,2010,2019]
years = []
for x in anos:
    ano = df.groupby('Residential_Subdivision').year.value_counts().unstack(fill_value=0).loc[:,x]
    years.append(ano)

for x in years:
    df_class = pd.concat([df_class, x], axis=1)

#Adicionando as variações de ocorrências entre anos

var_01_19 = (df_class[2019].div(df_class[2001]))-1
var_01_10 = (df_class[2010].div(df_class[2001]))-1
var_10_19 = (df_class[2019].div(df_class[2010]))-1

df_class: Union[DataFrame, Series] = pd.concat([df_class,var_01_19], axis=1)
df_class = df_class.rename(columns={0:'var_01.19'})
df_class: Union[DataFrame, Series] = pd.concat([df_class,var_01_10], axis=1)
df_class = df_class.rename(columns={0:'var_01.10'})
df_class: Union[DataFrame, Series] = pd.concat([df_class,var_10_19], axis=1)
df_class = df_class.rename(columns={0:'var_10.19'})

df_class = df_class.fillna(0)

#Renomeando colunas
df_class = df_class.rename(columns={'Residential_Subdivision':'Total_Crimes'})
df_class = df_class.rename(columns={'top_crimes':'%Top10_Crimes'})
print(df_class)

#Verificando a disperção dos dados

df_box = df_class.drop(['var_01.19','var_01.10','var_10.19'],axis=1)
sb.boxplot(data=df_box)
#plt.show()

#Eliminando outliers

df_box = df_box[df_box.Total_Crimes < 176]
sb.boxplot(data=df_box)
#plt.show()

print(df_box.describe())

#Criando função para classificação de nível de segurança das divisões residenciais

def nivel_seg(row):
    if row['Total_Crimes'] < 55 and row['var_10.19'] > row['var_01.10']:
        return 1 #Segurança Alta'
    if row['Total_Crimes'] < 55 and row['var_10.19'] < row['var_01.10']:
        return 2 #'Segurança Média'
    if row['Total_Crimes'] > 55 and row['var_10.19'] > row['var_01.10']:
        return 3 #'Segurança Baixa'
    if row['Total_Crimes'] > 55 and row['var_10.19'] < row['var_01.10']:
        return 4 #'Inseguro'
    else:
        return 0 # 'Neutro'

def evolução(row):
    if row['var_01.19'] <= 0:
        return 1 #'Melhorando'
    if row['var_01.19'] > 0:
        return 0 #'Piorando'

df_class['Ind_Segurança'] = df_class.apply(nivel_seg, axis=1)
df_class['Evolução'] = df_class.apply(evolução, axis=1)

print(df_class['Ind_Segurança'].value_counts())

np.all(np.isfinite(df_class))
# Removendo valores infitos

df_class = df_class.replace([np.inf, -np.inf], 0)
print(df_class)

#Transformando Index em Coluna de Subdivisão Residencial e resetando o index

df_class = df_class.reset_index()
df_class = df_class.rename(columns={'index':'Residential_Subdivision'})
print(df_class)

#Aplicando algoritmos de classificação

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

#Criando elementos de feature e de target

df_feature = df_class.drop(['Residential_Subdivision',2001,2010,2019,'Ind_Segurança','Evolução'],axis=1)
df_target = df_class['Ind_Segurança']

print(df_feature.shape)
print(df_target.shape)

#Criando o Modelo de Classificação usando Random Forest
clf = RandomForestClassifier()
clf.fit(df_feature,df_target)

#Importância de cada feature
print('Importância de cada Featured')
print(clf.feature_importances_)

#Fazendo predição

print(clf.predict(df_feature[0:9]))
print(clf.predict_proba(df_feature[0:9]))

#Dividindo o dataset (80/20)

feature_train, feature_test, target_train, target_test = train_test_split(df_feature,df_target, test_size=0.2)

print(feature_train.shape,target_train.shape)
print(feature_test.shape,target_test.shape)

#Reconstruindo Random Forest

clf.fit(feature_train,target_train)

#Fazendo predição
print('Predições do datatest')
print(clf.predict(feature_test[0:2]))
print(target_test[0:2])

print('Acurácia do Random Forest')
print(clf.score(feature_test,target_test))

# Etapa 3.2
# Testando algoritmo Support Vector Machine (SVM)

from sklearn.svm import SVC
cls = SVC()
cls.fit(feature_train,target_train)
print('Acurácia do SVM')
print(cls.score(feature_test,target_test))

# Etapa 3.3
# Testando algoritmo k-nearest
#Normalizando os dados para caber no algoritmo

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

sc_feature = StandardScaler()
feature_train = sc_feature.fit_transform(feature_train)
feature_test = sc_feature.transform(feature_test)

import math
print('')
print('Raiz Quadrada de datatest para estipular o valor de k')
print(math.sqrt(len(target_test)))

#Criando o modelo inicial
clk = KNeighborsClassifier(n_neighbors=13, p=2,metric="euclidean")
clk.fit(feature_train,target_train)

feature_pred = clk.predict((feature_test))

cm = confusion_matrix(target_test, feature_pred)
print(cm)
print('Acurácia do K-Neighbours')
print(clk.score(feature_test,target_test))

