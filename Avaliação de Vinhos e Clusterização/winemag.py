#Análise Exploratória em Python e Aplicação de Algoritmo de Clusterização
# Base de Dados de Vinhos#

#link do dataset: https://data.opendatasoft.com/explore/dataset/open-beer-database%40public-us/information/?sort=-srm

#importar pacotes importantes para o trabalho#
import numpy as np
import pandas as pd
import matplotlib as mpl
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
pd.set_option('display.max_columns',22)

#importar o documento#
df = pd.read_csv("winemag.csv", sep =",")

#Parte 1 - Se familiarizando com o dataset
#Verificar o começo da tabela
print(df.head(10))

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

#Deletando colunas com mais de 8k nulos exceto
df_fil = df.drop(columns=['Unnamed: 0','designation','region_1','region_2','taster_name','taster_twitter_handle'], axis=1)

#Deletando linhas com nulos
df_fil = df_fil.dropna()

print(df_fil)

#Respondendo perguntas básicas do dataset
print('Qual é o número de variedades?')
print(df_fil['variety'].nunique())
print(' ')
print('Quais são as variedades mais comuns?')
print(df_fil['variety'].value_counts().head(10))
print(' ')
print('Qual é o número de países do dataset?')
print(df_fil['country'].nunique())
print(' ')
print('Quais são os maiores produtores do dataset?')
print(df_fil['country'].value_counts().head(10))
print(' ')

#Primeiras visualizações

df_pais = df_fil['country'].value_counts().head(10)
df_pais = df_pais.reset_index()
df_pais = df_pais.drop(['country'],axis=1)
df_to_pas = df_fil[df_fil['country'].isin(df_pais['index'])]
sb.countplot(x='country',data=df_to_pas)
plt.title('Número de Vinhos por País dos 10 maiores produtores')
plt.show()

sb.histplot(x='points',data=df_fil)
plt.title('Distribuição dos pontos')
plt.show()

valor = list(range(0,201))
df_val = df_fil[df_fil.price.isin(valor)]
sb.histplot(x='price',data=df_val)
plt.title('Distribuição dos valores dos vinhos abaixo de $200')
plt.show()

sb.lmplot(x='price',y='points',data=df_fil)
plt.title('Distribuição de preços dos vinho por pontuação')
plt.show()

sb.lmplot(x='price',y='points',data=df_val)
plt.title('Distribuição de preços dos vinho por pontuação abaixo de $200')
plt.show()

#Aplicando modelos de ML para Clusterização
#Preparando os dados
#Para essa classificação vamos usar os Top 10 países e os 15 top variedades de vinho
df_clus = df_to_pas.drop(['description','province','title','winery'],axis=1)

print('Qual é o número de vinhos das top 15 variedades nos países escolhidos?')
print(np.sum(df_clus['variety'].value_counts().head(15)))
df_vinho = df_fil['variety'].value_counts().head(15)
df_vinho = df_vinho.reset_index()
df_vinho = df_vinho.drop(['variety'],axis=1)
df_clus = df_clus[df_clus['variety'].isin(df_vinho['index'])]


#Transformando categorias de países e variedades em números
print(df_clus['country'].value_counts())
print(df_clus['variety'].value_counts())
df_clus['country'] = df_clus['country'].map({'US':1,'France':2,'Italy':3,'Chile':4,'Argentina':5,'Portugal':6,'Germany':7,'Spain':8,'Australia':9,'Austria':10})
df_clus['variety'] = df_clus['variety'].map({'Pinot Noir':1,'Chardonnay':2,'Cabernet Sauvignon':3,'Red Blend':4,'Bordeaux-style Red Blend':5,
                                             'Riesling':6,'Sauvignon Blanc':7,'Syrah':8,'Rosé':9,'Merlot':10,'Zinfandel':11,'Malbec':12,
                                             'Sangiovese':13,'Nebbiolo':14,'Portuguese Red':15})
print(df_clus.describe())
sb.lmplot(x='price',y='points',data=df_clus)
plt.title('Distribuição de preços dos vinho por pontuação')
plt.show()

#Eliminando vinhos > $500
valor2 = list(range(0,501))
df_clus = df_clus[df_clus.price.isin(valor2)]
sb.lmplot(x='price',y='points',data=df_clus)
plt.title('Distribuição de preços dos vinho por pontuação (A baixo de $500)')
plt.show()

#Normalizando os dados
from sklearn.preprocessing import normalize
df_norm = normalize(df_clus)

#Verificando o valor de K com método do cotovelo

variacao = []
ranger = list(range(1,10))
for x in ranger:
    from sklearn.cluster import KMeans
    k_mean = KMeans(n_clusters=x)
    k_mean.fit(df_norm)
    variacao.append(k_mean.inertia_)

plt.plot(ranger,variacao,marker='*')
plt.xlabel('Valor de K')
plt.ylabel('Variação')
plt.show()

#Aplicando o algoritmo K-Means
from sklearn.cluster import KMeans
k_mean = KMeans(n_clusters=5)
df_k = k_mean.fit_predict(df_norm)
df_clus['KMeans'] = df_k

sb.lmplot(x='price',y='points',data=df_clus,hue='KMeans')
plt.title('Distribuição de preços dos vinho por pontuação')
plt.show()

#Preparar para Aplicar o algoritmo DBSCAN
#Descobrindo o valor de episilon
#from sklearn.neighbors import NearestNeighbors

#vizinhos = NearestNeighbors(n_neighbors=2)
#vizi = vizinhos.fit(df_norm)
#distances, indices = vizi.kneighbors(df_norm)

#distances = np.sort(distances, axis=0)
#distances = distances[:,1]
#plt.plot(distances)
#plt.show()

#Aplicando o algoritmo DBSCAN
#from sklearn.cluster import DBSCAN
#clustering = DBSCAN(eps=0.3,min_samples=5).fit(df_norm)
#cluster = clustering.labels_
#print(len(cluster))

#Aplicando Clusterização Hierárquica
#from scipy.cluster.hierarchy import dendrogram, linkage
#linked = linkage(df_norm,'single')
#labellist = range(1,6)
#dendrogram(linked,
#            orientation='top',
#           labels=labellist,
#           distance_sort='descending',
#           show_leaf_counts=True)
#plt.show()