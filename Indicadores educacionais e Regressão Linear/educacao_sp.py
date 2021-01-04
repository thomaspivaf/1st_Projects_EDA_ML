#Análise Exploratória em Python e Aplicação de Algoritmo de Regressão
# Base de Dados de Indicadores de Educação no Município de SP#

#link do dataset1: http://dados.prefeitura.sp.gov.br/pt_PT/dataset/microdados-psp
#link do dataset1: http://dados.prefeitura.sp.gov.br/pt_PT/dataset/ideb-e-prova-brasil-na-rede-municipal-de-ensino

#importar pacotes importantes para o trabalho#
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-talk')
import warnings
warnings.filterwarnings("ignore")

#alterar as configurações de coluna para mostrar mais colunas#
desired_width=320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns',22)

#importar o 1º documento
df = pd.read_csv("ideb_sp_2015.csv", sep =";")

#Parte 1 - Se familiarizando com o dataset do Ideb 2015
#Verificar o começo da tabela
print(df.head(10))

#Verificar os tipos de dados das colunas
df.info()
print(' ')

#Verificar nulos
print("Número de nulos")
print(df.isnull().sum())
print(' ')

#Lidando com os nulos com alternativas ao deletamento de linhas
df['NSE'].fillna(df['NSE'].mode()[0], inplace = True)
df_cols = ['taxa_aprov_1_a_5','taxa_aprov_1','taxa_aprov_2','taxa_aprov_3','taxa_aprov_4','taxa_aprov_5','indicador_rendimento_(P)',
           'provabr_matematica','provabr_portugues','provabr_media_padronizada','IDEB_2015_(N x P)']
for x in df_cols:
    df[x].fillna(df[x].median(), inplace=True)

#Resumo estatístico
print(df.describe())
print('')

#Eliminar colunas pouco relevantes
df = df.drop(['Código do Município','Código da Escola'],axis=1)

#Primeras visualizações
corr=df.corr()
sns.heatmap(corr,cmap = "YlOrRd",linewidths=0.1)
plt.title('Correlações entre variáveis')
plt.show()

taxa_aprov = df.drop(['CD_UNIDADE_EDUCACAO','NSE','indicador_rendimento_(P)','provabr_matematica','provabr_portugues','provabr_media_padronizada','IDEB_2015_(N x P)'],axis=1)
sns.violinplot(data=taxa_aprov)
plt.title('Taxa de Aprovação das Séries')
plt.ylabel('Taxa de Aprovação')
plt.show()

sns.histplot(data=df['indicador_rendimento_(P)'])
plt.title('Distribuição do Indicador de Rendimento')
plt.ylabel('Total')
plt.show()

sns.displot(data=df['NSE'],color='green')
plt.title('Distribuição do Nível Socioeconômico')
plt.xlabel('Nível Socioeconômico')
plt.ylabel('Total')
plt.show()

sns.lmplot(x='provabr_matematica',y='provabr_portugues',data=df)
plt.title('Correlação entre Notas de Matemática e Português')
plt.xlabel('Nota em Matemática')
plt.ylabel('Nota em Português')
plt.show()

sns.histplot(data=df['IDEB_2015_(N x P)'])
plt.title('Distribuição do IDEB')
plt.ylabel('Total')
plt.show()

#Parte 2 - Trazendo 2ª tabela com dados da Prova São Paulo 2017
#importar o 2º documento
df_p = pd.read_csv("MCD_PSP_2017.csv", sep =";")

#Se familiarizando com o dataset da Prova SP
#Verificar o começo da tabela
print(df_p.head(10))

#Verificar os tipos de dados das colunas
df_p.info()
print(' ')
#Deletando colunas pouco relevantes - Segundo o Dicionário de Dados
df_p = df_p.drop(['ANO','DRE','CD_TURMA','CD_TURNO','DESC_TURNO','CD_SERIE','NOME_TURMA','CD_CICLO_ENSINO','DESC_CICLO_ENSINO',
                  'CD_ALUNO_SME','CD_INEP_ALUNO','POSSUI_NEE','DESC_TP_PROVA','PROVA_LP','CADERNO_LP','PROVA_MAT',
                  'CADERNO_MAT','PROVA_CIE','CADERNO_CIE','VETOR_LP','VETOR_CORRIGIDO_LP','VETOR_MAT','VETOR_CORRIGIDO_MAT',
                  'VETOR_CIE','VETOR_CORRIGIDO_CIE','PESO_ALUNO_LP','PESO_ALUNO_MAT','PESO_ALUNO_CIE','PESO_ALUNO_RED'],axis=1)

#Verificar nulos
print("Número de nulos")
print(df_p.isnull().sum())
print(' ')


#Verificando as presenças nas provas antes de lidar com os valores nulos
num_pre = {}
presenca = ['IN_PRESENCA_LP','IN_PRESENCA_MAT','IN_PRESENCA_CIE','IN_PRESENCA_RED']
for x in presenca:
    num_pre[x] = df_p[x].value_counts().tolist()
pres = pd.DataFrame(data=num_pre)
pres= pres.transpose()

indx = np.arange(len(pres))
presentes = pres.iloc[:,0]
ausentes = pres.iloc[:,1]

prese = plt.bar(x=indx,height=presentes,width=0.35,label='Presença')
ausen = plt.bar(x=indx,height=ausentes,width=0.35,bottom=presentes,label='Ausência')
plt.xlabel('Provas')
plt.ylabel('Presença')
plt.xticks(indx,presenca)
plt.title('Presença dos Alunos por Tipo de Prova')
plt.legend()
plt.show()

#Podemos assumir que os valores nulos são na verdade zeros equivalente à ausencia do aluno em determinada prova
df_p = df_p.fillna(0)
print("Número de nulos")
print(df_p.isnull().sum())

#Corrigindo erro de leitura nas colunas de nível
df_p = df_p.replace('B�sico', 'Básico', regex=True)
df_p = df_p.replace('Avan�ado', 'Avançado', regex=True)
df_p = df_p.replace(',','.', regex=True)

colunas = ['PER_ALUNO_LP','PER_ALUNO_MAT','PER_ALUNO_CIE','PROFICIENCIA_LP','PROFICIENCIA_MAT','PROFICIENCIA_CIE']
for x in colunas:
    df_p[x] = df_p[x].astype(float)

#Explorando as colunas além dos valores nulos

#Respondendo perguntas básicas do dataset
print(df_p.head())
print('')

print('Qual é o número de unidades educacionais?')
print(df_p['CD_UNIDADE_EDUCACAO'].nunique())
print(' ')

print('Quais são os tipos de unidade??')
print(df_p['TIPO_ESCOLA'].value_counts())
print(' ')

print('Quais são os tipos de modalidade??')
print(df_p['MODALIDADE'].value_counts())
print(' ')

print('Quais são os tipos de segmento de modalidade??')
print(df_p['MODALIDADE_SEGMENTO'].value_counts())
print(' ')

print('Quais são as séries?')
print(df_p['DESC_SERIE'].value_counts())
print(' ')

print('Qual é a prova com mais abaixo do básico e avançado?')
nivel_pro = {}
nivel = ['DESC_NIVEL_LP','DESC_NIVEL_MAT','DESC_NIVEL_CIE','DESC_NIVEL_RED']
for x in nivel:
    nivel_pro[x] = df_p[x].value_counts().tolist()
nivel_pro = pd.DataFrame(data=nivel_pro)
nivel_pro.index = ['Adequado','Básico','Abaixo do Básico','Ausente','Avançado']
print(nivel_pro)


#Segundas visualizações
corr= df_p.corr()
sns.heatmap(corr,cmap = "YlOrRd",linewidths=0.1)
plt.title('Correlações entre variáveis')
plt.show()

distritos =df_p['NOME_DISTRITO'].value_counts().head(10)
distritos = distritos.reset_index()

sns.barplot(x=distritos['index'],y=distritos['NOME_DISTRITO'],data=distritos)
plt.title('10 Distritos com mais alunos')
plt.xlabel('Distrito')
plt.ylabel('Total de Alunos')
plt.show()

series = df_p['DESC_SERIE'].value_counts()
series = series.reset_index()

sns.barplot(x=series['index'],y=series['DESC_SERIE'],data=series)
plt.title('Distribuição das séries')
plt.xlabel('Séries')
plt.ylabel('Total de Alunos')
plt.show()


sns.scatterplot(x='PROFICIENCIA_LP',y='PROFICIENCIA_MAT',data=df_p)
plt.title('Correlação entre proficiência em Português e Matemática')
plt.xlabel('Proficiência em Português')
plt.ylabel('Proficiência em Matemática')
plt.show()

nivel_pro = nivel_pro.transpose()
print(nivel_pro)
ind = np.arange(len(nivel_pro))
adequado = nivel_pro.iloc[:,0]
basico = nivel_pro.iloc[:,1]
abaixo =nivel_pro.iloc[:,2]
avancado = nivel_pro.iloc[:,4]
aba = plt.bar(x=ind,height=abaixo,width=0.4,label='Abaixo do Básico',color='yellow')
bas = plt.bar(x=ind,height=basico,width=0.4,bottom=abaixo,label='Básico',color='yellowgreen')
ade = plt.bar(x=ind,height=adequado,width=0.4,bottom=abaixo+basico,label='Adequado',color='limegreen')
ava = plt.bar(x=ind,height=avancado,width=0.4,bottom=abaixo+basico+adequado,label='Avançado',color='green')
plt.xlabel('Nível nas Provas')
plt.ylabel('Total')
plt.title('Proeficiência dos alunos nas Provas')
plt.xticks(ind,nivel)
plt.legend()
plt.show()

#Etapa 3 - Preparando o dataset da prova sp para ser adicionado ao do ideb
#Eliminando valores que não existem na tabela do IDEB (Tudo o que não for do Fund.1)
df_p = df_p[df_p.MODALIDADE != 'ESPEC']
print(df_p['MODALIDADE'].value_counts())
df_p = df_p[df_p.MODALIDADE_SEGMENTO != 'Fund2']
print(df_p['DESC_SERIE'].value_counts())

#Excluindo colunas pouco relevantes
df_p = df_p.drop(['TIPO_ESCOLA','NOME_DISTRITO','MODALIDADE','MODALIDADE_SEGMENTO','IN_PRESENCA_LP','IN_PRESENCA_MAT',
                  'IN_PRESENCA_CIE','IN_PRESENCA_RED','ACERTOS_ALUNO_LP','ACERTOS_ALUNO_MAT','ACERTOS_ALUNO_CIE',
                  'PER_ALUNO_LP','PER_ALUNO_MAT','PER_ALUNO_CIE','NOTA_RED','DESC_NIVEL_LP','DESC_NIVEL_MAT',
                  'DESC_NIVEL_CIE','DESC_NIVEL_RED'],axis=1)

#Criando dataframe a partir das medianas dos indicadores por unidades educacional
df_prova = pd.DataFrame(df_p.groupby(['CD_UNIDADE_EDUCACAO']).mean())
df_prova = df_prova.reset_index()
print(df_prova)


#Preparando a tabela de Ideb para união
df_ideb = df.drop(['Sigla da UF','Nome do Município','DRE','Nome da Escola',
              'Rede','taxa_aprov_1_a_5','taxa_aprov_1','taxa_aprov_2','taxa_aprov_3','taxa_aprov_4','taxa_aprov_5'],axis=1)
df_ideb = df_ideb.sort_values(by=['CD_UNIDADE_EDUCACAO'])

#Eliminando linha inexistente na tabela da prova sp
df_ideb = df_ideb.drop([148])

#Corrigindo o index
df_ideb = df_ideb.reset_index()
df_ideb = df_ideb.drop(['index'],axis=1)
print(df_ideb)

#Etapa 4 - Unindo os datasets
df_reg = pd.concat([df_ideb,df_prova],axis=1)
print(df_reg)

#Verificar se as unidades de educação batem
def teste(row):
    if row['CD_UNIDADE_EDUCACAO'] == row['CD_UNIDADE_EDUCACAO1']:
        return 1
    else:
        return 0

df_reg['Teste'] = df_reg.apply(teste,axis=1)
print(df_reg['Teste'].value_counts())

#Exportar para verificar os erros no Excel
df_reg.to_csv('regressao.csv')

#Etapa 5 - Importando o dataset final
#Etapa 5 - Importando o dataset final

df_pronto = pd.read_csv("regressao.trab.csv", sep =";")
df_pronto = df_pronto.drop(['index','CD_UNIDADE_EDUCACAO'],axis=1)
print(df_pronto)

#Verificando a Correlação entre variáveis
#corr1 = df_pronto.corr()
#sns.heatmap(corr1,cmap = "YlOrRd",linewidths=0.1)
#plt.title('Correlações entre variáveis')
#plt.show()

sns.lmplot(x='provabr_portugues',y='PROFICIENCIA_LP',data=df_pronto)
plt.show()

sns.lmplot(x='provabr_matematica',y='PROFICIENCIA_MAT',data=df_pronto)
plt.show()

sns.lmplot(x='NSE',y='NIVEL_MEDIO',data=df_pronto)
plt.show()

sns.lmplot(x='IDEB_2015_(N x P)',y='PROFICIENCIA_MEDIA',data=df_pronto)
plt.show()

#Etapa 6 - Aplicando algoritmo de regressão linear simples
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#Selecionando as variáveis
X = df_pronto[['IDEB_2015_(N x P)']].values.reshape(-1,1)
Y = df_pronto[['NIVEL_MEDIO']].values.reshape(-1,1)

#Realizando o fit do modelo
linear = LinearRegression()
linear = linear.fit(X,Y)

#Testando o modelo
predicoes = linear.predict(X)
print(predicoes[:5])
plt.scatter(x=df_pronto['IDEB_2015_(N x P)'],y=df_pronto['NIVEL_MEDIO'],c='black')
plt.plot(df_pronto['IDEB_2015_(N x P)'],predicoes,c='blue',linewidth=2)
plt.show()

#Verificando as predições
print('Indicadores do 1º modelo')
import statsmodels.api as sm

XX = sm.add_constant(X)
est = sm.OLS(Y,XX)
est2 = est.fit()
print(est2.summary())

#Selecionando as variáveis do 2º modelo
X1 = df_pronto[['provabr_media_padronizada']]
Y1 = df_pronto[['PROFICIENCIA_MEDIA']]

#Dividindo o data set
x1_train,x1_test,y1_train,y1_test = train_test_split(X1,Y1,random_state=1)

#Realizando o fit do modelo
linear1 = LinearRegression()
linear1 = linear1.fit(x1_train,y1_train)

#Testando o modelo
predicoes1 = linear1.predict(x1_test)

#Verificando as predições
print('')
print('Indicadores do 2º modelo')
XA = sm.add_constant(X1)
estx = sm.OLS(Y1,XA)
estx2 = estx.fit()
print(estx2.summary())

#Selecionando as variáveis do 3º modelo
X2 = df_pronto[['NSE']]
Y2 = df_pronto[['PROFICIENCIA_MEDIA']]

#Dividindo o data set
x2_train,x2_test,y2_train,y2_test = train_test_split(X2,Y2,random_state=1)

#Realizando o fit do modelo
linear2 = LinearRegression()
linear2 = linear2.fit(x2_train,y2_train)

#Testando o modelo
predicoes2 = linear2.predict(x2_test)

#Verificando as predições
print('')
print('Indicadores do 3º modelo')
XB = sm.add_constant(X2)
esty = sm.OLS(Y2,XB)
esty2 = esty.fit()
print(esty2.summary())

#Fazendo regressão linear multivariada
X3 = df_pronto[['NSE','indicador_rendimento_(P)','provabr_media_padronizada']]
Y3 = df_pronto[['PROFICIENCIA_MEDIA']]

linearm = LinearRegression()
linearm = linearm.fit(X3,Y3)

#Verificando as predições
print('Indicadores do 4º modelo')
import statsmodels.api as sm

A = sm.add_constant(X3)
est0 = sm.OLS(Y3,A)
est00 = est0.fit()
print(est00.summary())