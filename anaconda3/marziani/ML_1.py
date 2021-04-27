import pandas as pd
import numpy as np

#### 1. Scegliere i dati
nomefile = './marziani.csv'
data = pd.read_csv(nomefile)
print(data.head())                  # per farsi un'idea del contenuto
print(">>colonne: ", data.columns)  # per vedere le intestazioni di colonna
print(">>tipi\n",data.dtypes)       # per vedere i tipi

#### 2. PREPARAZIONE DEI DATI
### ANALISI DEI DATI
print(">>describe Specie")
print(data.specie.unique())        # quali sono le specie
print(data['specie'].describe())   # quanti campioni ci sono

###
for specie in data.specie.unique():     # per ogni specie
    dati = data[data['specie'] == specie]
    print('>>', specie)
    for x in data.columns:
        print(dati[x].describe() )      # per avere statistiche sui dati 


### RENDIAMO NUMERICHE LE LABEL
colori = np.sort(data['colore'].dropna().unique())  # colori in ordine alfabetico
print(colori)

d = data.copy()        # facciamo una copia dei dati originali

for k in range (len(colori)):           # sostituiamo ogni colore con l'indice
    d.loc[:,'colore'].replace(colori[k], k, inplace = True)
print(d.head())

### ANALISI PER SCELTA FEATURE
import seaborn as sns   # conda install seaborn

sns.pairplot(d, hue='specie', dropna=True) 

### ELIMINAZIONE DATI MANCANTI
# Colonne con dati mancanti
cols_with_missing = [col for col in d.columns if d[col].isnull().sum()]
print(cols_with_missing)
# caratteristiche scelte per la classificazione
cols_selected = ['peso','altezza', 'larghezza']
# si eliminano le righe con dati mancanti solo nelle colonne selezionate
d = data.dropna(axis='index', subset = cols_selected) 
print(d.shape)                
# Colonne rimanenti con dati mancanti
print([col for col in d.columns if d[col].isnull().sum()])

### INDIVIDUAZIONE X E y
X = d[['peso','altezza','larghezza']]
y = d['specie']
print(X.head())
print(y.head())

#### 3. SUDDIVISIONE TRAIN E TEST
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,
                    train_size=0.7, random_state=0)

print(X_train.head())
print("Numero di campioni in X: ",X_train.shape[0])    
print(y_train.value_counts())       # quanti valori per ogni specie
print(y_train.head())

### STANDARIZZAZIONE
pd.options.display.float_format = '{:.3f}'.format #visualizza i dati con solo 3 decimali

m = X_train.mean()
print(f">>Media: \n{m}")
s = X_train.std()
print(f">>Deviazione standard: \n{s}")

X_train_std = ((X_train-m)/s)     # normalizziamo
print(f">>X train Normalizzato \n {X_train_std.describe()}")

## VERIFICA 
import seaborn as sns  
import matplotlib.pyplot as plt

for col in X_train.columns:
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(5,2)) #condividi asse y 
    fig.suptitle(col,y=0)  #titolo posizionato in basso
    # titolo dei due grafici affiancati
    axes[0].set_title("dati iniziali"); axes[1].set_title("dati normalizzati") 
    # eliminiamo tutte le altre etichette
    axes[0].set_xlabel(' ');  axes[1].set_xlabel(' ') 
    # disegniamo gli istrogrammi e la curva KDE = Kernel Density Estimation
    sns.histplot(ax=axes[0], x=X_train[col], kde=True) 
    sns.histplot(ax=axes[1], x=X_train_std[col], kde=True)

## NORMALIZZAZIONE DEL TEST SET
X_test_std = ((X_test-m)/s)     # normalizziamo
print(f">>X test Normalizzato \n {X_test_std.describe()}")

pd.options.display.float_format = None # ripristina la visualizzazione di default
