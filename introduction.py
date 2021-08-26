import pandas as pd
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing, svm, model_selection
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')

df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume',]]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df ['Adj. Low'] * 100
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df ['Adj. Open'] * 100

# Me quedo con la info que me interesa
df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

# La columna que voy a predecir
forecast_col = 'Adj. Close'
# Reemplazo los na como -99999 para que todo sean numeros
df.fillna(-99999, inplace=True)

# La predicción será el 10% de todo lo que tengo (35)
forecast_out = math.ceil(0.01*len(df))

# Agrego una columna label y subo los Adj. CLose 35 
# espacios para rellenar con la info que ya tengo 
df['label'] = df[forecast_col].shift(-forecast_out)

# X es toda la info menos label, porque es lo que voy a predecir
X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:] #los ultimos 34
X = X[:-forecast_out] #todo menos los ultimos 34
df.dropna(inplace=True)
# y es solo label, que es lo que voy a predicir, eliminando los na antes para que no interfiera en los calculos
y = np.array(df['label'])

# Mezcla y divide el df 
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# Modelo Linear Regression diviendo los trabajos tanto como mi computadora pueda
## clf = LinearRegression(n_jobs=-1)
# clf = svm.SVR(kernel='poly')
## clf.fit(X_train, y_train) #train
# Guardar el modelo entrenado
## with open('linearregression.pickle', 'wb') as f:
##    pickle.dump(clf, f)

pickle_in = open('linearregression.pickle', 'rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(X_test, y_test) #test
# print(accuracy)

forecast_set = clf.predict(X_lately)

# Agarro la ultima fecha y convierto a timestamp para liego sumarle 1 dia en seg
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

# Creo la columna Forecast de predicciones y lleno de nan
df['Forecast'] = np.nan
for i in forecast_set:
    # Convierto next_unix a tipo datetime
    next_date =  datetime.datetime.fromtimestamp(next_unix)
    # ubica el indice de la siguiente fecha y rellena las columnas con nan
    # excepto la ultima (-1), y pone lo que hay en forecast_set en la ultima 
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+ [i]
    # actualiza next_unix sumandole 1 dia en seg
    next_unix += one_day

# Grafico
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()



