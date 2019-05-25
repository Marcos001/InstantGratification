'''
Marcos Vinícius dos Santos Ferreira
24/05/2019
'''

''' ================= DATA ========================= '''
''' basic libs '''
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

''' onde hot ecoding'''
from keras.utils import to_categorical

''' curva ROC '''
from sklearn.metrics import roc_curve, auc

''' meu modulo de funções '''
import modulos as md

''' configure  '''

vez = 3

path_root = '/media/marcos/Files - ntfs/datasets/InstantGratification/'


''' ================= DATA ========================= '''
print('read the data...')
df = pd.read_csv(path_root+'train.csv')


print('select features ans labels')
''' select features ans labels '''
features = df.ix[:,1:257]
labels = df.ix[:,257]

print('SHAPE -> Features :',features.shape ,'Labels :',labels.shape, 'df Rows :', len(df), 'df Columms :', len(df.columns))

''' selecionado uma amostra '''

size = 262144

print('pre-processing')
''' convert data in tensor '''
x = np.array(features.ix[:size,:])
y = np.array(labels.ix[:size])

''' tranform data in float '''
x = x.astype('float')

''' normalize your data '''
x -= x.mean(axis=0)
x /= x.std(axis=0)

''' one hot encoding '''
y = to_categorical(y)



''' split data in train, test e validation '''

ini, end = md.proportion_split(size, verbose=True)

x_train = x[:ini,]
x_test = x[ini:end,]
x_val = x[end:,]

y_train = y[:ini,]
y_test = y[ini:end,]
y_val = y[end:,]



''' ================= MODEL ========================= '''

num_classes = 2
epocas = 20
lote = 32
taxa_aprendizado = 0.001


print('define Neural Network')
nn = md.NeuralNetwork(x.shape[1],num_classes)

print('train NN')
hist = nn.fit(x_train,y_train, epochs=20, validation_data=(x_val, y_val), verbose=1)

print('Results > ')
md.plots_log_train(hist.history, '/home/marcos/Imagens/Train NN/'+str(vez)+'.1.png')
res = nn.evaluate(x_test, y_test)
md.print_score(nn.metrics_names, res)

y_pred = nn.predict(x_test)

fpr, tpr, thresholds = roc_curve(y_test.ravel(), y_pred.ravel())
auc = auc(fpr, tpr)


md.plot_ROC(fpr, tpr, 'NN (area = {:.3f})'.format(auc), '/home/marcos/Imagens/Train NN/'+str(vez)+'.2.png')
print('Done.')