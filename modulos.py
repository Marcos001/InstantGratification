''' network '''
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, Adam

''' visualize '''
import matplotlib.pyplot as plt
plt.style.use('seaborn')


def NeuralNetwork(entrada, num_classes):

    model = Sequential()
    model.add(Dense(10, activation='relu', input_shape=(entrada,)))
    model.add(Dropout(0.2))
    model.add(Dense(12, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='sigmoid'))

    model.compile(optimizer=Adam(lr=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model




def proportion_split(total, verbose=False):
    p_train = int(total/2)
    p_test  = int(p_train/2)
    p_lim_test = int((p_train+p_test))
    if verbose:
        print('Train(:%i) Test(%i:%i) Validation(:%i)' %(p_train,p_train,p_lim_test,p_lim_test))
        print('-'*30)
        return p_train, p_lim_test
    else:
        return p_train, p_lim_test


def print_score(metricas, evaluation):
    print('_' * 20)
    print('Model Evaluate')
    print('-' * 20)
    for i in range(len(evaluation)):
        print(metricas[i] + ' = %.2f' % (evaluation[i]))
    print('-' * 20)


def plot_log_train(log):
    chaves = list(log.keys())
    print(chaves)
    plt.figure(figsize=(15, 6))
    for i in range(len(chaves)):
        plt.plot(log[chaves[i]], '-o', label=chaves[i])
    plt.legend()
    plt.show()


def plots_log_train(log, save=None):
    chaves = list(log.keys())
    fig = plt.figure(figsize=(18, 5))

    ax = fig.add_subplot(121)
    ax.plot(log[chaves[0]], '-o', label=chaves[0])
    ax.plot(log[chaves[2]], '-o', label=chaves[2])
    ax.set_title('Loss')
    ax.legend()

    ax = fig.add_subplot(122)
    ax.set_title('Accuracy')
    ax.plot(log[chaves[1]], '-o', label=chaves[1])
    ax.plot(log[chaves[3]], '-o', label=chaves[3])
    ax.legend()

    if save:
        plt.savefig(save)

    plt.show()


def plot_ROC(fpr, tpr, label, save=None):
    ''' plot Curve ROC '''
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label=label)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    if save:
        plt.savefig(save)
    plt.show()