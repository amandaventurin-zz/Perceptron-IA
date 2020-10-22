#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from random import seed
from sklearn.datasets import fetch_openml # SKLEARN SERÁ USADO APENAS PARA COLETAR O DATASET
import seaborn as sns
import matplotlib.pyplot as plt

data = fetch_openml('mnist_784')
X = data.data / 255
y = data.target.astype(int)

np.set_printoptions(precision=None, suppress=True)

np.random.seed(42)


# Adicionando coluna de bias

# In[ ]:


X = np.append(np.ones((70000, 1)), X, axis=1)


# Vamos embaralhar X e y

# In[ ]:


idx_shuffled = np.arange(X.shape[0])
np.random.shuffle(idx_shuffled)

X = X[idx_shuffled]
y = y[idx_shuffled]


# Como no dataset original, escolheremos 60.000 dados para o conjunto de treinamento e 10.000 para o de testes:

# In[ ]:


X_train = X[:60000]
y_train = y[:60000]

X_test = X[60000:]
y_test = y[60000:]


# Definimos aqui a classe do Perceptron capaz de reconhecer um único número:

# In[ ]:


class Perceptron:
  def __init__(self, X_train, y_train, X_test, y_test, target_number, learning_rate=0.01):
    self.learning_rate = learning_rate
    self.target_number = target_number
    self.reset_weights()
    # pra treinamento
    self.X_train = X_train
    self.y_train = (y_train == self.target_number).astype(int)
    # para testar acurácia:
    self.X_test = X_test
    self.y_test = (y_test == self.target_number).astype(int)

  def reset_weights(self):
    self.weights = np.random.uniform(-0.5, 0.5, size=(785, 1))
  
  def update_weights(self):
    pred = self.predict(self.X_train)
    delta = self.y_train-pred
    self.weights = self.weights + self.learning_rate * self.X_train.T.dot(delta.reshape(-1,1))

  def train_one_epoch(self):
    self.update_weights()

  def train(self, epochs=50):
    self.reset_weights()
    for epoch in range(epochs):
      self.train_one_epoch()

  def predict_num(self, X):
    return X.dot(self.weights).flatten()

  def predict(self, X):
    return (self.predict_num(X) > 0).astype(int)

  def get_accuracy_test(self):
    total = self.y_test.shape[0]
    y_pred = self.predict(self.X_test)

    error = np.abs(self.y_test - y_pred).sum()
    correct = total - error

    return correct / total

  def get_accuracy_train(self):
    total = self.y_train.shape[0]
    y_pred = self.predict(self.X_train)

    error = np.abs(self.y_train - y_pred).sum()
    correct = total - error

    return correct / total


# Agora é hora de treiná-lo para que reconheça o número 1:

# In[ ]:


perceptron = Perceptron(X_train, y_train, X_test, y_test, target_number=1, learning_rate=0.01)


# Treinaremos e veremos a acurácia:

# In[12]:


perceptron.train()
print("Acurácia de Teste: {:.2f}%".format(perceptron.get_accuracy_test()*100))
print("Acurácia de treino: {:.2f}%".format(perceptron.get_accuracy_train()*100))


# Agora veremos um Perceptron que consegue reconhecer todos os dígitos

# In[ ]:


class AllDigitsPerceptron:
  def __init__(self, X_train, y_train, X_test, y_test, learning_rate=0.01):
    self.learning_rate = learning_rate
    self.X_train = X_train
    self.y_train = y_train
    self.X_test = X_test
    self.y_test = y_test

    self.perceptrons = []
    for i in range(10):
      perceptron = Perceptron(X_train, y_train, X_test, y_test, target_number=i, learning_rate=self.learning_rate,)
      self.perceptrons.append(perceptron)

  def train(self, epochs=50, generate_accuracy_history=False):
    accuracy_history_test = []
    accuracy_history_train = []
    print_time_with_message("Iniciando treinamento")
    for epoch in range(1, epochs+1):
      if epoch % 10 == 0:
        print("Época {}".format(epoch))
      if generate_accuracy_history: 
        accuracy_history_train.append(self.get_accuracy_train())
        accuracy_history_test.append(self.get_accuracy_test())        
      for perceptron in self.perceptrons:
        perceptron.train_one_epoch()
    print_time_with_message("Treinamento finalizado")
    last_accuracy_train = self.get_accuracy_train() # no for ele não calcula a acurácia de treino final
    last_accuracy_test = self.get_accuracy_test()   # no for ele não calcula a acurácia de teste final
    accuracy_history_train.append(last_accuracy_train)
    accuracy_history_test.append(last_accuracy_test)
    print("Acurácia treino: {:.2f}%".format(last_accuracy_train*100))
    print("Acurácia teste: {:.2f}%".format(last_accuracy_test*100))
    return accuracy_history_train, accuracy_history_test
  
  def predict(self, X):
    _predicts = []
    for perceptron in self.perceptrons:
      y_num_pred = perceptron.predict_num(X)
      _predicts.append(y_num_pred)
    predicts = np.array(_predicts).argmax(0)
    return predicts

  def get_accuracy_train(self):
    total = self.y_train.shape[0]
    y_pred = self.predict(self.X_train)

    error = (self.y_train != y_pred).sum()
    correct = total - error

    return correct / total

  def get_accuracy_test(self):
    total = self.y_test.shape[0]
    y_pred = self.predict(self.X_test)

    error = (self.y_test != y_pred).sum()
    correct = total - error

    return correct / total


# A função auxiliar abaixo serve apenas para printar uma mensagem com o horário atual ao lado.

# In[14]:


import pytz
from datetime import datetime

brazil_timezone = pytz.timezone('America/Sao_Paulo')

def print_time_with_message(message):
  _current_time = datetime.now(brazil_timezone)
  current_time = "[{:02d}/{:02d} {:02d}:{:02d}:{:02d}]".format(
      _current_time.day,
      _current_time.month,
      _current_time.hour,
      _current_time.minute,
      _current_time.second,
  )
  print("{}: {}".format(current_time, message))

print_time_with_message("TESTE")


# Bora testar o "Mega Perceptron"

# In[ ]:


all = AllDigitsPerceptron(X_train, y_train, X_test, y_test)


# In[16]:


accuracy_history_train, accuracy_history_test = all.train(epochs=50, generate_accuracy_history=True)


# In[ ]:


def plot_graph(accuracy_history_train, accuracy_history_test):  

  X1_plot = range(len(accuracy_history_train))
  y1_plot = accuracy_history_train

  X2_plot = range(len(accuracy_history_test))
  y2_plot = accuracy_history_test

  plt.figure(figsize=(11,5))
  plt.subplot(1, 2, 1)
  plt.xlabel("Acurácia de treino")
  plt.plot(X1_plot, y1_plot, 'b-')

  plt.subplot(1, 2, 2)
  plt.xlabel("Acurácia de teste")
  plt.plot(X2_plot, y2_plot, 'r-')

  plt.show()


# In[ ]:


def plot_confusion_matrix(y, y_pred, accuracy):
  _confusion_matrix = np.zeros((10,10))
  correct_or_incorrect = (y == y_pred)
  for i, booleano in enumerate(correct_or_incorrect):
    if booleano:
      _confusion_matrix[y[i]][y[i]] = _confusion_matrix[y[i]][y[i]] + 1
    else:
      _confusion_matrix[y[i]][y_pred[i]] = _confusion_matrix[y[i]][y_pred[i]] + 1
  plt.figure(figsize=(10,10))
  sns.heatmap(_confusion_matrix, annot=True, linewidths=.5, square = True, cmap = 'BrBG', fmt='g');

  plt.ylabel('Correto')
  plt.xlabel('Errado')
  plt.title('Acurácia: {:.2f}%'.format(accuracy*100))
  plt.show()


# In[ ]:


def mega_plot(learning_rate):
  global X_test
  global y_test
  all = AllDigitsPerceptron(X_train, y_train, X_test, y_test, learning_rate=learning_rate)
  accuracy_history_train, accuracy_history_test = all.train(epochs=50, generate_accuracy_history=True)
  plot_graph(accuracy_history_train, accuracy_history_test)

  y_pred = all.predict(X_test)
  plot_confusion_matrix(y_test, y_pred, accuracy_history_test[-1])


# # Primeira parte do relatório

# In[20]:


mega_plot(0.01)


# In[21]:


mega_plot(0.1)


# In[ ]:


mega_plot(1.0)


# # Cross Validation
# 

# In[ ]:


class CrossValidation:
  def k_folds_split(data, k_folds):
    data_splited_x = list()
    data_splited_y = list()
    fold_size = len(X)/k_folds
    k = 0
    for i in range (fold_size):
      folds_x = []
      folds_y = []
      while (len(folds) < fold_size):
        folds_x = X[k:k+fold_size]
        folds_y = y[k:k+fold_size]
        k=k+fold_size+1
      data_splited_x.append(folds_x)
      data_splited_y.append(folds_y)
    data_splited_x = np.array(data_splited_x)
    data_splited_y = np.array(data_splited_y)


  def predict(data_splited):
    for i in range(10):
      X_test = data_splited_x[i]
      y_test = data_splited_y[i]
      X_train = np.delete(data_splited_x, i, 0).reshape(-1)
      y_train = np.delete(data_splited_y, i, 0).reshape(-1)
      perceptron = Perceptron(X_train, y_train, X_test, y_test, target_number=1, learning_rate=0.01)
      perceptron.train()
      print("Acurácia de Teste: {:.2f}%".format(perceptron.get_accuracy_test()*100))
      print("Acurácia de treino: {:.2f}%".format(perceptron.get_accuracy_train()*100))




0


# In[ ]:



a = [[1,1], [1,2]]
b = a[0] + a[1]

