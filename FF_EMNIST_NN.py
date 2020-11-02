# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 17:49:57 2020

@author: barce
"""


# -*- coding: utf-8 -*-
#стандартные модули
import os.path
import time
from datetime import timedelta

#импорт модулей pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# from torch.autograd import Variable
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker



DATAPATH='../dataEMNIST'

# Объявим класс для нашей нейронной сети
class Net(nn.Module):
    def __init__(self,device):
       super(Net,self).__init__()       
       self.device=device
       #определяем слои нейросети    
       self.fc1 = nn.Linear(28 * 28, 784).to(device)
       self.fc2 = nn.Linear(784, 784).to(device)
       self.fc3 = nn.Linear(784, 62).to(device)
       
       
       # self.fc1 = nn.Linear(28 * 28, 200).to("cuda:0")
       # self.fc2 = nn.Linear(200, 200).to("cuda:0")
       # self.fc3 = nn.Linear(200, 10).to("cuda:0")
       # #перебрасываем слои на корректное устройство
       # self.fc1=self.fc1.to(device)
       # self.fc2=self.fc2.to(device)
       # self.fc2=self.fc3.to(device)
       
       
       #в качестве функций активациию Relu, в конце softmax
    def forward(self, x):
       x = F.relu(self.fc1(x)).to(self.device)
       x = F.relu(self.fc2(x)).to(self.device)
       x = self.fc3(x).to(self.device)
       return F.log_softmax(x).to(self.device)
      #вывод архитектуры нейросети
    def printNet(self):
        print(self)
   



def  load_traindata(batch_size):    
    transformations = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.1722,), (0.3310,)) ])    
    
    train_loader = torch.utils.data.DataLoader(
                   datasets.EMNIST(DATAPATH,split='byclass', train=True, download=True,transform=transformations),
                   batch_size=batch_size, shuffle=True,pin_memory=True)         
    
    test_loader = torch.utils.data.DataLoader(
                    datasets.EMNIST(DATAPATH,split='byclass', train=False, download=False,transform=transformations),
                    batch_size=batch_size, shuffle=True,pin_memory=True)  
    
    return (train_loader,test_loader)
    
    
def train_net(net,train_loader,optimizer,criterion,device, losses_info,acc_info):
    # запускаем главный тренировочный цикл
    #пройдёмся по батчам из наших тестовых наборов
    #каждый проход меняется эпоха
    loss=0
    correct=0
    total=0
    maxbatch_count=len(train_loader.dataset)//batch_size    
    
    for epoch in range(epochs):
       final_loss=0
       for batch_id, (data, labels) in enumerate(train_loader):
           data, labels = data.to(device), labels.to(device)
    # # изменим размер с (batch_size, 1, 28, 28) на (batch_size, 28*28)
           data = data.view(-1, 28*28)
    #оптимизатор
    #для начала обнулим градиенты перед работой
           optimizer.zero_grad()
    #считаем выход нейросети
           net_out = net(data)
    #оптимизировать функцию потерь будем на основе целевых данных и выхода нейросети
           loss = criterion(net_out, labels)
    #делаем обратный ход
           loss.backward()
    #оптимизируем функцию потерь, запуская её по шагам полученным на backward этапе
           optimizer.step()  
    #вывод информации           
           print('Train Epoch: {} [{}/{} ({:.0f}%)]; Loss: {:.6f}'.format(
                       epoch+1, (batch_id+1) * len(data), len(train_loader.dataset),
                              100. * (batch_id+1) / len(train_loader), loss.data)) 
           #len(train_loader.dataset)  
           if (batch_id+1) * len(data)==maxbatch_count* len(data):  
               final_loss=loss.data.item()
                          # Отслеживание точности
               total = labels.size(0)
               _, predicted = torch.max(net_out.data, 1)
               correct = (predicted == labels).sum().item()
       #вносим текущее значение функции потерь
       losses_info.append(final_loss)
       #вносим текущее значение точности распознавания
       acc_info.append(correct / total)
       
   
               
def test_nn(net,test_loader,device):                
    #тестирование
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
           data, target = data.to(device), target.to(device)
           #data = data.view(-1, 28 * 28)
           net_out = net(data)
        # Суммируем потери со всех партий
           test_loss += criterion(net_out, target).data
           pred = net_out.data.max(1)[1]  # получаем индекс максимального значения
        #сравниваем с целевыми данными, если совпадает добавляем в correct
           correct += pred.eq(target.data).sum()
    
    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
           test_loss, correct, len(test_loader.dataset),
           100. * correct / len(test_loader.dataset)))
 
def graphics_show_loss_acc(losses, accuraces):
    epochs_count=len(losses)
    epochs=[x+1 for x in range(epochs_count) ]
    for i in range(0,epochs_count):
        print("Epoch {}:Loss - {:.3f}".format( epochs[i],losses[i]))
        print("Epoch {}:Accuracy - {:.3f}".format( epochs[i],accuraces[i]))
        
    fig, axes = plt.subplots(2,1)
    #график значения функции потерь на каждой эпохе
    axes[0].plot(epochs, losses) 
    axes[0].grid(which='major',
        color = 'k')

    axes[0].grid(which='minor',
        color = 'gray',
        linestyle = ':')
    #axes[0].set_ylim(0, 1.2)
    axes[0].set_xlim(0, 20)
    
    axes[0].set_xlabel("Epochs") # ось абсцисс
    axes[0].set_ylabel("Losses") # ось ординат
    axes[0].set_title("Train Loss")    
    
    axes[0].xaxis.set_major_locator(ticker.MultipleLocator(2))
    axes[0].xaxis.set_minor_locator(ticker.MultipleLocator(1))
    axes[0].yaxis.set_major_locator(ticker.MultipleLocator(1))
    axes[0].yaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    axes[0].minorticks_on()
   
    #график точности на каждой эпохе
    axes[1].plot(epochs, accuraces) 
    axes[1].grid(which='major',
        color = 'k')

    axes[1].grid(which='minor',
        color = 'gray',
        linestyle = ':')
    axes[1].set_ylim(0, 1.2)
    axes[1].set_xlim(0, 20)
    
    axes[1].set_xlabel("Epochs") # ось абсцисс
    axes[1].set_ylabel("Accuraces") # ось ординат
    axes[1].set_title("Train Accuraces")
   
    
    axes[1].xaxis.set_major_locator(ticker.MultipleLocator(2))
    axes[1].xaxis.set_minor_locator(ticker.MultipleLocator(1))
    axes[1].yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    axes[1].yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    axes[1].minorticks_on()

    plt.subplots_adjust(hspace=0.5)
    plt.show()
    
    
#    
#    axs[1].plot(epochs)
#    axs[1].set_title("Train Epoch")

#функции потерь на каждой эпохе
epoch_losses=list()
#список значений точности на каждой эпохе
acc_list=list()

model_path="FFNN_EMNIST_model";
modelsave_exists=os.path.isfile(model_path)

#веса


#Назначаем устройство на котором будет работать нейросеть, по возможности CUDA
dev = "cuda" if torch.cuda.is_available() else "cpu"  
device = torch.device(dev) 
 
#создаём модель
#передаём вычисления на нужное устройство gpu/cpu
net= Net(device).to(device)
net.printNet()



#проверяем есть ли сохранённая модель
if (modelsave_exists):
    net.load_state_dict(torch.load(model_path))
    net.eval()


#скорость обучения
learning_rate=0.01
# Создаём оптимизатор- метод градиентного спуска, параметры learning_rate,momentum
#optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
#optimizer = optim.Adagrad(net.parameters(), lr=learning_rate)
#optimizer = optim.Adam(net.parameters(), lr=learning_rate,betas=(0.2,0.01))
optimizer = optim.Adam(net.parameters(), lr=learning_rate)


# функция потерь - логарифмическая функция потерь (log cross entropy loss)
criterion = nn.NLLLoss()

#задаём остальные параметры
batch_size=10000
learning_rate=0.01
epochs=20
#(train_loader,test_loader) 
train_data,test_data = load_traindata(batch_size)
# a=5
#train_loader
#test_loader
#train_net(net,train[0],optimizer,criterion,device)
#test_nn(net,train[1],device)
start_time = time.time()

# обучение сети
train_net(net,train_data,optimizer,criterion,device, epoch_losses,acc_list)

graphics_show_loss_acc(epoch_losses, acc_list)

test_nn(net,test_data,device)


elapsed_time_secs = time.time() - start_time
msg = "Execution took: %s secs (Wall clock time)" % timedelta(seconds=round(elapsed_time_secs))
print(msg)  

if (not modelsave_exists):
    torch.save(net.state_dict(),model_path)
    print("Model Saved!")
