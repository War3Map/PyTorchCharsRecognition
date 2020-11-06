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

import graphs_shower 

DATAPATH='../dataEMNIST'

# Объявим класс для нашей нейронной сети
class Net(nn.Module):
    def __init__(self,device):
       super(Net,self).__init__()       
       self.device=device
       #определяем слои нейросети
       self.Conv1 = nn.Sequential( 
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)) 
       self.Conv2 = nn.Sequential( 
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
       self.fc1 = nn.Linear(7*7*64, 200).to(device)
       self.fc2 = nn.Linear(200, 62).to(device)
       
       
       # self.fc1 = nn.Linear(28 * 28, 200).to("cuda:0")
       # self.fc2 = nn.Linear(200, 200).to("cuda:0")
       # self.fc3 = nn.Linear(200, 10).to("cuda:0")
       # #перебрасываем слои на корректное устройство
       # self.fc1=self.fc1.to(device)
       # self.fc2=self.fc2.to(device)
       # self.fc2=self.fc3.to(device)
       
       
       #в качестве функций активациию Relu, в конце softmax
    def forward(self, x):
       x = self.Conv1(x).to(self.device)
       x = self.Conv2(x).to(self.device)
       x = x.reshape(x.size(0), -1)
       x = self.fc1(x).to(self.device)
       x = self.fc2(x).to(self.device)
       return F.log_softmax(x).to(self.device)
      #вывод архитектуры нейросети
    def printNet(self):
        print(self)
   



def  load_traindata(batch_size):    
    transformations = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.1736,), (0.3317,)) ])    
    
    train_loader = torch.utils.data.DataLoader(
                   datasets.EMNIST(DATAPATH,split="byclass" , train=True, download=True,transform=transformations),
                   batch_size=batch_size, shuffle=True,pin_memory=True)         
    
    labels_loader = torch.utils.data.DataLoader(
                    datasets.EMNIST(DATAPATH,split="byclass", train=False, download=False,transform=transformations),
                    batch_size=batch_size, shuffle=True,pin_memory=True)  
    
    return (train_loader,labels_loader)
    
    
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
       correct=0
       batch_count=0;
       for batch_id, (data, labels) in enumerate(train_loader):
           data, labels = data.to(device), labels.to(device)
    # # изменим размер с (batch_size, 1, 28, 28) на (batch_size, 28*28)
           #data = data.view(-1, 28*28)
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
           batch_count+=1;
           final_loss+=loss.data.item()
                        # Отслеживание точности
           if (batch_id+1) * len(data)==maxbatch_count* len(data):  
               total = labels.size(0)
               _, predicted = torch.max(net_out.data, 1)
               correct += (predicted == labels).sum().item()
        #вносим текущее значение функции потерь
       losses_info.append(final_loss/batch_count)
       print("Average Loss={}".format(final_loss/batch_count))
       #вносим текущее значение точности распознавания
       acc_info.append(correct / total)
       
   
               
def test_nn(net,test_loader,device):                
    #тестирование
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, labels in test_loader:
           data, labels = data.to(device), labels.to(device)
           # data = data.view(-1, 28 * 28)
           
           net_out = net(data)
        # Суммируем потери со всех партий
           test_loss += criterion(net_out, labels).data
           pred = net_out.data.max(1)[1]  # получаем индекс максимального значения
        #сравниваем с целевыми данными, если совпадает добавляем в correct
           correct += pred.eq(labels.data).sum()
    
    test_loss /= len(test_loader.dataset)
    test_acc = float( 100. * correct / len(test_loader.dataset))
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
           test_loss, correct, len(test_loader.dataset),
           test_acc))
    return test_acc
 


#Сохраняет статистику в файл
def save_stats(accuraces,losses,common_time,save_file):    
    with open(save_file, "w+") as file:
        file.write("Train:{}\nTest:{}\n".format(common_time[0],common_time[1]))        
        epochs_count=len(accuraces)
        for i in range(0,epochs_count):
            file.write("{}:{}\n".format(accuraces[i],losses[i]))
    
    

#функции потерь на каждой эпохе
epoch_losses=list()
#список значений точности на каждой эпохе
acc_list=list()

model_path="CNN_EMNIST_model";
modelsave_exists=os.path.isfile(model_path)

DATASET="EMNIST"
OPTIMIZER="Adam"
NETWORK_TYPE="CNN"

#Назначаем устройство на котором будет работать нейросеть, по возможности CUDA
dev = "cuda" if torch.cuda.is_available() else "cpu"  
device = torch.device(dev) 
print("Running on Device:{}".format(device))
#создаём модель
#передаём вычисления на нужное устройство gpu/cpu
net= Net(device).to(device)
net.printNet()



#проверяем есть ли сохранённая модель
if (modelsave_exists):
    net.load_state_dict(torch.load(model_path))
    net.eval()


#скорость обучения
learning_rate=0.001
# Создаём оптимизатор- метод градиентного спуска, параметры learning_rate,momentum
#optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
#optimizer = optim.Adagrad(net.parameters(), lr=learning_rate)
#optimizer = optim.Adam(net.parameters(), lr=learning_rate,betas=(0.2,0.01))
optimizer = optim.Adam(net.parameters(), lr=learning_rate)


# функция потерь - логарифмическая функция потерь (log cross entropy loss)
criterion = nn.NLLLoss()

#задаём остальные параметры
batch_size=400
learning_rate=0.001
epochs=15

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
train_time = time.time() - start_time
train_time_str = str(timedelta(seconds=round(train_time)))
print("Train time: %s secs (Wall clock time)" % timedelta(seconds=round(train_time)))  
#тест результатов обучения сети
start_time = time.time()
avg_test_acc=test_nn(net,test_data,device)
print(avg_test_acc)
test_time = time.time() - start_time
test_time_str = str(timedelta(seconds=round(test_time)))
print("Test time: %s secs (Wall clock time)" % timedelta(seconds=round(test_time))) 
#время тестирования
time=(train_time_str,test_time_str) 
result_file = "{}_{}(op={},ep={},acc={:.3f})".format(NETWORK_TYPE,
                                               DATASET,
                                               OPTIMIZER,
                                               epochs,
                                               avg_test_acc)
#строим график обучения
graphs_shower.graphics_show_loss_acc(epoch_losses, acc_list, result_file+".png")
#Сохраняем результаты в файл
save_stats(acc_list,epoch_losses,time,result_file+".txt")



if (not modelsave_exists):
    torch.save(net.state_dict(),model_path)
    print("Model Saved!")
