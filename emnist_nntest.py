# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 19:42:45 2020
@author: barce_000
"""
# -*- coding: utf-8 -*-
#импорт модулей pytorch
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os.path


import time
from datetime import timedelta


# Объявим класс для нашей нейронной сети
class Net(nn.Module):
    def __init__(self,device):
       super(Net,self).__init__()       
       self.device=device
       #определяем слои нейросети    
       self.fc1 = nn.Linear(28 * 28, 200).to(device)
       self.fc2 = nn.Linear(200, 200).to(device)
       self.fc3 = nn.Linear(200, 62).to(device)
       
       
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
        train_loader = torch.utils.data.DataLoader(
        datasets.EMNIST('../dataEMNIST',split="byclass", train=True, download=True,
            transform=transforms.ToTensor()),                 
            batch_size=batch_size, shuffle=True,pin_memory=True)
        
        channels_sum,channels_squared_sum, num_batches=0,0,0
        
        for data,_ in train_loader:
            channels_sum+=torch.mean(data)
            channels_squared_sum+=torch.mean(data**2)
            num_batches+=1
            
        mean=channels_sum/num_batches
        std=(channels_squared_sum/num_batches-mean**2)**0.5
        print("Mean=%.4f, STD=%.4f" % (mean,std))
        
        
        test_loader = 0
        # torch.utils.data.DataLoader(
        # datasets.EMNIST('../data2', split="letters", train=False, 
        #     transform=transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.1307,), (0.3081,))
        #     ])),        
        #     batch_size=batch_size, shuffle=True,pin_memory=True)    
        return (train_loader,test_loader)
    
    
def train_net(net,train_loader,optimizer,criterion,device):
    # запускаем главный тренировочный цикл
    #пройдёмся по батчам из наших тестовых наборов
    #каждый проход меняется эпоха
   
    for epoch in range(epochs):
       for batch_idx, (data, target) in enumerate(train_loader):
           data, target = Variable(data).to(device), Variable(target).to(device)
    # изменим размер с (batch_size, 1, 28, 28) на (batch_size, 28*28)
           data = data.view(-1, 28*28)
    #оптимизатор
    #для начала обнулим градиенты перед работой
           optimizer.zero_grad()
    #считаем выход нейросети
           net_out = net(data)
    #оптимизировать функцию потерь будем на основе целевых данных и выхода нейросети
           loss = criterion(net_out, target)
    #делаем обратный ход
           loss.backward()
    #оптимизируем функцию потерь, запуская её по шагам полученным на backward этапе
           optimizer.step()
    #вывод информации
           if batch_idx % log_interval == 0:
               print('Train Epoch: {} [{}/{} ({:.0f}%)]tLoss: {:.6f}'.format(
                       epoch, batch_idx * len(data), len(train_loader.dataset),
                              100. * batch_idx / len(train_loader), loss.data))    
               
def test_nn(net,test_loader,device):                
    #тестирование
    test_loss = 0
    correct = 0
    for data, target in test_loader:
       data, target = Variable(data, requires_grad=False).to(device), Variable(target).to(device)
       data = data.view(-1, 28 * 28)
       net_out = net(data)
    # Суммируем потери со всех партий
       test_loss += criterion(net_out, target).data
       pred = net_out.data.max(1)[1]  # получаем индекс максимального значения
    #сравниваем с целевыми данными, если совпадает добавляем в correct
       correct += pred.eq(target.data).sum()
    
    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
           test_loss, correct, len(test_loader.dataset),
           100. * correct / len(test_loader.dataset)))
    
 
model_path="model2";
modelsave_exists=os.path.isfile(model_path)

#проверяем можно ли использовать gpu
if torch.cuda.is_available():  
  dev = "cuda" 
else:  
  dev = "cpu"
print("Used Device: {}".format(dev))
# dev = "cpu"
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
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

# функция потерь
criterion = nn.NLLLoss()

#задаём остальные параметры
batch_size=1000
learning_rate=0.01
epochs=20
log_interval=1
#(train_loader,test_loader) 
train = load_traindata(batch_size)


dataiter = iter(train[0])
images, labels = dataiter.next()

figure = plt.figure()
num_of_images = 100
for index in range(1, num_of_images + 1):
    plt.subplot(10, 10, index)
    plt.axis('off')
    plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')
 
    
load_traindata(1)   
# #train_loader
# #test_loader
# #train_net(net,train[0],optimizer,criterion,device)
# #test_nn(net,train[1],device)

# start_time = time.time()

# train_net(net,train[0],optimizer,criterion,device)
#test_nn(net,train[1],device)



# elapsed_time_secs = time.time() - start_time

# msg = "Execution took: %s secs (Wall clock time)" % timedelta(seconds=round(elapsed_time_secs))

# print(msg)  

# if (not modelsave_exists):
#     torch.save(net.state_dict(),model_path)
#     print("Model Saved!")
