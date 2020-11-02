# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 17:17:25 2020

@author: IVAN
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def graphics_show_loss_acc(saved_file,epmin=1,epmax=20,
                           lossmin=0,lossmax=20,accmin=0.0,accmax=1.0):
    #читаем из файла
    losses=list()
    accuraces=list()
    lossacc_list=list()
    with open(saved_file) as file:
        lossacc_list=file.read()
        
 
    lossacc_list=lossacc_list.split('\n')
    lossacc_list=[value for value in lossacc_list if value]
    
    print(lossacc_list)
    rows_count=len(lossacc_list)   
    for i in range(2,rows_count):
        row=lossacc_list[i].split(':')       
        accuraces.append(float(row[0]))
        losses.append(float(row[1]))
    
    print(losses)
    print(accuraces)
    epochs_count=len(losses)
    epochs=[x+1 for x in range(epochs_count) ]
    # for i in range(0,epochs_count):
    #     print('''Epoch {}:Average Loss ={:.3f}
    #           Average Accuracy = {:.3f}'''
    #           .format( epochs[i],losses[i],accuraces[i]))        
        
    fig, axes = plt.subplots(2,1)
    #график значения функции потерь на каждой эпохе
    axes[0].plot(epochs, losses) 
    axes[0].grid(which='major',
        color = 'k')

    axes[0].grid(which='minor',
        color = 'gray',
        linestyle = ':')
    axes[0].set_ylim(lossmin, lossmax)
    axes[0].set_xlim(epmin, epmax)
    
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
    axes[1].set_ylim(accmin, accmax)
    axes[1].set_xlim(epmin, epmax)
    
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
    fig.savefig(saved_file+".png")
    

print("Input Report Filename:")
saved_file=str(input())
graphics_show_loss_acc(saved_file,epmax=16,lossmax=2,accmin=0.7)