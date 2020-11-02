# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 17:17:25 2020

@author: IVAN
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def graphics_show_loss_acc(losses, accuraces,save_file):
    epochs_count=len(losses)
    epochs=[x+1 for x in range(epochs_count) ]
    for i in range(0,epochs_count):
        print('''Epoch {}:Average Loss ={:.3f}
              Average Accuracy = {:.3f}'''
              .format( epochs[i],losses[i],accuraces[i]))        
        
    fig, axes = plt.subplots(2,1)
    #график значения функции потерь на каждой эпохе
    axes[0].plot(epochs, losses) 
    axes[0].grid(which='major',
        color = 'k')

    axes[0].grid(which='minor',
        color = 'gray',
        linestyle = ':')
    axes[0].set_ylim(0, 10)
    axes[0].set_xlim(0, 16)
    
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
    axes[1].set_ylim(0.5, 1.0)
    axes[1].set_xlim(0, 16)
    
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
    fig.savefig(save_file)