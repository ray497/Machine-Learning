from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

neuralnetwork=buildNetwork(2,3,1)
#for XOR
dataset=SupervisedDataSet(2,1)

dataset.addSample((0,0),(0,))

dataset.addSample((0,1),(1,))

dataset.addSample((1,0),(1,))

dataset.addSample((1,1),(0,))

trainer=BackpropTrainer(neuralnetwork,dataset)

for i in range(1,10000):
    trainer.train()
    if i % 1000 == 0:
        print(neuralnetwork.activate([0,0]))
        print(neuralnetwork.activate([0,1]))
        print(neuralnetwork.activate([1,0]))
        print(neuralnetwork.activate([1,1]))



        
    
