# PyTorchCharsRecognition

Testing Neural Networks work with PyToch


## Requirements

- PyToch
- matploitlib
- opencv(for test)

## Installation torch and opencv

```
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```

```
conda install -c conda-forge opencv
```

## Run

```
python test_network
```

Structure :
- datasets: datasets and their loaders

- models : saved neural networks models

- networks(package): neural networks classes

- results : graphs and other testing results
  - *.png* - learning results graphs
  
  - *.txt* - learning results saved as file
- other:
  - test_network(starting file): for testing networks
  - graphs_shower.py: shows learning graphs
  - graphs_loader.py: loads data for graphs from file
  - data_loader.py: loads dataset data - OLD!!!
  - network_testing.py: testing algorithm
  - network_training: training algorithm
  - image_prepare(future file).py: for image preparation
  - inference_test (future file).py: for inference
  - create_dataset.py(future file):  for creating datasets
  
  


