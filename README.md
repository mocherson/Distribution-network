# Distribution-network

Usage
--------
1. Use the run_mnist.py script for training, and validation and test distribution networks for MNIST dataset.
```
python run_mnist.py <options>
```
```
python run_mnist.py -h
```
```     
usage: run_mnist.py [-h] [--seed N] [-b N] [--path PATH] [--savepath
                    [--start-epoch N] [--gpu N] [--action ACTION]

Training the distribution network for MNIST

optional arguments:
  -h, --help            show this help message and exit
  --seed N              the random seed
  -b N, --batch-size N  mini-batch size (default: 128)
  --path PATH           data path
  --savepath SAVEPATH   path to save models
  -d N, --dim N         the dimention of latent space (default:10)
  -s SIGMASHAPE, --sigmashape SIGMASHAPE
                        the form of covariance matrix
  -e N, --epochs N      number of total epochs to run
  --start-epoch N       manual epoch number (useful on restarts)
  --gpu N               the GPU number (default auto schedule)
  --action ACTION       train or test (default train-test)
```


2. Use the run_cifar.py script for training, and validation and test distribution networks for CIFAR10 dataset.
```
python run_cifar.py <options>
```
```
python run_cifar.py -h
```
```     
usage: run_cifar.py [-h] [--seed N] [-b N] [--path PATH] [--savepath SAVEPATH] [-d N] [-s SIGMASHAPE] [-e N]
                    [--start-epoch N] [--gpu N] [--action ACTION]

Training the distribution network for CIFAR10

optional arguments:
  -h, --help            show this help message and exit
  --seed N              the random seed
  -b N, --batch-size N  mini-batch size (default: 128)
  --path PATH           data path
  --savepath SAVEPATH   path to save models
  -d N, --dim N         the dimention of latent space (default:10)
  -s SIGMASHAPE, --sigmashape SIGMASHAPE
                        the form of covariance matrix
  -e N, --epochs N      number of total epochs to run
  --start-epoch N       manual epoch number (useful on restarts)
  --gpu N               the GPU number (default auto schedule)
  --action ACTION       train or test (default train-test)
```

3. Use the run_ohsu.py script for training, and validation and test distribution networks for Ohsumed dataset.
```
python run_ohsu.py <options>
```
```
python run_ohsu.py -h
```
```     
usage: run_ohsu.py [-h] [--seed N] [-b N] [--path PATH] [--wordvecfile WORDVECFILE] [--savepath SAVEPATH] [-d N]
                   [-s SIGMASHAPE] [-e N] [--start-epoch N] [--gpu N] [--action ACTION]

Training the distribution network for Ohsumed

optional arguments:
  -h, --help            show this help message and exit
  --seed N              the random seed
  -b N, --batch-size N  mini-batch size (default: 128)
  --path PATH           data path
  --wordvecfile WORDVECFILE
                        wordvector files
  --savepath SAVEPATH   path to save models
  -d N, --dim N         the dimention of latent space (default:10)
  -s SIGMASHAPE, --sigmashape SIGMASHAPE
                        the form of covariance matrix
  -e N, --epochs N      number of total epochs to run
  --start-epoch N       manual epoch number (useful on restarts)
  --gpu N               the GPU number (default auto schedule)
  --action ACTION       train or test (default train-test)
```
