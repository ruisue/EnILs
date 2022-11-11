# EnILs:A novel ensemble computational approach for prediction of interleukins inducing peptides
## surui 

## Table of contents
* [Install](#Install)
	* [Install using `pip`](#Install)
	* [package](#Package)
* [Usage](#Usage)
	* [Feature extraction](#Feature) 
	* [Classifier](#Classifier) 
## Install
### Install using `pip`

Firstly, we suggest to build a new virtual environment through `Anaconda`:
```
conda create -n    python=3.6
```
Create and activate the virtual environment environment `    `:
```
conda activate  
```
### Package
| package | version |
| :----: | :----: |
| keras  | 2.2.4 |
| tensorFlow | 1.14.0 |
| scikit-learn | 0.24.1 |
| genism | 3.8.3 |
## Usage
### Feature extraction：
  * word2vec.py is the implementation of word2vec feature.
### Classifier:
  * train.py is the implemention of our model
  * predict.py is used to predict new samples
