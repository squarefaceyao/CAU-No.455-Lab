# A Graph Neural Network Model for Predicting Biological Mechanisms of Plant Electrical Signal Classification
![python](https://img.shields.io/badge/Python%20tested-3.9.x%20%7C%203.8.x%20%7C%203.7.x%20-blue)
![requirements](https://img.shields.io/badge/requirements-up%20to%20date-brightgreen)
![last_update](https://img.shields.io/badge/last%20update-March%2020%2C%202022-yellowgreen)
![license](https://img.shields.io/github/license/squarefaceyao/CAU-No.455-Lab)


Implementation of A Graph Neural Network Model for Predicting Biological Mechanisms of Plant Electrical Signal Classification in PyTorch Geometric.



<div>
<img width=18% src="./images/Fig. 1.png"/>
</div>


# Requirements

  * Python 3.6.2
  * For the other packages, please refer to the requirements.txt.

# Usage

## step 1.  Data preprocessing  
1. Expanding electrical signal data using sliding windows.  
2. Loading the PesPPI dataset using the PyG framework.

```py
python process_datasets.py
```
## step 2.  pmesp training
Recording the mean of 10-fold cross-validation.

```py
python pmesp_train.py
```
## step 3.  Case studies
To ensure reproducibility of the results of the paper, please use the data in the case_study.py.

```py
python case_study.py
```

## Notes:
### 1. Traditional Link Prediction Methods

```py
python tradition_ppi.py
```

### 2. The details of PesPPI datasets used in PMESP

| Datasets | PP   | SP   | Association | LD     | AD   |
| :------- | :--- | :--- | :---------- | :----- | :--- |
| PesPPI   | 2779 | 821  | 23214       | 0.0018 | 6.45 |

```py
python summarydatasets.py
```

### 3. The code for the sliding window method is in utils/slide_window.py

### 4. PesPPI Datasets

1. The raw data used to train the model is uploaded to the [PlantES](http://www.plantes.cn/) platform. The raw data includes electrical signal data, protein interaction and protein semantic information, and the raw data can be downloaded via [API](http://39.100.142.42:8080/dataset/29/zip?name=ara-protein).  

2. The class [ARAPPI](utils/Arappi.py) we wrote can automatically download the raw data and preprocess it. The use of classes in [process_datasets.py](process_datasets.py) files




# Acknowledgments
Code is inspired by [PyG](https://github.com/pyg-team/pytorch_geometric)

