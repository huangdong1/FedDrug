# Multi-party Collaborative Drug Discovery via Federated Learning

This repository is a implementation of "Multi-party Collaborative Drug Discovery via Federated Learning".

## 1. Datasets

- Davis and KIBA: https://github.com/hkmztrk/DeepDTA/tree/master/data
- DrugBank: https://github.com/kanz76/SSI-DDI/tree/master/data

## 2. Requirements

crypten == 0.4.1  
numpy == 1.24.2  
pandas == 1.5.3  
rdkit == 2022.9.1  
scikit-learn == 1.1.1  
scipy == 1.10.1  
torch == 1.12.0  
torch-geometric == 2.1.0  
tqdm == 4.65.0  

## 3. Usage

### How to run FL-DTA
```
python data_preprocessing_dta.py
python FL-DTA.py
```

### How to run FL-DDI
`python FL-DDI.py`
