Affective Computing with AMIGOS Dataset
===

## Requirement

Python == 3.6.3  
biosppy == 0.5.0  
EMD-signal == 0.2.3  
numpy == 1.13.3  
scipy == 0.19.1  
scikit-learn == 0.19.0  
xgboost == 0.6  

## Usage

```bash
# Feature extraction (features stored in data/features.p)
python preprocess.py --data [AMIGOS_DATA_DIRECTORY (default is ./data)]

# Training and Cross Validation
python main.py --data [AMIGOS_DATA_DIRECTORY (default is ./data)]
               --feat [MODALITY_TYPE (default is all)]
               --clf [CLASSIFIER_TYPE (default is xgb)]
               --nor [NORMALIZATION_METHOD (default is no)]

# Print help message
python main.py -h
```
