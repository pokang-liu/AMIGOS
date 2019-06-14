Affective computing using AMIGOS dataset
===

## Requirement

Python == 3.6.5  


## Usage

```bash
# Preprocessing(features stored in data/features.p)
python preprocess.py --input [AMIGOS_DATA_DIRECTORY (default is ./data)]

# Training and Cross Validation
python main.py --data [AMIGOS_DATA_DIRECTORY (default is ./data)]
               --feat [MODALITY_TYPE (default is all)]
               --clf [CLASSIFIER_TYPE (default is xgb)]
               --nor [NORMALIZATION_METHOD (default is no)]

# Print help message
python main.py -h
```
