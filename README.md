Affective Computing with AMIGOS Dataset
===

## Requirement

Python == 3.6.3  
biosppy == 0.5.0  
numpy == 1.13.3  
scipy == 0.19.1 
scikit-learn == 0.19.0

## Usage

```bash
# Feature extraction (features stored in data/features.p)
python ALL_preprocess.py

# Training and Cross Validation
python main.py --feature [MODALITY_TYPE]

# Print help message
python main.py -h
```
