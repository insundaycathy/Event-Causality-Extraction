# Event-Causality-Extraction
## Description:
This repository contains the implementation for the paper: Event Causality Is Key to Computational Story Understanding (https://arxiv.org/abs/2311.09648)

The paper proposes a prompt for extracting event causality from free form story text with LLMs, and show that the extracted event causality improves performence on downstram tasks.
## Requirement:
Python version is 3.9

requirements:

```
numpy                     1.23.5
openai                    0.27.0
openssl                   1.1.1w
pandas                    2.0.2
torch                     2.0.1
tokenizers                0.13.3
transformers              4.28.0
sacrebleu                 2.3.1                    
scikit-learn              1.2.2
rouge                     1.0.1
```

You can install all requirements with the command:
```
pip install -r requirements.txt
```
## Datasets
### Event Causality Extraction
#### COPES
The COPES dataset can be downloaded from: https://github.com/HKUST-KnowComp/COLA/tree/master
#### GLUCOSE
The GLUCOSE dataset can be downloaded from: https://github.com/ElementalCognition/glucose

### Downstream Tasks:
To be filled


## Run event causality extraction on COPES:
The code is in COPES/event_causality_extraction_copes.py
```
event_causality_extraction_copes.py --save_dict {output file} --model {model_name} --input_data {path to COPES.json} --data_split {path to data split} --prompt {path to prompt}
```

## Run event causality extraction on GLUCOSE
The code is in GLUCOSE/event_causality_extraction_glucose.py
```
event_causality_extraction_glucose.py --save_dict {output file} --input_data {path to input data file} --prompt {path to prompt}
```

## Prompt
The prompt for event causality extraction is stored in prompt.txt
