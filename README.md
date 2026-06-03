# project-datamining

## Setup

Create a Python environment and install the required libraries:

pip install -r requirements.txt


## Data

The project expects a `datasets/` directory in the main project folder. This folder should contain the dataset files provided in the project description, including the training, validation and test CSV files, as well as the embedding files.

Example structure:

```text
project-root/
├── datasets/
│   ├── search_stream_training.csv
│   ├── click_validation_query.csv
│   ├── click_validation_answer.csv
│   ├── click_test_query.csv
│   ├── ad_validation_query.csv
│   ├── ad_validation_answer.csv
│   ├── ad_test_query.csv
│   ├── searchinfo.csv
│   ├── userinfo.csv
│   ├── adinfo.csv
│   ├── searchinfo_text_embs.npy
│   └── adinfo_title_embs.npy
├── src/
└── requirements.txt
```

## Running scripts

Run scripts from the project root directory, for example:

```bash
python src/task1_xgboost.py
python src/task2_rank.py
python src/task2_ppr_rank.py
```

The scripts use the file paths defined directly in the code. If a different input file should be treated as validation or test data, update the corresponding filename manually inside the selected script.


## Outputs

Generated prediction files are saved in the specific directories, which names can be easily identified by the directories names in the repository.
