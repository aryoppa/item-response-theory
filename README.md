# ESL Feature Extraction (EFE)

This project explores Natural Language Processing (NLP) techniques for feature extraction in ESL/TOEFL specifically Error Identification questions. By identifying these features, the project aims to improve the accuracy of competency classification.

## Getting Started

* **Prerequisites:**
  - Python 3.10.13
  - Spacy
  - en_core_web_sm (You can use the md or lg model if you like)
  - TOEFL ITP Error Identification type questions (csv format)
* **Installation:**  Use virtual env to install dependencies from the requirements.txt

## Running the Code

* **Basic Usage:**  Run the main.py followed by -f or --filename to specify the filename of the error identification question data.
* **Examples:**
`python main.py -f <filename>`
   - `<filename>`: should be the filename of the Error Identification question data on `data/<filename>.csv` path
   - `-f` `--filename`: Flag for input filename in `data/` folder with .csv format
* **Output**: `<filename>_features_v2.json` located in `data/` folder

## Notes
* This project is actively exploring Natural Language Processing (NLP) techniques for feature extraction in ESL/TOEFL questions, with a focus on Error Identification. By identifying these features, the project aims to improve the accuracy of competency classification in ESL/TOEFL assessments. Additionally, it serves as a learning platform for delving deeper into the capabilities of NLP.

