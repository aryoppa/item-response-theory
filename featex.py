import os
import json
import argparse
import spacy
from wordfreq import word_frequency
from src.pipeline.extract_features_v2 import extract_features_v2
from src.pipeline.mapping_features_v2 import json_to_csv_with_level2
from src.utils.file_handling import csv_to_json
from src.datas.preprocess import DataPreprocess

# Configure argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filename", help="Input file name (without path) in data/raw/ folder (.csv)", required=True)

try:
    args = parser.parse_args()
except SystemExit:
    parser.print_help()
    exit(1)

# Configure NLP and paths
filename = os.path.join("raw", os.path.splitext(str(args.filename))[0])
nlp = spacy.load("en_core_web_sm")
word_freq = word_frequency

# Load data and preprocess
print('Start preprocessing...')
data = csv_to_json(filename)
preprocess = DataPreprocess(data, nlp, filename)
preprocess.start()
print('Processed!')

# Extract features
print('Start feature extraction...')
path = os.path.join(os.getcwd(), 'data', f'{filename}_preprocessed.json')
with open(path, 'r') as f:
    data = json.load(f)
    extract_features_v2(data, filename, nlp, word_freq)
print('Extracted!')

# Map features to CSV
print('Mapping...')
json_to_csv_with_level2('q_bank_features_v2')
print('Successfully Mapped Questions with Competencies!')

# Load and process JSON data
with open('data/raw/q_bank_features_v2.json', 'r') as f:
    json_data = json.load(f)

output = []
questions_by_level = {"level_1": [], "level_2": [], "level_3": []}

# Process each item and organize by levels
for item in json_data:
    row_data = {'question_text': item['question_text'], 'answer': item['answer'], 'options_comp': item['options_comp']}
    output.append(row_data)
    if item['levels']['Level_1']:
        questions_by_level['level_1'].append(row_data)
    elif item['levels']['Level_2']:
        questions_by_level['level_2'].append(row_data)
    else:
        questions_by_level['level_3'].append(row_data)

# Write processed data to JSON
with open('data/questions_by_level.json', 'w') as jsonfile:
    json.dump(questions_by_level, jsonfile, indent=4)
with open('data/output.json', 'w') as jsonfile:
    json.dump(output, jsonfile, indent=4)

print("Data successfully written to output.json")
