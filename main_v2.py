from src.pipeline.extract_features_v2 import extract_features_v2
from src.pipeline.mapping_features_v2 import json_to_csv_with_level2
from src.utils.file_handling import set_training_data
from src.datas.preprocess import DataPreprocess
from src.utils.file_handling import csv_to_json
from wordfreq import word_frequency
import spacy
import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filename", help="Input file name (without path) in data/raw/ folder (.csv)", required=True)

try:
    args = parser.parse_args()
except SystemExit:
    parser.print_help()
    exit(1)

# Join the folder path with the provided filename
filename = os.path.join("raw", os.path.splitext(str(args.filename))[0])
nlp = spacy.load("en_core_web_sm")
word_freq = word_frequency
data = csv_to_json(filename)

print('Start preprocessing...')
preprocess = DataPreprocess(data, nlp, filename)
preprocess.start()
print('Processed!')

path = os.path.join(os.getcwd(), 'data', f'{filename}_preprocessed.json')
print('Start feature extraction...')
with open(path, 'r') as f:
    data = json.load(f)
    featex = extract_features_v2(data, filename, nlp, word_freq)
print('Extracted!')

print('Mapping...')
json_to_csv_with_level2('cobi_features_v2')
print('Successfully Mapped Questions with Competencies!')

# Load the JSON data from the input file
with open('data/raw/cobi_features_v2.json', 'r') as f:
    json_data = json.load(f)

output = []

# Initialize a dictionary to store questions by level
questions_by_level = {
    "level_1": [],
    "level_2": [],
    "level_3": [],
    # Add other levels as needed
}

# Loop through the items in the loaded JSON data
for item in json_data:
    # Create a new dictionary for each item with `question_text` and `level2` values
    row_data = {'question_text': item['question_text']}
    row_data.update({'answer': item['answer']})
    row_data.update({'options_comp': item['options_comp']})
    # row_data.update(item['options_comp'])
    # row_data.update(item['level2'])
    output.append(row_data)
    if(item['levels']['Level_1'] == True):
        questions_by_level['level_1'].append(row_data)
    elif(item['levels']['Level_2'] == True):
        questions_by_level['level_2'].append(row_data)
    else:
        questions_by_level['level_3'].append(row_data)

# Write the output JSON data to a new JSON file
with open('data/questions_by_level.json', 'w') as jsonfile:
    json.dump(questions_by_level, jsonfile, indent=4)

# Write the output JSON data to a new JSON file
with open('data/output.json', 'w') as jsonfile:
    json.dump(output, jsonfile, indent=4)

print("Data successfully written to output.json")
