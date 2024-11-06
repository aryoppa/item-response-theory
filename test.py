from src.pipeline.extract_features_v2 import extract_features_v2
from src.data.preprocess import DataPreprocess
from src.utils.file_handling import csv_to_json
import spacy
import os
import json
import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument("-f", "--filename", help="Input file name in data/ folder (.csv)", required=True)
# try:
#     args = parser.parse_args()
# except SystemExit:
#     parser.print_help()
#     exit(1)

# filename = os.path.splitext(str(args.filename))[0]
# filename = 'question_bank'
# nlp = spacy.load("en_core_web_sm")
# data = csv_to_json(filename)

# # print('Start preprocessing...')
# # preprocess = DataPreprocess(data, nlp, filename)
# # preprocess.start()
# # print('Done')

# path = os.path.join(os.getcwd(), 'data', f'{filename}_preprocessed.json')
# print('Start feature extraction...')
# with open(path, 'r') as f:
#     data = json.load(f)
#     featex = extract_features_v2(data, filename, nlp)
# print('Done')

filename = 'q_bank'
path = os.path.join(os.getcwd(), 'data', f'{filename}_features_v2.json')
# Load the JSON file
with open(path, 'r') as file:
    data = json.load(file)

# Features to remove from level3
features_to_remove = ["c_count", "s_count", "u_word"]

# Iterate over each question in the data
for question in data:
    level3 = question.get("level3", {})
    for feature in features_to_remove:
        if feature in level3:
            del level3[feature]

# Save the modified JSON back to a file
with open('q_bank_features_v2_modified.json', 'w') as file:
    json.dump(data, file, indent=4)

print("Features removed successfully!")
