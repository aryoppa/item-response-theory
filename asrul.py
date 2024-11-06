from src.data.preprocess_asrul import DataPreProcessing
from src.utils.file_handling import csv_to_json_asrul
import spacy
import os
import json
import argparse

json_path = os.path.join(os.getcwd(), 'data', 'apriori', 'questions' + '.json')
# Read the JSON data
with open(json_path, 'r', encoding='utf-8') as json_file:
    questions = json.load(json_file)

json_path = os.path.join(os.getcwd(), 'data', 'apriori', 'result' + '.json')
# Read the JSON data
with open(json_path, 'r', encoding='utf-8') as json_file:
    result = json.load(json_file)

json_path = os.path.join(os.getcwd(), 'data', 'apriori', 'mapped_competencies' + '.json')
# Read the JSON data
with open(json_path, 'r', encoding='utf-8') as json_file:
    mapped_competencies = json.load(json_file)

datapreprocess = DataPreProcessing(result, questions, mapped_competencies)
datapreprocess.start()