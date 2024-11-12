import os
import csv
import json

def json_to_csv_with_level2(filename):
    json_path = os.path.join(os.getcwd(), 'data', 'raw', filename + '.json')

    # Read the JSON data
    with open(json_path, 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)

    # Extract the CSV path
    csv_path = os.path.join(os.getcwd(), 'data', 'raw', 'mapped_questions_competencies.csv')

    # Prepare the data for CSV with 'question_text' and 'level2' features
    csv_data = []
    for item in json_data:
        # Create a dictionary with 'question_text' and unpack all level2 features
        # print(item.get("level2", {}))
        row = {"question_text": item["question_text"]}
        row.update(item.get("level2", {}))  # Add all level2 features to the row
        csv_data.append(row)

    # Write the CSV data
    if csv_data:
        # Get the CSV headers from the first row's keys
        keys = csv_data[0].keys()

        with open(csv_path, 'w', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=keys)
            csv_writer.writeheader()
            csv_writer.writerows(csv_data)
    
    return csv_path
