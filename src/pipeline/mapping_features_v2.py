import os
import csv
import json
import pandas as pd

def json_to_csv_with_level2(filename):
    json_path = os.path.join(os.getcwd(), 'data', 'raw', filename + '.json')

    # Read the JSON data
    with open(json_path, 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)

    # Prepare the data for CSV with 'question_text' and 'level2' features
    csv_data = []
    for item in json_data:
        # Only include questions with at least one true value in level2 or level3 features
        if any(value is True for value in item.get("level2", {}).values()):
            row = {"question_text": item["question_text"]}
            row.update(item.get("level2", {}))  # Add all level2 features to the row
            csv_data.append(row)

    csv_data2 = []
    for item in json_data:
        # Only include questions with at least one true value in level2 or level3 features
        if any(value is True for value in item.get("level2", {}).values()):
            row = {"question_text": item["question_text"]}
            row.update({"option_A": item["option_A"][0]})
            row.update({"option_B": item["option_B"][0]})
            row.update({"option_C": item["option_C"][0]})
            row.update({"option_D": item["option_D"][0]})
            row.update({"key_answer": item["answer"]})
            if item['levels']['Level_1']:
                row.update({"Level": "Level 1"})
            elif item['levels']['Level_2']:
                row.update({"Level": "Level 2"})
            else:
                row.update({"Level": "Level 3"})
            csv_data2.append(row)

    # Extract the CSV path
    csv_path = os.path.join(os.getcwd(), 'data', 'mapped', 'mapped_questions_competencies.csv')
    csv_path2 = os.path.join(os.getcwd(), 'data', 'mapped', 'clean_questions.csv')
    # Write the CSV data
    if csv_data:
        # Get the CSV headers from the first row's keys
        keys = csv_data[0].keys()

        with open(csv_path, 'w', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=keys)
            csv_writer.writeheader()
            csv_writer.writerows(csv_data)

    # print(csv_data2)
    if csv_data2:
        # Get the CSV headers from the first row's keys
        keys = csv_data2[0].keys()

        with open(csv_path2, 'w', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=keys)
            csv_writer.writeheader()
            csv_writer.writerows(csv_data2)

    # Load the CSV data into a DataFrame to add indexing and drop duplicates
    df = pd.read_csv(csv_path)
    df.drop_duplicates(subset=['question_text'], inplace=True)  # Drop duplicate questions based on 'question_text'
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'question_id'}, inplace=True)
    # Save the indexed DataFrame back to the CSV
    df.to_csv(csv_path, index=False)

    df2 = pd.read_csv(csv_path2)
    # df2 = pd.DataFrame()
    df2.drop_duplicates(subset=['question_text'], inplace=True)  # Drop duplicate questions based on 'question_text'
    df2.reset_index(inplace=True)
    df2.rename(columns={'index': 'question_id'}, inplace=True)
    # Save the indexed DataFrame back to the CSV
    df2.to_csv(csv_path2, index=False)
    
    return csv_path
