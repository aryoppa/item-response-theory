import os
import csv
import json

# def csv_to_json(filename):
#     data_path = os.path.join(os.getcwd(), 'data', filename + '.csv')
#     csv_data = []

#     with open(data_path, 'r', encoding='utf-8', errors='replace') as csv_file:
#         csv_reader = csv.DictReader(csv_file)
#         for row in csv_reader:
#             csv_data.append(row)

#     json_path = os.path.join(os.getcwd(), 'data', filename + '.json')

#     # Write the data to a JSON file
#     with open(json_path, 'w') as json_file:
#         json.dump(csv_data, json_file, indent=4)

#     return csv_data

def csv_to_json(filename):
    # Paths for CSV and JSON files
    data_path = os.path.join(os.getcwd(), 'data', filename + '.csv')
    exist_path = os.path.join(os.getcwd(), 'data', 'raw', 'q_bank' + '.json')
    csv_data = []

    # Read the CSV file
    with open(data_path, 'r', encoding='utf-8', errors='replace') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            csv_data.append(row)

    # Load existing JSON data if file exists, otherwise start with an empty list
    # if os.path.exists(exist_path):
    with open(exist_path, 'r', encoding='utf-8') as json_file:
        existing_data = json.load(json_file)
    # else:
    #     existing_data = []

    # Combine existing JSON data with new CSV data
    combined_data = existing_data + csv_data
    json_path = os.path.join(os.getcwd(), 'data', filename + '.json')

    # Write the combined data back to the JSON file
    with open(json_path, 'w', encoding='utf-8') as json_file:
        json.dump(combined_data, json_file, indent=4)

    return combined_data


def csv_to_json_asrul(filename):
    data_path = os.path.join(os.getcwd(), 'data', 'apriori', filename + '.csv')
    csv_data = []

    with open(data_path, 'r', encoding='utf-8', errors='replace') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            csv_data.append(row)

    json_path = os.path.join(os.getcwd(), 'data', 'apriori', filename + '.json')

    # Write the data to a JSON file
    with open(json_path, 'w') as json_file:
        json.dump(csv_data, json_file, indent=4)

    return csv_data

def json_to_csv(filename):
    json_path = os.path.join(os.getcwd(), 'data', filename + '.json')

    # Read the JSON data
    with open(json_path, 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)

    # Extract the CSV path
    csv_path = os.path.join(os.getcwd(), 'data', filename + '.csv')

    # Get the keys (columns) from the first dictionary entry if JSON is a list of dictionaries
    if isinstance(json_data, list) and len(json_data) > 0:
        keys = json_data[0].keys()

        # Write the CSV data
        with open(csv_path, 'w', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=keys)
            csv_writer.writeheader()
            csv_writer.writerows(json_data)
    
    return csv_path


def set_training_data(filename, data):
    path = os.path.join(os.getcwd(), 'data', filename + '.json')

    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
    return path