import json

# Load the JSON data from the input file
with open('data/questions_features_v2.json', 'r') as f:
    json_data = json.load(f)

# Create a list to hold the transformed JSON data
output_data = []

# Loop through the items in the loaded JSON data
for item in json_data:
    # Create a new dictionary for each item with `question_text` and `level2` values
    row_data = {'question_text': item['question_text']}
    row_data.update(item['level2'])
    
    # Append the transformed row to the output list
    output_data.append(row_data)

# Write the output JSON data to a new JSON file
with open('data/output.json', 'w') as jsonfile:
    json.dump(output_data, jsonfile, indent=4)

print("Data successfully written to output.json")
