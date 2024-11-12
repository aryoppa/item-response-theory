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


# path = os.path.join(os.getcwd(), 'data', f'{filename}_features_v2.json')
# print('Finishing the data...')
# # Load the JSON file
# with open(path, 'r') as file:
#     data = json.load(file)

# # Features to remove from level3
# features_to_remove = ["c_count", "s_count", "u_word"]

# # Iterate over each question in the data
# for question in data:
#     level3 = question.get("level3", {})
#     for feature in features_to_remove:
#         if feature in level3:
#             del level3[feature]

# set_training_data(f'{filename}_features_v2_modified', data)
# print("Features removed successfully!")

# Load the JSON data from the input file
# with open('data/questions_features_v2.json', 'r') as f:
#     json_data = json.load(f)

# # Create a list to hold the transformed JSON data
# output_data = []

# # Loop through the items in the loaded JSON data
# for item in json_data:
#     # Create a new dictionary for each item with `question_text` and `level2` values
#     row_data = {'question_text': item['question_text']}
#     row_data.update(item['level2'])
    
#     # Append the transformed row to the output list
#     output_data.append(row_data)

# # Write the output JSON data to a new JSON file
# with open('data/output.json', 'w') as jsonfile:
#     json.dump(output_data, jsonfile, indent=4)

# print("Data successfully written to output.json")
