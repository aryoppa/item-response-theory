import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os
import json

# Paths
DATA_DIR = os.path.join(os.getcwd(), 'data')
MAPPED_DIR = os.path.join(DATA_DIR, 'mapped')

# Define the 3PL model function
def calculate_probability(theta, a, b, c):
    exponent = -a * (theta - b)
    probability = c + (1 - c) / (1 + np.exp(exponent))
    return probability

# Define item information function
def item_information(theta, a, b, c):
    P_theta = calculate_probability(theta, a, b, c)
    return (a ** 2) * P_theta * (1 - P_theta)

# Update theta using gradient descent
def update_theta(current_theta, response, question_params, learning_rate=0.1):
    question = question_params.loc[question_params['question_id'] == response['question_id']].iloc[0]
    a, b, c = question['a'], question['b'], question['c']
    prob = calculate_probability(current_theta, a, b, c)
    gradient = (response['is_correct'] - prob) * a
    new_theta = current_theta + learning_rate * gradient
    return new_theta

# Select the next most informative question with competency filtering
def competency_similarity(target_vector, question_vector):
    target_vector = np.array(target_vector).reshape(1, -1)
    question_vector = np.array(question_vector).reshape(1, -1)
    return cosine_similarity(target_vector, question_vector)[0][0]

def select_next_question(theta, questions, target_competencies):
    questions = questions.copy()
    questions['similarity'] = questions.apply(
        lambda row: competency_similarity(target_competencies, row[competency_columns].values), axis=1
    )
    questions['information'] = questions.apply(
        lambda row: item_information(theta, row['a'], row['b'], row['c']) * row['similarity'], axis=1
    )
    selected_question = questions.loc[questions['information'].idxmax()]
    question_competencies = selected_question[competency_columns].to_dict()
    return selected_question, question_competencies

# Load dataset
csv_filepath = os.path.join(MAPPED_DIR, 'questions_database.csv')
questions_df = pd.read_csv(csv_filepath)

# Identify competency columns
competency_columns = [
    "main_verbs", "tense", "infinitives", "passives", "have_+_participle",
    "auxiliary_verbs", "pronouns", "nouns", "determiners",
    "other_adjectives", "prepositions", "conjunctions", "subject_verb_agreement"
]

# Test simulation
theta = 0.0
asked_questions = []
responses = []
selected_question_competencies = []
target_competencies = [1] * len(competency_columns)

for _ in range(10):
    available_questions = questions_df[~questions_df['question_id'].isin(asked_questions)]

    if available_questions.empty:
        print("No more questions available.")
        break

    # Select the next question and extract competencies
    next_question, question_competencies = select_next_question(theta, available_questions, target_competencies)

    print(f"Next question: {next_question['question_text']}")
    print(f"Competencies: {question_competencies}")

    prob_correct = calculate_probability(theta, next_question['a'], next_question['b'], next_question['c'])
    is_correct = np.random.choice([0, 1], p=[1 - prob_correct, prob_correct])
    print(f"Simulated response: {is_correct}")

    # Store responses and competencies
    responses.append({"question_id": next_question['question_id'], "is_correct": is_correct})
    selected_question_competencies.append({
        "question_id": next_question['question_id'],
        "competencies": question_competencies
    })
    asked_questions.append(next_question['question_id'])

    # Update theta
    theta = update_theta(theta, responses[-1], questions_df)
    print(f"Updated theta: {theta}\n")

# Extract competencies for incorrect answers
incorrect_competencies = [
    entry["competencies"]
    for entry, response in zip(selected_question_competencies, responses)
    if response["is_correct"] == 0
]

# Sum all True competencies from incorrect answers
competency_totals = {key: 0 for key in competency_columns}
for competencies in incorrect_competencies:
    for key, value in competencies.items():
        if value:  # If the competency is True (or 1)
            competency_totals[key] += 1

print("Total counts of True competencies from incorrect answers:")
for key, value in competency_totals.items():
    print(f"{key}: {value}")

# Save totals to a file
competency_totals_file = os.path.join(DATA_DIR, 'competency_totals_from_incorrect.json')
with open(competency_totals_file, 'w') as file:
    json.dump(competency_totals, file, indent=4)
print(f"Competency totals from incorrect answers saved to {competency_totals_file}.")

# Create matrix for incorrect answers
incorrect_competency_matrix = []
for competencies in incorrect_competencies:
    row = [int(competencies[comp]) for comp in competency_columns]  # Retain binary format for all competencies
    incorrect_competency_matrix.append(row)

# Save incorrect competency matrix
matrix_file = os.path.join(DATA_DIR, 'incorrect_competency_matrix.json')
with open(matrix_file, 'w') as file:
    json.dump({
        "matrix": incorrect_competency_matrix
    }, file, indent=4)
print(f"Incorrect competency matrix saved to {matrix_file}.")

print("Test completed.")
print(f"Final estimated ability (theta): {theta}")
# Load competency_totals_from_incorrect.json
totals_file = os.path.join(DATA_DIR, 'competency_totals_from_incorrect.json')
with open(totals_file, 'r') as file:
    competency_totals = json.load(file)

# Sort competencies by values and resolve ties for top 3
sorted_totals = sorted(competency_totals.items(), key=lambda x: x[1], reverse=True)
top_3_cutoff = sorted_totals[2][1]  # Value of the third-highest competency

# Get all competencies that are in the top 3 or tied
top_competencies = [comp for comp, value in sorted_totals if value >= top_3_cutoff]
print("Top competencies:", top_competencies)

# Create the final matrix based on the top competencies
final_matrix = []

for competencies in incorrect_competencies:
    row = [1 if comp in top_competencies and competencies[comp] else 0 for comp in competency_columns]
    final_matrix.append(row)

# Save the final matrix to a JSON file
final_matrix_file = os.path.join(DATA_DIR, 'final_competency_matrix.json')
with open(final_matrix_file, 'w') as file:
    json.dump({
        "top_competencies": top_competencies,
        "matrix": final_matrix
    }, file, indent=4)

print(f"Final matrix saved to {final_matrix_file}.")
