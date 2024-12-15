import os
import json
import pandas as pd
from pathlib import Path

def clean_question_data(filename):
    # Define paths using Pathlib
    base_path = Path(os.getcwd())
    json_path = base_path / 'data' / 'raw' / f'{filename}.json'
    csv_path = base_path / 'data' / 'mapped' / 'questions_database.csv'

    # Read the JSON data
    with open(json_path, 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)

    # Define mappings for parameter a and b
    competency_to_a = {
        "main_verbs": 0.5, "tense": 0.6, "infinitives": 0.8, "passives": 1.0,
        "have_+_participle": 1.2, "auxiliary_verbs": 1.4, "pronouns": 1.5,
        "nouns": 1.4, "determiners": 1.2, "other_adjectives": 1.0,
        "prepositions": 0.8, "conjunctions": 0.7, "subject_verb_agreement": 0.5
    }
    level_to_b = {"Level 1": -2, "Level 2": 0, "Level 3": 2}

    # Prepare the data for CSV with 'question_text', 'level2', and parameters
    csv_data = []
    for item in json_data:
        # Only include questions with at least one true value in level2 features
        if any(item.get("level2", {}).values()):
            # Calculate parameter a: average of true competencies in level2
            level2_features = item.get("level2", {})
            true_competencies = [competency_to_a[comp] for comp, value in level2_features.items() if value]
            a_value = sum(true_competencies) / len(true_competencies) if true_competencies else 0

            # Determine the Level and parameter b
            level = (
                "Level 1" if item['levels']['Level_1'] else
                "Level 2" if item['levels']['Level_2'] else
                "Level 3"
            )
            b_value = level_to_b[level]

            # Calculate parameter c: guessing likelihood
            num_options = len(item.get("option_A", []))  # Assuming all options have equal length
            c_value = 1 / num_options if num_options > 0 else 0

            # Create a row for the question
            row = {
                "question_text": item["question_text"],
                "option_A": item["option_A"][0],
                "option_B": item["option_B"][0],
                "option_C": item["option_C"][0],
                "option_D": item["option_D"][0],
                "key_answer": item["answer"],
                "Level": level,
                "a_value": a_value,
                "b_value": b_value,
                "c_value": c_value
            }
            # Add all level2 features to the row
            row.update(level2_features)
            csv_data.append(row)

    # If no valid questions, return early
    if not csv_data:
        return None

    # Convert the data to a DataFrame
    df = pd.DataFrame(csv_data)

    # Drop duplicates, reset index, and add 'question_id' column
    df = df.drop_duplicates(subset=['question_text']).reset_index(drop=True)
    df.insert(0, 'question_id', df.index)

    # Save the DataFrame to CSV
    df.to_csv(csv_path, index=False, encoding='utf-8')
    
    return str(csv_path)
