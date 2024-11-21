import json
import pandas as pd
import numpy as np
import gen_rec  # Import the generate_recommendations module
from sklearn.metrics.pairwise import cosine_similarity

def load_json_file(filepath):
    """
    Load a JSON file and return its contents.
    """
    try:
        with open(filepath, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: File {filepath} not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: File {filepath} is not in the correct JSON format.")
        return None

def load_association_rules(csv_filepath):
    """
    Load association rules from CSV and return as a DataFrame.
    """
    try:
        df = pd.read_csv(csv_filepath)
        return df
    except FileNotFoundError:
        print(f"Error: File {csv_filepath} not found.")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: File {csv_filepath} is empty.")
        return None

def convert_to_matrix(competencies, columns):
    """
    Convert competencies dictionary to a binary matrix representation.
    
    Parameters:
    competencies (dict): A dictionary containing competencies and their true/false values.
    columns (list): List of column names representing competencies.
    
    Returns:
    np.array: A binary matrix representing competencies.
    """
    return np.array([[1 if competencies.get(col, False) else 0 for col in columns]])

def map_antecedents_to_competencies(antecedents, columns):
    """
    Convert antecedents list to a binary matrix representation similar to competencies.
    
    Parameters:
    antecedents (list): A list of antecedents from association rules.
    columns (list): List of column names representing competencies.
    
    Returns:
    np.array: A binary matrix representing antecedents.
    """
    return np.array([1 if col in antecedents else 0 for col in columns])

def check_triggered_rules(competencies, rules_df):
    """
    Check which association rules are triggered based on competencies.
    
    Parameters:
    competencies (dict): A dictionary containing competencies and their true/false values.
    rules_df (DataFrame): A DataFrame containing the association rules.
    
    Returns:
    DataFrame of triggered rules with all columns.
    """
    columns = ["main_verbs", "tense", "infinitives", "passives", "have_+_participle", 
               "auxiliary_verbs", "pronouns", "nouns", "determiners", 
               "other_adjectives", "prepositions", "conjunctions", "subject_verb_agreement"]
    competencies_matrix = convert_to_matrix(competencies, columns)
    
    similarity_scores = []
    for _, row in rules_df.iterrows():
        antecedents = row['antecedents'].split(', ')
        antecedents_matrix = map_antecedents_to_competencies(antecedents, columns)
        
        # Calculate the similarity score between the competencies matrix and the antecedents matrix
        similarity = cosine_similarity(competencies_matrix, [antecedents_matrix])[0][0]
        similarity_scores.append((similarity, row))
    
    # Sort by similarity score in descending order and get the top 3
    top_rules = sorted(similarity_scores, key=lambda x: x[0], reverse=True)[:3]
    triggered_rules = [rule[1] for rule in top_rules]
    
    return pd.DataFrame(triggered_rules)  # Return the top 3 triggered rules with all columns

def main():
    # Load the competencies JSON file
    json_filepath = 'data/competencies.json'  # Update the path as needed
    competencies = load_json_file(json_filepath)
    if competencies is None:
        return

    # Load the association rules CSV file
    csv_filepath = 'data/apriori/association_rules.csv'  # Update the path as needed
    rules_df = load_association_rules(csv_filepath)
    if rules_df is None:
        return

    # Check which rules are triggered based on the antecedents column
    triggered_rules_df = check_triggered_rules(competencies, rules_df)

    # Print the results
    if not triggered_rules_df.empty:
        print("Triggered Rules:")
        print(triggered_rules_df)
    else:
        print("No rules were triggered.")

if __name__ == '__main__':
    main()
