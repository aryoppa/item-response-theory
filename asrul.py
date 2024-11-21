import pandas as pd
import logging
from mlxtend.frequent_patterns import association_rules
from src.service.data_preprocessing import DataPreProcessing
from src.service.apriori import Apriori

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='log/asrul_v3.log')

data_preprocessing = DataPreProcessing()
apriori = Apriori()

def load_data():
    """
    Load data from CSV files.
    """
    try:
        df_mapping_question_comp = pd.read_csv("data/apriori/mapping-assessment-question-competency.csv")
        df_questions = pd.read_csv("data/apriori/assessment-questions.csv")
        df_test_results = pd.read_csv("data/apriori/assessment-result.csv")
        return df_mapping_question_comp, df_questions, df_test_results
    except FileNotFoundError as e:
        logging.error("Error loading data: %s", e)
        return None, None, None

def preprocess_data():
    """
    Preprocess the data using various data transformation techniques.
    """
    df_mapping_question_comp, df_questions, df_test_results = load_data()
    if any(df is None for df in [df_mapping_question_comp, df_questions, df_test_results]):
        return None

    transformed_data = data_preprocessing.transform_result_to_biner(df_test_results, df_questions)
    student_comp = data_preprocessing.mapping_student_competency(transformed_data, df_mapping_question_comp)
    final_dataset = data_preprocessing.generate_final_dataset(student_comp)
    return final_dataset

def generate_association_rules(final_dataset):
    """
    Generate association rules using the preprocessed dataset.
    """
    if final_dataset is None:
        logging.error("Final dataset is not available for generating association rules.")
        return

    frequent_itemsets = apriori.apriori(final_dataset, min_support=0.97)
    results = pd.DataFrame(list(frequent_itemsets.items()), columns=['itemsets', 'support'])
    results['itemsets'] = results['itemsets'].apply(lambda x: tuple(x))
    results = results.sort_values(by='support', ascending=False).reset_index(drop=True)

    rules = association_rules(results, metric="confidence", min_threshold=0.97)
    rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(map(str, x)) if isinstance(x, (set, frozenset)) else x)
    rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(map(str, x)) if isinstance(x, (set, frozenset)) else x)
    rules['combined'] = rules['antecedents'] + ',' + rules['consequents']

    # Add indexing to the association rules
    rules.reset_index(inplace=True)
    rules.rename(columns={'index': 'rule_id'}, inplace=True)
    rules.set_index('rule_id', inplace=True)

    rules[['antecedents', 'consequents', 'combined']].to_csv("./data/apriori/association_rules.csv", index=True)
    logging.info("Association rules generated successfully with indexing.")

def main():
    """
    Main function to preprocess data and generate association rules.
    """
    final_dataset = preprocess_data()
    generate_association_rules(final_dataset)

if __name__ == "__main__":
    main()