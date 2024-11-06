"""
Import required dependencies
"""
from src.service.data_preprocessing import DataPreProcessing
# from src.repo.recommendation_repo import RecommendationRepo
# from src.repo.student_repo import StudentRepo
from src.service.apriori import Apriori
from src.service.fp_growth import FpGrowth

from mlxtend.frequent_patterns import fpgrowth, association_rules

import pandas as pd
import json

data_preprocessing = DataPreProcessing()
apriori = Apriori()

# read data from csv file
df_mapping_question_comp = pd.read_csv("data/data/mapping-assessment-question-competency.csv")
df_questions = pd.read_csv("data/data/assessment-questions.csv")
df_test_results = pd.read_csv("data/data/assessment-result.csv")

# Data preprocessing or data transformation
transormed_data = data_preprocessing.transform_result_to_biner(
    df_test_results, df_questions)
student_comp = data_preprocessing.mapping_student_competency(
    transormed_data, df_mapping_question_comp)
final_dataset = data_preprocessing.generate_final_dataset(student_comp)
transform_dataset = data_preprocessing.data_transformation(
    final_dataset)
# # Data modelling
# items = fpgrowth(transform_dataset, 0.9, use_colnames=True)
frequent_itemsets = apriori.apriori(final_dataset, 0.97)
# Convert the result to a DataFrame
results = pd.DataFrame(list(frequent_itemsets.items()), columns=['itemsets', 'support'])
results['itemsets'] = results['itemsets'].apply(lambda x: tuple(x))
results = results[['support', 'itemsets']].sort_values(by='support', ascending=False).reset_index(drop=True)

# # Building association rules
rules = association_rules(results, metric="confidence", min_threshold=0.97)
# rules = association_rules(items, metric="confidence", min_threshold=0.97)

# Convert rules DataFrame to CSV
rules.to_csv("association_rules.csv", index=False)

# Convert association rules to JSON format
# rules_json = data_preprocessing.association_rules_to_json(rules)

# # Write rules_json to a JSON file
# with open("rules.json", "w") as f:
#     json.dump(rules_json, f, indent=4)

# print(rules_json)

