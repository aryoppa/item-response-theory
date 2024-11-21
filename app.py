from flask import Flask, request, jsonify
import os
import json
import spacy
import pandas as pd
from wordfreq import word_frequency
from src.pipeline.extract_features_v2 import extract_features_v2
from src.pipeline.mapping_features_v2 import json_to_csv_with_level2
from src.utils.file_handling import set_training_data, csv_to_json
from src.datas.preprocess import DataPreprocess
from src.service.data_preprocessing import DataPreProcessing
from src.service.apriori import Apriori
from mlxtend.frequent_patterns import association_rules
from gen_rec import check_triggered_rules, load_json_file, load_association_rules

app = Flask(__name__)

@app.route('/process_data', methods=['POST'])
def process_data():
    filename = request.json.get('filename')
    if not filename:
        return jsonify({"error": "Filename is required"}), 400

    # Process data as in main_v2.py
    try:
        filepath = os.path.join("raw", os.path.splitext(str(filename))[0])
        nlp = spacy.load("en_core_web_sm")
        word_freq = word_frequency
        data = csv_to_json(filepath)

        preprocess = DataPreprocess(data, nlp, filepath)
        preprocess.start()

        path = os.path.join(os.getcwd(), 'data', f'{filepath}_preprocessed.json')
        with open(path, 'r') as f:
            data = json.load(f)
            featex = extract_features_v2(data, filepath, nlp, word_freq)

        json_to_csv_with_level2('q_bank_features_v2')

        with open('data/raw/q_bank_features_v2.json', 'r') as f:
            json_data = json.load(f)

        output = []
        questions_by_level = {"level_1": [], "level_2": [], "level_3": []}

        for item in json_data:
            row_data = {'question_text': item['question_text'], 'answer': item['answer'], 'options_comp': item['options_comp']}
            output.append(row_data)
            if item['levels']['Level_1']:
                questions_by_level['level_1'].append(row_data)
            elif item['levels']['Level_2']:
                questions_by_level['level_2'].append(row_data)
            else:
                questions_by_level['level_3'].append(row_data)

        with open('data/questions_by_level.json', 'w') as jsonfile:
            json.dump(questions_by_level, jsonfile, indent=4)

        with open('data/output.json', 'w') as jsonfile:
            json.dump(output, jsonfile, indent=4)

        return jsonify({"message": "Data processed successfully"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/generate_rules', methods=['POST'])
def generate_rules():
    try:
        # Process data as in asrul_v2.py
        data_preprocessing = DataPreProcessing()
        apriori = Apriori()

        df_mapping_question_comp = pd.read_csv("data/apriori/mapping-assessment-question-competency.csv")
        df_questions = pd.read_csv("data/apriori/assessment-questions.csv")
        df_test_results = pd.read_csv("data/apriori/assessment-result.csv")

        transormed_data = data_preprocessing.transform_result_to_biner(df_test_results, df_questions)
        student_comp = data_preprocessing.mapping_student_competency(transormed_data, df_mapping_question_comp)
        final_dataset = data_preprocessing.generate_final_dataset(student_comp)
        transform_dataset = data_preprocessing.data_transformation(final_dataset)

        frequent_itemsets = apriori.apriori(final_dataset, 0.97)
        results = pd.DataFrame(list(frequent_itemsets.items()), columns=['itemsets', 'support'])
        results['itemsets'] = results['itemsets'].apply(lambda x: tuple(x))
        results = results[['support', 'itemsets']].sort_values(by='support', ascending=False).reset_index(drop=True)

        rules = association_rules(results, metric="confidence", min_threshold=0.97)
        rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(map(str, x)) if isinstance(x, (set, frozenset)) else x)
        rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(map(str, x)) if isinstance(x, (set, frozenset)) else x)
        rules['combined'] = rules['antecedents'] + ',' + rules['consequents']

        rules[['antecedents', 'consequents', 'combined']].to_csv("./data/apriori/association_rules.csv", index=False)

        return jsonify({"message": "Association rules generated successfully"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/generate_recommendations', methods=['POST'])
def generate_recommendations():
    try:
        # Load competencies from the JSON file
        json_filepath = 'data/competencies.json'  # Update the path as needed
        competencies = load_json_file(json_filepath)
        if competencies is None:
            return jsonify({"error": "Competencies file not found"}), 404

        # Load the association rules from CSV
        csv_filepath = 'data/apriori/association_rules.csv'  # Update the path as needed
        rules_df = load_association_rules(csv_filepath)
        if rules_df is None:
            return jsonify({"error": "Association rules file not found"}), 404

        # Get the triggered rules
        triggered_rules = check_triggered_rules(competencies, rules_df)

        # Return the top 3 triggered rules as a JSON response
        return jsonify({"triggered_rules": triggered_rules}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)