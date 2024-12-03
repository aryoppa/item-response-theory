from flask import Flask, request, jsonify
import os
import json
import spacy
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
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

DATA_DIR = os.path.join(os.getcwd(), 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
APRIORI_DIR = os.path.join(DATA_DIR, 'apriori')
MAPPED_DIR = os.path.join(DATA_DIR, 'mapped')


@app.route('/process_data', methods=['POST'])
def process_data():
    filename = request.json.get('filename')
    if not filename:
        return jsonify({"error": "Filename is required"}), 400

    try:
        filepath = os.path.join(RAW_DIR, os.path.splitext(str(filename))[0])
        nlp = spacy.load("en_core_web_sm")
        data = csv_to_json(filepath)

        preprocess = DataPreprocess(data, nlp, filepath)
        preprocess.start()

        preprocessed_path = os.path.join(DATA_DIR, f'{filepath}_preprocessed.json')
        with open(preprocessed_path, 'r') as f:
            data = json.load(f)
            extract_features_v2(data, filepath, nlp, word_frequency)

        json_to_csv_with_level2('q_bank_features_v2')

        with open(os.path.join(DATA_DIR, 'raw/q_bank_features_v2.json'), 'r') as f:
            json_data = json.load(f)

        output, questions_by_level = [], {"level_1": [], "level_2": [], "level_3": []}
        for item in json_data:
            row_data = {
                'question_text': item['question_text'],
                'answer': item['answer'],
                'options_comp': item['options_comp']
            }
            output.append(row_data)
            if item['levels']['Level_1']:
                questions_by_level['level_1'].append(row_data)
            elif item['levels']['Level_2']:
                questions_by_level['level_2'].append(row_data)
            else:
                questions_by_level['level_3'].append(row_data)

        with open(os.path.join(DATA_DIR, 'questions_by_level.json'), 'w') as jsonfile:
            json.dump(questions_by_level, jsonfile, indent=4)
        with open(os.path.join(DATA_DIR, 'output.json'), 'w') as jsonfile:
            json.dump(output, jsonfile, indent=4)

        return jsonify({"message": "Data processed successfully"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/generate_rules', methods=['POST'])
def generate_rules():
    try:
        data_preprocessing = DataPreProcessing()
        apriori = Apriori()

        df_mapping_question_comp, df_questions, df_test_results = load_data_files()
        if any(df is None for df in [df_mapping_question_comp, df_questions, df_test_results]):
            return jsonify({"error": "Error loading CSV files"}), 500

        transformed_data = data_preprocessing.transform_result_to_biner(df_test_results, df_questions)
        student_comp = data_preprocessing.mapping_student_competency(transformed_data, df_mapping_question_comp)
        final_dataset = data_preprocessing.generate_final_dataset(student_comp)
        frequent_itemsets = apriori.apriori(final_dataset, 0.97)
        results = pd.DataFrame(list(frequent_itemsets.items()), columns=['itemsets', 'support'])
        results['itemsets'] = results['itemsets'].apply(lambda x: tuple(x))
        results = results.sort_values(by='support', ascending=False).reset_index(drop=True)
        rules = association_rules(results, metric="confidence", min_threshold=0.97)
        rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(map(str, x)))
        rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(map(str, x)))
        rules['combined'] = rules['antecedents'] + ',' + rules['consequents']
        rules[['antecedents', 'consequents', 'combined']].to_csv(os.path.join(APRIORI_DIR, "association_rules.csv"),
                                                                  index=False)

        return jsonify({"message": "Association rules generated successfully"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def validate_input_data(data):
    if not data or not isinstance(data, dict):
        return "'data' key is missing or the format is incorrect"
    competencies = data.get("data")
    if not competencies or not isinstance(competencies, list):
        return "'data' must be provided as a non-empty list"
    for index, value in enumerate(competencies):
        if not isinstance(value, dict) or 'question_id' not in value:
            return f"'question_id' key missing in data at index {index}"
    return None


def get_consequents_matrix(triggered_rules, competencies_list):
    consequents_matrix = []
    for _, row in triggered_rules.iterrows():
        consequents = row['consequents']
        if isinstance(consequents, str):
            consequents = [consequents]
        competency_mapping = {competency: 0 for competency in competencies_list}
        for consequent in consequents:
            competency_mapping[consequent.strip().lower()] = 1
        consequents_matrix.append([competency_mapping[competency] for competency in competencies_list])
    return np.array(consequents_matrix)


@app.route('/generate_recommendations', methods=['POST'])
def generate_recommendations():
    try:
        data = request.get_json()
        validation_error = validate_input_data(data)
        if validation_error:
            return jsonify({"error": validation_error}), 400

        competencies = data.get("data")
        question_ids = [str(value['question_id']) for value in competencies]

        df_questions = get_mapped_questions()
        df_questions['question_id'] = df_questions['question_id'].astype(str)
        filtered_df = df_questions[df_questions['question_id'].isin(question_ids)]
        combined_dict = filtered_df.loc[:, 'main_verbs':].any().to_dict()
        combined_dict = {k: bool(v) for k, v in combined_dict.items()}
        rules_df = load_association_rules(os.path.join(APRIORI_DIR, 'association_rules.csv'))
        if rules_df is None:
            return jsonify({"error": "Association rules file not found"}), 404

        triggered_rules = check_triggered_rules(combined_dict, rules_df)
        competencies_list = ["main_verbs", "tense", "infinitives", "passives", "have_+_participle",
                             "auxiliary_verbs", "pronouns", "nouns", "determiners",
                             "other_adjectives", "prepositions", "conjunctions", "subject_verb_agreement"]
        consequents_matrix = get_consequents_matrix(triggered_rules, competencies_list)
        rules1, rules2, rules3 = consequents_matrix[0], consequents_matrix[1], consequents_matrix[2]

        level_2_questions = df_questions[df_questions['Level'] == 'Level 2']
        questions_matrix_df = level_2_questions[["question_id"]].join(level_2_questions[competencies_list].astype(int))
        questions_matrix = questions_matrix_df[competencies_list].values

        similarity_scores_1 = cosine_similarity(questions_matrix, [rules1])
        similarity_scores_2 = cosine_similarity(questions_matrix, [rules2])
        similarity_scores_3 = cosine_similarity(questions_matrix, [rules3])

        questions_matrix_df['similarity_score_1'] = similarity_scores_1.flatten()
        questions_matrix_df['similarity_score_2'] = similarity_scores_2.flatten()
        questions_matrix_df['similarity_score_3'] = similarity_scores_3.flatten()

        top_recommendations = pd.concat([
            questions_matrix_df.sort_values(by='similarity_score_1', ascending=False).head(1),
            questions_matrix_df.sort_values(by='similarity_score_2', ascending=False).head(1),
            questions_matrix_df.sort_values(by='similarity_score_3', ascending=False).head(1)
        ]).drop_duplicates()

        triggered_rules_dict = triggered_rules.to_dict(orient='records')
        recommended_questions = top_recommendations.to_dict(orient='records')
        level_2_questions_dict = df[(df['question_id'].isin([rec['question_id'] for rec in recommended_questions]))].to_dict(
            orient='records')

        return jsonify({"triggered_rules": triggered_rules_dict, "level_2_questions": level_2_questions_dict}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/get_random_level_2_questions', methods=['GET'])
def get_random_level_2_questions():
    try:
        csv_filepath = os.path.join(MAPPED_DIR, 'clean_questions.csv')
        df = pd.read_csv(csv_filepath)
        random_questions = df[df['Level'] == 'Level 2'].sample(n=3).to_dict(orient='records')
        return jsonify({"questions": random_questions}), 200
    except FileNotFoundError:
        return jsonify({"error": "clean_questions.csv file not found"}), 404
    except ValueError:
        return jsonify({"error": "Not enough level 2 questions available"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/item_response_theory', methods=['POST', 'GET'])
def item_response_theory():
    try:
        # Placeholder for item_response_theory implementation
        return jsonify({"message": "Item Response Theory placeholder"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def load_data_files():
    try:
        df_mapping_question_comp = pd.read_csv(os.path.join(APRIORI_DIR, "mapping-assessment-question-competency.csv"))
        df_questions = pd.read_csv(os.path.join(APRIORI_DIR, "assessment-questions.csv"))
        df_test_results = pd.read_csv(os.path.join(APRIORI_DIR, "assessment-result.csv"))
        return df_mapping_question_comp, df_questions, df_test_results
    except FileNotFoundError:
        return None, None, None


def get_mapped_questions():
    try:
        df_questions = pd.read_csv(os.path.join(MAPPED_DIR, "mapped_questions_competencies.csv"))
        return df_questions
    except FileNotFoundError:
        return None


if __name__ == '__main__':
    app.run(debug=True)