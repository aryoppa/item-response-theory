from flask import Flask, request, session, jsonify
import pandas as pd
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gen_rec import load_association_rules, check_triggered_rules
from flask_cors import CORS
# from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_unique_and_secret_key'  # For session management
CORS(app)  # Allow CORS for communication with other systems (e.g., Laravel)

# Load dataset and configurations
DATA_DIR = os.path.join(os.getcwd(), 'data')
csv_filepath = os.path.join(DATA_DIR, 'mapped', 'questions_database.csv')
questions_df = pd.read_csv(csv_filepath)

competency_columns = [
    "main_verbs", "tense", "infinitives", "passives", "have_+_participle",
    "auxiliary_verbs", "pronouns", "nouns", "determiners",
    "other_adjectives", "prepositions", "conjunctions", "subject_verb_agreement"
]

# Load GPT-2 model and tokenizer
# model_name = 'gpt2'  # or 'gpt2-medium', 'gpt2-large' if you need larger models
# model = GPT2LMHeadModel.from_pretrained(model_name)
# tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Helper functions
def calculate_probability(theta: float, a: float, b: float, c: float) -> float:
    """
    Calculate the probability of a correct response based on Item Response Theory (IRT) parameters.
    
    Parameters:
        theta (float): The learner's ability level.
        a (float): The discrimination parameter of the question.
        b (float): The difficulty parameter of the question.
        c (float): The guessing parameter of the question (minimum probability of correctness).
        
    Returns:
        float: The probability of the learner answering the question correctly.
    """
    exponent = -a * (theta - b)
    return c + (1 - c) / (1 + np.exp(exponent))

def competency_similarity(target_vector: list, question_vector: list) -> float:
    """
    Calculate cosine similarity between target and question competency vectors.
    
    Parameters:
        target_vector (list): A list representing the learner's target competencies.
        question_vector (list): A list representing the question's competency requirements.
        
    Returns:
        float: The cosine similarity value between the two vectors (range: -1 to 1).
    
    Raises:
        ValueError: If the vectors are not of the same dimension.
    """
    target_vector = np.array(target_vector).reshape(1, -1)
    question_vector = np.array(question_vector).reshape(1, -1)
    return cosine_similarity(target_vector, question_vector)[0][0]

def select_next_question(theta: float, questions: pd.DataFrame, target_competencies: list) -> pd.Series:
    """
    Select the next question for the learner based on cosine similarity to target competencies 
    and information gain calculated from IRT parameters.
    
    Parameters:
        theta (float): The learner's ability level.
        questions (pd.DataFrame): A DataFrame containing questions with columns:
                                  - 'a': Discrimination parameter.
                                  - 'b': Difficulty parameter.
                                  - 'c': Guessing parameter.
                                  - Competency columns (vectors for similarity calculation).
        target_competencies (list): A list representing the learner's target competencies.
    
    Returns:
        pd.Series: The row corresponding to the question with the highest combined information gain and similarity.
    
    Raises:
        ValueError: If the input DataFrame is empty.
    """
    questions = questions.copy()
    questions['similarity'] = questions.apply(
        lambda row: competency_similarity(target_competencies, row[competency_columns].values), axis=1
    )
    questions['information'] = questions.apply(
        lambda row: calculate_probability(theta, row['a'], row['b'], row['c']) * row['similarity'], axis=1
    )
    # print(questions.loc[questions['information'].idxmax()])
    # if questions.empty:
    #     raise ValueError("The questions DataFrame is empty. Cannot select the next question.")
    
    return questions.loc[questions['information'].idxmax()]
    
def generate_recommendation(antecedents, consequents):
    # Create a recommendation sentence based on the competencies
    prompt = f"Berdasarkan hasil tes Anda, Anda perlu mempelajari ulang bab berikut: {', '.join(antecedents)}. " \
             f"Anda juga dihimbau untuk mempelajari bab {', '.join(consequents)} untuk meningkatkan kemampuan Anda."
    
    # inputs = tokenizer.encode(prompt, return_tensors='pt')
    # outputs = model.generate(inputs, max_length=150, num_return_sequences=1, temperature=0.0)
    
    # recommendation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return prompt

# API Endpoints
@app.route('/get-question', methods=['POST'])
def get_question():
    """Fetch the next question based on current state."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request body must be JSON"}), 400

        asked_questions = data.get('asked_questions', [])
        target_competencies = data.get('target_competencies', [1] * len(competency_columns))
        
        # Retrieve theta value (default to 0.0 if not found)
        theta = data.get('theta', 0.0)

        # Filter available questions
        available_questions = questions_df[~questions_df['question_id'].isin(asked_questions)]

        # Return an error if no more questions are available
        if available_questions.empty:
            return jsonify({"error": "No more questions available"}), 400

        # Select the next question
        if not asked_questions:
            question = available_questions.sample(1).iloc[0]
        else:
            question = select_next_question(theta, available_questions, target_competencies)

        response = {
            "question_id": int(question['question_id']),
            "question_text": question['question_text'],
            "option_A": question['option_A'],
            "option_B": question['option_B'],
            "option_C": question['option_C'],
            "option_D": question['option_D'],
            "key_answer": question['key_answer'],
            "a_value": float(question['a']),
            "b_value": float(question['b']),
            "c_value": float(question['c']),
            "theta": theta
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/submit-answer', methods=['POST'])
def submit_answer():
    """Process the submitted answer and update session data."""
    try:
        data = request.get_json()

        # Validate input data
        if not data or 'question_id' not in data or 'is_correct' not in data:
            return jsonify({"error": "Missing required fields: 'question_id' and 'is_correct'"}), 400
        # print(data)

        # Extract input data
        theta = float(data.get('theta', 0))  # Default to 0 if theta is missing
        question_id = int(data['question_id'])
        is_correct = int(data['is_correct'])

        # Validate the is_correct field
        if is_correct not in [0, 1]:
            return jsonify({"error": "'is_correct' must be 0 or 1"}), 400

        # Retrieve the question details
        question = questions_df[questions_df['question_id'] == question_id]
        if question.empty:
            return jsonify({"error": "Invalid question_id"}), 400
        question = question.iloc[0]

        # Update theta based on correctness
        a, b, c = question['a'], question['b'], question['c']
        prob = calculate_probability(theta, float(a), float(b), float(c))
        gradient = (is_correct - prob) * float(a)
        theta += 0.1 * gradient  # Learning rate is 0.1

        # Initialize session key if it does not exist
        if 'incorrect_answers' not in session:
            session['incorrect_answers'] = []

        # Prepare response
        response = {
            "theta": theta,
            "question_id": question_id,
            "question_text": question['question_text'],
            "correct_answer": question['key_answer']
        }
        return jsonify(response)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
        
@app.route('/analyze-incorrect-answers', methods=['POST'])
def analyze_incorrect_answers():
    """Analyze uploaded data, filter incorrect answers, and sum competencies."""
    try:
        # Retrieve data sent from Laravel
        data = request.get_json()

        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Filter incorrect answers
        incorrect_answers = [entry for entry in data if entry.get('is_correct') == 0]

        # Filter correct answers
        correct_answers = [entry for entry in data if entry.get('is_correct') == 1]

        if not incorrect_answers:
            return jsonify({"message": "No incorrect answers found"}), 200

        # Extract question IDs from incorrect answers
        incorrect_question_ids = [int(answer['question_id']) for answer in incorrect_answers]

        # Extract question IDs from correct answers
        correct_question_ids = [int(answer['question_id']) for answer in correct_answers]

        # Ensure question_id in DataFrame is of type integer for comparison
        questions_df['question_id'] = questions_df['question_id'].astype(int)

        # Filter questions from the DataFrame based on incorrect question IDs
        incorrect_filtered_questions = questions_df[questions_df['question_id'].isin(incorrect_question_ids)]

        # Filter questions from the DataFrame based on correct question IDs
        correct_filtered_questions = questions_df[questions_df['question_id'].isin(correct_question_ids)]

        # Sum the competencies matrix for the filtered questions
        incorrect_competencies_sum = incorrect_filtered_questions[competency_columns].sum().tolist()
        incorrect_competencies_sums = incorrect_filtered_questions[competency_columns].sum().to_dict()
        correct_competencies_sum = correct_filtered_questions[competency_columns].sum().tolist()
        # Remove all competencies with a value of 1
        adjusted_competencies_sum = {key: (0 if value == 1 else value) for key, value in incorrect_competencies_sums.items()}
        # cek = filtered_competencies_sum.to_dict()
        print(adjusted_competencies_sum)
        # Get the top 3 sums while maintaining the original order of the competencies
        top_3_competencies = {
            key: value
            for key, value in adjusted_competencies_sum.items()
            if value in sorted(adjusted_competencies_sum.values(), reverse=True)[:3]
        }

        # Keep the order of the competencies in the original dictionary
        top_3_competencies_ordered = {
            key: top_3_competencies[key]
            for key in adjusted_competencies_sum.keys()
            if key in top_3_competencies
        }
        # Extract the competencies that need improvement
        competencies_needing_improvement = [comp for comp, score in adjusted_competencies_sum.items() if score > 0]

        if not competencies_needing_improvement:
            return jsonify({"message": "No competencies identified for improvement."}), 200

        # Load association rules
        rules_df = load_association_rules(os.path.join(DATA_DIR, 'apriori', 'association_rules.csv'))
        if rules_df is None:
            return jsonify({"error": "Association rules file not found"}), 404

        # Check triggered rules based on incorrect competencies
        triggered_rules_df = check_triggered_rules(top_3_competencies_ordered, rules_df)

        # Extracting and combining antecedents and consequents from all triggered rules
        antecedents = set()
        consequents = set()
        for _, rule in triggered_rules_df.iterrows():
            antecedents.update(rule['antecedents'].split(', '))
            consequents.update(rule['consequents'].split(', '))

        # Generate the recommendation sentence
        recommendation = generate_recommendation(list(antecedents), list(consequents))

        # Convert triggered rules to JSON serializable format
        triggered_rules = triggered_rules_df.to_dict(orient='records')

        # Prepare and return the response
        response = {
            "incorrect_question_ids": incorrect_question_ids,
            "filtered_competencies_sum": top_3_competencies_ordered,
            "incorrect_competencies_sum": incorrect_competencies_sum,
            "correct_question_ids": correct_question_ids,
            "correct_competencies_sum": correct_competencies_sum,
            "triggered_rules": triggered_rules,
            "recommendations": recommendation
        }
        return jsonify(response)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
