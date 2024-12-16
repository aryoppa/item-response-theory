from flask import Flask, request, session, jsonify
import pandas as pd
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS

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

# Helper functions
def calculate_probability(theta, a, b, c):
    """Calculate the probability of a correct response based on IRT parameters."""
    exponent = -a * (theta - b)
    return c + (1 - c) / (1 + np.exp(exponent))

def competency_similarity(target_vector, question_vector):
    """Calculate cosine similarity between target and question competency vectors."""
    target_vector = np.array(target_vector).reshape(1, -1)
    question_vector = np.array(question_vector).reshape(1, -1)
    return cosine_similarity(target_vector, question_vector)[0][0]

def select_next_question(theta, questions, target_competencies):
    """Select the next question based on similarity and information gain."""
    questions = questions.copy()
    questions['similarity'] = questions.apply(
        lambda row: competency_similarity(target_competencies, row[competency_columns].values), axis=1
    )
    questions['information'] = questions.apply(
        lambda row: calculate_probability(theta, row['a'], row['b'], row['c']) * row['similarity'], axis=1
    )
    return questions.loc[questions['information'].idxmax()]

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

        # Filter available questions
        available_questions = questions_df[~questions_df['question_id'].isin(asked_questions)]

        # Return an error if no more questions are available
        if available_questions.empty:
            return jsonify({"error": "No more questions available"}), 400

        # Select the next question
        if not asked_questions:
            question = available_questions.sample(1).iloc[0]
        else:
            question = select_next_question(session.get('theta', 0.0), available_questions, target_competencies)

        response = {
            "question_id": int(question['question_id']),
            "question_text": question['question_text'],
            "option_A": question['option_A'],
            "option_B": question['option_B'],
            "option_C": question['option_C'],
            "option_D": question['option_D'],
            "key_answer": question['key_answer'],
            "a": float(question['a']),
            "b": float(question['b']),
            "c": float(question['c']),
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/submit-answer', methods=['POST'])
def submit_answer():
    """Process the submitted answer and update session data."""
    try:
        data = request.get_json()
        if not data or 'question_id' not in data or 'is_correct' not in data:
            return jsonify({"error": "Missing required fields: 'question_id' and 'is_correct'"}), 400

        session.setdefault('theta', 0.0)  # Initialize theta in session if not present
        session.setdefault('incorrect_answers', [])  # Initialize incorrect answers list if not present

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
        prob = calculate_probability(session['theta'], float(a), float(b), float(c))
        gradient = (is_correct - prob) * float(a)
        session['theta'] += 0.1 * gradient

        # Store incorrect answers
        if is_correct == 0:
            session['incorrect_answers'].append({
                "question_id": question_id,
                "question_text": question['question_text'],
                "correct_answer": question['key_answer'],
                "selected_answer": data.get('selected_answer', None)
            })

        response = {
            "theta": session['theta'],
            "incorrect_answers": session['incorrect_answers']
        }
        return jsonify(response)

    except Exception as e:
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

        # Debugging: Print DataFrame and question_id types
        print("Incorrect Question IDs:", incorrect_question_ids)
        print("DataFrame question_id Types:", questions_df['question_id'].dtype)

        # Filter questions from the DataFrame based on incorrect question IDs
        incorrect_filtered_questions = questions_df[questions_df['question_id'].isin(incorrect_question_ids)]

        # Filter questions from the DataFrame based on correct question IDs
        correct_filtered_questions = questions_df[questions_df['question_id'].isin(correct_question_ids)]

        # Debugging: Print filtered questions
        print("Filtered Questions:\n", incorrect_filtered_questions)

        if incorrect_filtered_questions.empty:
            return jsonify({"error": "No matching questions found for incorrect answers"}), 400

        # Sum the competencies matrix for the filtered questions
        incorrect_competencies_sum = incorrect_filtered_questions[competency_columns].sum().tolist()
        # Sum the competencies matrix for the filtered questions
        correct_competencies_sum = correct_filtered_questions[competency_columns].sum().tolist()

        # Prepare and return the response
        response = {
            "incorrect_question_ids": incorrect_question_ids,
            "competencies_sum": incorrect_competencies_sum,
            "correct_question_ids": correct_question_ids,
            "correct_competencies_sum": correct_competencies_sum
        }
        return jsonify(response)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500




if __name__ == '__main__':
    app.run(debug=True)
