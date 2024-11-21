from flask import Flask, render_template, request, jsonify
import pandas as pd
import random
import json
import os

app = Flask(__name__)

# Load questions data
questions_df = pd.read_csv('data/mapped/clean_questions.csv')
questions_df2 = pd.read_csv('data/mapped/mapped_questions_competencies.csv')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/stage_one', methods=['GET'])
def stage_one():
    # Filter questions with level == 2
    level_two_questions = questions_df[questions_df['Level'] == "Level 2"]
    
    # Select 3 random questions
    stage_one_questions = level_two_questions.sample(n=3)
    
    # Convert questions to dictionary for rendering
    questions = stage_one_questions.to_dict(orient='records')
    return render_template('stage_one.html', questions=questions)

import traceback

@app.route('/check_answers', methods=['POST'])
def check_answers():
    try:
        user_answers = request.json.get('answers')
        print("Debug: User answers received:", user_answers)  # Debugging line

        if not user_answers:
            return jsonify({'error': 'No answers provided'}), 400

        wrong_answers = []

        for answer in user_answers:
            question_id = answer.get('question_id')
            user_option = answer.get('selected_option')
            print(f"Debug: Processing question ID {question_id} with user option {user_option}")  # Debugging line

            if question_id is None or user_option is None:
                continue

            correct_answer = questions_df.loc[questions_df['question_id'] == int(question_id), 'key_answer'].values
            print(f"Debug: Correct answer for question ID {question_id} is {correct_answer}")  # Debugging line

            if correct_answer.size > 0 and user_option != correct_answer[0]:
                wrong_answers.append(question_id)

        # Save wrong question_ids to a JSON file in the specified location
        output_dir = 'data/answers'
        os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist
        output_path = os.path.join(output_dir, 'answers.json')
        
        json_data = {'wrong_questions': wrong_answers}
        with open(output_path, 'w') as json_file:
            json.dump(json_data, json_file, indent=4)

        # Return the results
        return jsonify({'wrong_answers': wrong_answers})

    except Exception as e:
        print(f"Debug Error: {e}")  # Debugging line
        traceback.print_exc()
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True)
