import numpy as np
import pandas as pd
import os
DATA_DIR = os.path.join(os.getcwd(), 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
APRIORI_DIR = os.path.join(DATA_DIR, 'apriori')
MAPPED_DIR = os.path.join(DATA_DIR, 'mapped')
# Define the 3PL model function
def calculate_probability(theta, a, b, c):
    """
    Calculate the probability of a correct response using the 3PL model.
    """
    exponent = -a * (theta - b)
    probability = c + (1 - c) * (1 / (1 + np.exp(exponent)))
    return probability

# Define item information function
def item_information(theta, a, b, c):
    """
    Calculate item information at a given theta.
    """
    P_theta = calculate_probability(theta, a, b, c)
    return (a ** 2) * P_theta * (1 - P_theta)

# Update theta using gradient descent
def update_theta(current_theta, responses, question_params, learning_rate=0.1):
    """
    Update the test-taker's ability (theta) using gradient descent.
    """
    gradient = 0
    for response in responses:
        question = question_params.loc[question_params['question_id'] == response['question_id']].iloc[0]
        a, b, c = question['a'], question['b'], question['c']
        prob = calculate_probability(current_theta, a, b, c)
        gradient += (response['is_correct'] - prob) * a

    return current_theta + learning_rate * gradient

# Select the next most informative question
def select_next_question(theta, questions):
    """
    Select the most informative question for a given theta.
    """
    questions = questions.copy()  # Create a copy to avoid the SettingWithCopyWarning
    questions['information'] = questions.apply(
        lambda row: item_information(theta, row['a'], row['b'], row['c']), axis=1
    )
    return questions.loc[questions['information'].idxmax()]


# Load the dataset
csv_filepath = os.path.join(MAPPED_DIR, 'questions_database.csv')
questions_df = pd.read_csv(csv_filepath)


# Test simulation
theta = 0.0  # Initial ability estimate
asked_questions = []
responses = []

for _ in range(10):  # Limit to 10 questions for this example
    # Select the next question
    next_question = select_next_question(theta, questions_df[~questions_df['question_id'].isin(asked_questions)])
    print(f"Next question: {next_question['question_text']}")

    # Simulate a response
    prob_correct = calculate_probability(theta, next_question['a'], next_question['b'], next_question['c'])
    is_correct = np.random.choice([0, 1], p=[1 - prob_correct, prob_correct])
    print(is_correct)

    # Record the response
    responses.append({"question_id": next_question['question_id'], "is_correct": is_correct})
    asked_questions.append(next_question['question_id'])

    # Update theta
    theta = update_theta(theta, responses, questions_df)
    print(f"Updated theta: {theta}\n")

print("Test completed.")
print(f"Final estimated ability (theta): {theta}")
