from flask import Flask, request, render_template, jsonify  # Import jsonify
import numpy as np
import pandas as pd
import pickle
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure the Gemini API
api_key = os.getenv("GEMINI_API_KEY")
gemini_model = None
if not api_key:
    print("Warning: GEMINI_API_KEY not found. The AI chatbot will not work.")
else:
    genai.configure(api_key=api_key)
    # Set system instruction for the model
    system_instruction = "You are a helpful medical AI assistant called DoseWise. Always end your response with the disclaimer: 'Disclaimer: I am an AI assistant and not a substitute for professional medical advice. Please consult a doctor for any health concerns.'"
    gemini_model = genai.GenerativeModel(
        'gemini-1.5-flash-latest', system_instruction=system_instruction
    )
    print("Gemini AI Model Initialized.")
# flask app
app = Flask(__name__)

# --- Path Setup ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'Model')

# --- Data and Model Loading ---

# Load the trained model
svc = pickle.load(open(os.path.join(MODEL_DIR, 'svc.pkl'), 'rb'))

# Load the symptoms dictionary and diseases list from pickle files
with open(os.path.join(MODEL_DIR, 'symptoms_dict.pkl'), 'rb') as f:
    symptoms_dict = pickle.load(f)
with open(os.path.join(MODEL_DIR, 'diseases_list.pkl'), 'rb') as f:
    diseases_list = pickle.load(f)

# Normalize symptom keys to lowercase for consistent matching
normalized_symptoms_dict = {key.lower().strip(): value for key, value in symptoms_dict.items()}

# Load datasets into memory at startup for efficiency
description_df = pd.read_csv("datasets/description.csv", index_col='Disease')
precautions_df = pd.read_csv("datasets/precautions_df.csv", index_col='Disease')
medications_df = pd.read_csv('datasets/medications.csv', index_col='Disease')
diets_df = pd.read_csv("datasets/diets.csv", index_col='Disease')
workout_df = pd.read_csv("datasets/workout_df.csv", index_col='disease')

# --- Helper Functions ---

def _to_list(data):
    """Converts pandas Series to list, or wraps a single value in a list."""
    if isinstance(data, pd.Series):
        return data.dropna().tolist()
    if isinstance(data, np.ndarray):
        return [item for item in data.flatten() if pd.notna(item)]
    if pd.notna(data):
        return [data]
    return []

def get_disease_details(disease):
    """
    Retrieves description, precautions, medications, diet, and workout for a given disease.
    """
    desc = description_df.loc[disease, 'Description']
    precautions = _to_list(precautions_df.loc[disease].values.flatten())
    medications = _to_list(medications_df.loc[disease, 'Medication'])
    diet = _to_list(diets_df.loc[disease, 'Diet'])
    workout = _to_list(workout_df.loc[disease, 'workout'])
    return desc, precautions, medications, diet, workout

# Model Prediction function
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    missing_symptoms = []
    symptoms_found_count = 0
    for item in patient_symptoms:
        item = item.lower().strip()  # Normalize user input
        if item in normalized_symptoms_dict:
            input_vector[normalized_symptoms_dict[item]] = 1
            symptoms_found_count += 1
        else:
            missing_symptoms.append(item)  # Track missing symptoms

    if missing_symptoms:
        app.logger.warning(f"Symptoms not found: {', '.join(missing_symptoms)}")

    # If no valid symptoms are found, we can't make a prediction.
    if symptoms_found_count == 0:
        return None

    # Get prediction probabilities
    probabilities = svc.predict_proba(input_vector.reshape(1, -1))[0]
    prediction_index = np.argmax(probabilities)
    confidence = probabilities[prediction_index]
    
    return diseases_list[prediction_index] # Return the disease name

# --- Routes ---
@app.route("/ask_ai", methods=["POST"])
def ask_ai():
    try:
        if not gemini_model:
            return jsonify({"error": "AI model is not configured. Missing API key."}), 503

        data = request.get_json()
        user_message = data.get("message")

        if not user_message:
            return jsonify({"error": "No message provided"}), 400
        
        # Send the user's message directly to the model
        response = gemini_model.generate_content(user_message)

        # Check if the response was blocked or is empty
        if not response.parts:
            # Check for specific block reasons
            block_reason = response.prompt_feedback.block_reason.name if response.prompt_feedback else "UNKNOWN"
            print(f"AI response was blocked. Reason: {block_reason}")
            return jsonify({"reply": "I'm sorry, I can't respond to that. Please ask a different question."}), 200
        else:
            return jsonify({"reply": response.text})

    except Exception as e:
        print(f"Error in /ask_ai: {e}") # Log the specific error to the console
        return jsonify({"error": str(e)}), 500

@app.route("/")
def index():
    return render_template("index.html")

# Define a route for the home page
@app.route('/predict', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        symptoms = request.form.get('symptoms')
        if not symptoms or symptoms.strip().lower() == "symptoms":
            message = "Please either write symptoms or you have written misspelled symptoms."
            return render_template('index.html', message=message)
        else:
            try:
                # Process symptoms
                user_symptoms = [s.strip() for s in symptoms.split(',')]
                user_symptoms = [symptom.strip("[]' ") for symptom in user_symptoms]

                # Predict the disease
                predicted_disease = get_predicted_value(user_symptoms)
                
                if predicted_disease is None:
                    message = "Could not determine the disease with high confidence. Please provide more symptoms for a better prediction."
                    return render_template('index.html', message=message)
                

                # Get additional details about the disease using the new helper
                dis_des, my_precautions, medications, rec_diet, workout = get_disease_details(predicted_disease)

                # Pass the predicted disease and other details to the template
                return render_template(
                    'index.html', 
                    predicted_disease=predicted_disease,
                    dis_des=dis_des,
                    my_precautions=my_precautions,
                    medications=medications,
                    my_diet=rec_diet,
                    workout=workout,
                    show_map=True
                )
            except Exception as e:
                message = f"Error: {str(e)}"
                app.logger.error(f"An error occurred during prediction: {e}")
                return render_template('index.html', message=message)

    return render_template('index.html')

@app.route('/symptoms')
def symptoms():
    valid_symptoms = list(symptoms_dict.keys())
    return render_template('symptoms.html', symptoms=valid_symptoms)

# about view funtion and path
@app.route('/about')
def about():
    return render_template("about.html")
# contact view funtion and path
@app.route('/contact')
def contact():
    return render_template("contact.html")

# developer view funtion and path
@app.route('/developer')
def developer():
    return render_template("developer.html")

# about view funtion and path
@app.route('/blog')
def blog():
    return render_template("blog.html")

@app.route('/map')
def map():
    return render_template('map.html')

if __name__ == '__main__':

    app.run(debug=True)
