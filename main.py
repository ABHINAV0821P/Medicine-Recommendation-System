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
# allow overriding model name from env
gemini_model_name = os.getenv("GEMINI_MODEL", "models/gemini-flash-latest")
if not api_key:
    print("Warning: GEMINI_API_KEY not found. The AI chatbot will not work.")
else:
    genai.configure(api_key=api_key)
    # Best-effort: list available models the API key can access and their supported methods
    try:
        print("Attempting to list available Gemini/Generative models for this API key...")
        models_info = genai.list_models()
        try:
            # models_info may be an iterable of objects
            available_models = [getattr(m, 'name', m) for m in models_info]
        except Exception:
            available_models = models_info
        print("Available models (showing first 50):")
        for m in available_models[:50]:
            print(" -", m)
    except Exception as e:
        print(f"Could not list models at startup: {e}")
    # Set system instruction for the model
    system_instruction = (
        "You are a helpful medical AI assistant called DoseWise. "
        "Always end your response with the disclaimer: 'Disclaimer: I am an AI assistant and not a substitute for professional medical advice. Please consult a doctor for any health concerns.'"
    )
    try:
        gemini_model = genai.GenerativeModel(gemini_model_name, system_instruction=system_instruction)
        print(f"Gemini AI Model '{gemini_model_name}' Initialized.")
    except Exception as e:
        gemini_model = None
        # If model not found or not supported for the requested method, attempt to list available models
        err_str = str(e)
        print(f"Warning: Failed to initialize Gemini model: {err_str}")
        try:
            # best-effort: list available models the client can see
            models = genai.list_models()
            print("Available models:")
            for m in models:
                try:
                    print(" -", getattr(m, 'name', m))
                except Exception:
                    print(" -", m)
        except Exception as list_err:
            print(f"Could not list models: {list_err}")
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
            return jsonify({"error": "AI model is not configured or failed to initialize. Check GEMINI_API_KEY and GEMINI_MODEL in your .env. See server logs for available models (if listed)."}), 503

        data = request.get_json()
        user_message = data.get("message")

        if not user_message:
            return jsonify({"error": "No message provided"}), 400
        
        # Send the user's message directly to the model
        try:
            response = gemini_model.generate_content(user_message)
        except Exception as gen_err:
            # Detect model unsupported errors and return a helpful message
            gen_err_str = str(gen_err)
            app.logger.error(f"Generative API error: {gen_err_str}")
            if 'not found' in gen_err_str or 'not supported' in gen_err_str or 'ListModels' in gen_err_str:
                return jsonify({"error": "Requested Gemini model is not available or not supported for generateContent. Check GEMINI_MODEL and your API permissions; see server logs for available models."}), 404
            raise

        # Check if the response was blocked or is empty
        if not getattr(response, 'parts', None):
            # Check for specific block reasons
            block_reason = getattr(response, 'prompt_feedback', None)
            block_reason_name = getattr(block_reason.block_reason, 'name', 'UNKNOWN') if block_reason else 'UNKNOWN'
            app.logger.warning(f"AI response was blocked. Reason: {block_reason_name}")
            return jsonify({"reply": "I'm sorry, I can't respond to that. Please ask a different question."}), 200
        return jsonify({"reply": getattr(response, 'text', '')})

    except Exception as e:
        # Detect common API key invalid message from google generative API and return clearer status
        err_str = str(e)
        print(f"Error in /ask_ai: {err_str}")
        if 'API key not valid' in err_str or 'API_KEY_INVALID' in err_str or 'api key' in err_str.lower():
            return jsonify({"error": "AI API key invalid or unauthorized. Check GEMINI_API_KEY in your .env."}), 401
        return jsonify({"error": err_str}), 500

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
