import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os

# Load the dataset
df = pd.read_csv("datasets/symtoms_df.csv")

# Clean column names by stripping whitespace
df.columns = df.columns.str.strip()

# Define symptom columns
symptom_cols = [col for col in df.columns if 'Symptom' in col]

# Clean the data by stripping whitespace from all symptom entries
for col in symptom_cols:
    df[col] = df[col].str.strip().str.lower().str.replace('_', ' ')

# Combine all symptoms into a single list
all_symptoms = set()
for col in symptom_cols:
    all_symptoms.update(df[col].dropna().unique())

# Create a dictionary to map symptoms to indices
symptoms_dict = {symptom: idx for idx, symptom in enumerate(sorted(all_symptoms))}

# Create input vectors for symptoms
def create_input_vector(row):
    vector = np.zeros(len(symptoms_dict))
    for col in symptom_cols:
        symptom = row[col]
        if pd.notna(symptom) and symptom in symptoms_dict:
            vector[symptoms_dict[symptom]] = 1
    return vector

X = np.array(df.apply(create_input_vector, axis=1).tolist())
y = df['Disease']

# Encode the target labels (diseases)
diseases_list = {disease: idx for idx, disease in enumerate(y.unique())}
reverse_diseases_list = {idx: disease for disease, idx in diseases_list.items()}
y = y.map(diseases_list)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
svc = SVC(kernel='linear', probability=True, random_state=42)
svc.fit(X_train, y_train)

# Evaluate the model
y_pred = svc.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=reverse_diseases_list.values()))

# --- Save the model and dictionaries ---

# Create Model directory if it doesn't exist
MODEL_DIR = 'Model'
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
    print(f"Created directory: {MODEL_DIR}")

# Save the trained model
with open(os.path.join(MODEL_DIR, 'svc.pkl'), 'wb') as model_file:
    pickle.dump(svc, model_file)

# Save the symptoms dictionary and diseases list
with open(os.path.join(MODEL_DIR, 'symptoms_dict.pkl'), 'wb') as symptoms_file:
    pickle.dump(symptoms_dict, symptoms_file)

with open(os.path.join(MODEL_DIR, 'diseases_list.pkl'), 'wb') as diseases_file:
    pickle.dump(reverse_diseases_list, diseases_file)

print("Model and helper files saved successfully.")