import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import pickle

# Load the dataset
data = pd.read_csv("datasets/symtoms_df.csv", header=None)
data.columns = ['Index', 'Disease', 'Symptom1', 'Symptom2', 'Symptom3', 'Symptom4', 'Symptom5']

# Combine all symptoms into a single list
all_symptoms = set()
for col in ['Symptom1', 'Symptom2', 'Symptom3', 'Symptom4', 'Symptom5']:
    all_symptoms.update(data[col].dropna().unique())

# Create a dictionary to map symptoms to indices
symptoms_dict = {symptom: idx for idx, symptom in enumerate(sorted(all_symptoms))}

# Create input vectors for symptoms
def create_input_vector(row):
    vector = np.zeros(len(symptoms_dict))
    for col in ['Symptom1', 'Symptom2', 'Symptom3', 'Symptom4', 'Symptom5']:
        symptom = row[col]
        if pd.notna(symptom) and symptom in symptoms_dict:
            vector[symptoms_dict[symptom]] = 1
    return vector

X = np.array(data.apply(create_input_vector, axis=1).tolist())
y = data['Disease']

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

# Save the trained model
with open('Model/svc.pkl', 'wb') as model_file:
    pickle.dump(svc, model_file)

# Save the symptoms dictionary and diseases list
with open('Model/symptoms_dict.pkl', 'wb') as symptoms_file:
    pickle.dump(symptoms_dict, symptoms_file)

with open('Model/diseases_list.pkl', 'wb') as diseases_file:
    pickle.dump(reverse_diseases_list, diseases_file)