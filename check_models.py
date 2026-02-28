import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
target_model = os.getenv("GEMINI_MODEL", "models/gemini-1.5-flash")

print("--- Gemini API Debugger ---")

if not api_key:
    print("❌ Error: GEMINI_API_KEY not found. Check your .env file.")
    exit()
else:
    print(f"✅ API Key loaded (starts with: {api_key[:6]}...)")

genai.configure(api_key=api_key)

print("\n--- Available Models for this Key ---")
try:
    available_models = []
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f" - {m.name}")
            available_models.append(m.name)
    
    print(f"\nTarget Model from .env: '{target_model}'")
    
    if target_model in available_models:
        print(f"✅ The model '{target_model}' is valid and available.")
        print("Attempting test generation...")
        model = genai.GenerativeModel(target_model)
        response = model.generate_content("Hello, this is a test.")
        print(f"Response received: {response.text}")
    else:
        print(f"❌ The model '{target_model}' is NOT in the available list.")
        print("   Please copy one of the available model names above into your .env file.")

except Exception as e:
    print(f"❌ API Error: {e}")