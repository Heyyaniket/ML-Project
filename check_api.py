# First, run: pip install google-genai
from google import genai

# Replace with your ACTUAL key
MY_API_KEY = "AIzaSyA84qAtJO_8FargXm_Z0kJp-jA1i-3H2pQ"

client = genai.Client(api_key=MY_API_KEY)

try:
    print("Testing connection to Gemini 2026...")
    # Using gemini-2.0-flash (Stable 2026 model)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents="Is the Gemini API active? Answer in 5 words."
    )
    print(f"✅ SUCCESS! AI says: {response.text}")

except Exception as e:
    print("❌ API FAILED")
    print(f"Error Details: {e}")