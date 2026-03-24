from google import genai

# Use your NEW API key here
client = genai.Client(api_key="AIzaSyA84qAtJO_8FargXm_Z0kJp-jA1i-3H2pQ")

try:
    print("Connecting to 2026 Diagnostic Engine...")
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Check API status. Reply with: 'System Live'"
    )
    print(f"✅ {response.text}")
except Exception as e:
    print(f"❌ Connection Failed: {e}")