import requests
import json

def check_api_key(api_key):
    """Check if the Gemini API key is valid by making a test request."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{"parts": [{"text": "Hello"}]}]
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        print("✅ API Key is valid!")
        return True
    else:
        print(f"❌ Invalid API Key: {response.json()}")
        return False

def chat_with_gemini():
    """Simple terminal-based chat with Gemini API."""
    api_key = input("Enter your Gemini API Key: ")
    if not check_api_key(api_key):
        return

    print("\n💬 Chatbot is ready! Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("👋 Goodbye!")
            break
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        data = {"contents": [{"parts": [{"text": user_input}]}]}

        response = requests.post(url, headers=headers, data=json.dumps(data))

        if response.status_code == 200:
            reply = response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "No response")
            print("Gemini:", reply)
        else:
            print("⚠️ Error:", response.json())
            break

if __name__ == "__main__":
    chat_with_gemini()
