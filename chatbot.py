
from google import genai
from dotenv import load_dotenv
import os

# Load the API key from the .env file
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is missing from your .env file.")

# Connect to API
client = genai.Client(api_key=api_key)

# Instructions for our AI tutor
SYSTEM_INSTRUCTION = """
You are an AI tutor for students.
Explain topics in simple English.
Give examples when helpful.
Be patient, friendly, and encouraging.
Help students understand instead of just memorising.
"""


def start_chatbot():
    # Start a conversation with the API
    chat = client.chats.create(
        model="gemini-3.8-flash",
        config={
            "system_instruction": SYSTEM_INSTRUCTION
        }
    )

    print("=" * 40)
    print("       AI TUTOR CHATBOT")
    print("=" * 40)
    print("Ask a question to start learning.")
    print("Type 'exit' to stop the chatbot.")

    while True:
        question = input("\nYou: ").strip()

        if question.lower() == "exit":
            print("AI Tutor: Goodbye! Keep learning.")
            break

        if not question:
            print("AI Tutor: Please enter a question.")
            continue

        try:
            response = chat.send_message(message=question)
            print("\nAI Tutor:", response.text)

        except Exception as error:
            print("\nSorry, something went wrong.")
            print("Error:", error)


if __name__ == "__main__":
    start_chatbot()
