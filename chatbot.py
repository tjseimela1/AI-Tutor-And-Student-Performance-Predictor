
import os
from dotenv import load_dotenv
from groq import Groq

# Load the API key from the .env file
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "GROQ_API_KEY is missing. Check your .env file."
    )

# Connect to Groq
client = Groq(api_key=api_key)

# Instructions for the AI tutor
SYSTEM_INSTRUCTION = """
You are an AI tutor for students.

Explain topics in simple English.
Answer general questions and school-related questions.
Give examples when helpful.
Be patient, friendly, and encouraging.
Help students understand instead of just memorising.
If you are unsure, say so honestly.
"""


def start_chatbot():
    # Store the conversation history
    messages = [
        {
            "role": "system",
            "content": SYSTEM_INSTRUCTION
        }
    ]

    print("=" * 40)
    print("          AI TUTOR CHATBOT")
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

        # Add the student's question
        messages.append({
            "role": "user",
            "content": question
        })

        try:
            response = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=messages
            )

            answer = response.choices[0].message.content

            if answer:
                print("\nAI Tutor:", answer)

                # Save the answer for conversation context
                messages.append({
                    "role": "assistant",
                    "content": answer
                })
            else:
                print("AI Tutor: No answer was returned.")
                messages.pop()

        except Exception as error:
            # Remove the unanswered question
            messages.pop()

            print("\nAI Tutor: Sorry, something went wrong.")
            print("Technical details:", error)


if __name__ == "__main__":
    start_chatbot()