
import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq, APIConnectionError, AuthenticationError, RateLimitError

# Load API key from the project .env file
load_dotenv(Path(__file__).with_name(".env"))

API_KEY = os.getenv("GROQ_API_KEY")

# Connect to Groq
client = Groq(api_key=API_KEY) if API_KEY else None

# AI assistant instructions
SYSTEM_PROMPT = """
You are a helpful, intelligent and knowledgeable AI assistant.

Your job is to answer questions on a wide range of topics, including:

- Education and school subjects
- Mathematics and science
- Information Technology and programming
- History and geography
- General knowledge
- Technology and artificial intelligence
- Writing and communication
- Everyday questions and problem solving

RULES:

1. Answer questions clearly and accurately.
2. Use simple English when explaining difficult topics.
3. Give step-by-step explanations when necessary.
4. Provide examples where they help understanding.
5. Do not restrict yourself to school-related questions.
6. Do not invent facts when you are uncertain.
7. If a question is unclear, ask for clarification.
8. Be friendly, patient and respectful.
9. Keep simple answers short and give detailed answers when needed.
10. Help students understand concepts instead of simply memorising answers.

You are a general-purpose AI assistant, not just a school chatbot.
"""


def answer(question, history=None):

    question = (question or "").strip()

    if not question:
        return {
            "answer": "Please enter a question.",
            "topic": "Guidance",
            "resource": None,
            "matched": False
        }

    if client is None:
        return {
            "answer": "The AI service is not configured. Please check your GROQ_API_KEY in the .env file.",
            "topic": "System",
            "resource": None,
            "matched": False
        }

    try:
        # Start with the assistant instructions
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

        # Add the previous conversation
        if history:
            for message in history[-10:]:
                role = message.get("role")
                content = message.get("content", "")

                if role in ("user", "assistant") and content:
                    messages.append({
                        "role": role,
                        "content": str(content)[:3000]
                    })

        # Add the current question
        messages.append({
            "role": "user",
            "content": question
        })

        # Send the conversation to Groq
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages,
            max_completion_tokens=1024
        )

        ai_answer = response.choices[0].message.content

        return {
            "answer": ai_answer or "I couldn't generate an answer. Please try again.",
            "topic": "AI Assistant",
            "resource": None,
            "matched": True
        }

    except AuthenticationError:
        message = "API authentication failed. Please check your Groq API key."

    except RateLimitError:
        message = "The AI service has reached its current usage limit. Please try again later."

    except APIConnectionError:
        message = "Unable to connect to the AI service. Check your internet connection."

    except Exception:
        message = "Something went wrong while generating your answer. Please try again."

    return {
        "answer": message,
        "topic": "System",
        "resource": None,
        "matched": False
    }
