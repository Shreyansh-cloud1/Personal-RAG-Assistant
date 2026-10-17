import os
from dotenv import load_dotenv
from google import genai

# Load environment variables from the .env file
load_dotenv()

# Read the API key from the environment
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Validate that the API key exists
if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY is not set. Please add it to your .env file."
    )

# Read the model name from the environment.
# If GEMINI_MODEL is not set, use "gemini-2.5-flash".
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Create a Gemini client using the new google.genai package
client = genai.Client(api_key=GOOGLE_API_KEY)


def generate_answer(context: str, question: str) -> str:
    """
    Generate an answer using the retrieved context and the user's question.

    Parameters:
        context (str): Retrieved document chunks from the vector database.
        question (str): The user's question.

    Returns:
        str: The generated answer.
    """

    # Build the prompt sent to Gemini
    prompt = f"""
        You are a document QA assistant.

        Your job is to answer questions strictly from the provided context.

        If the answer is not explicitly stated in the context, respond exactly:

        I don't know based on the provided documents.

        Do not:
        - use external knowledge
        - make assumptions
        - provide related information
        - explain nearby topics

        Context:
        {context}

        Question:
        {question}
    """

    try:
        # Send the prompt to the Gemini model
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )

        # Validate that text was returned
        if not hasattr(response, "text") or not response.text:
            return "No response was generated."

        # Return the generated answer
        return response.text.strip()

    except Exception as e:
        # Return a readable error message instead of crashing the API
        return f"Error generating response: {str(e)}"