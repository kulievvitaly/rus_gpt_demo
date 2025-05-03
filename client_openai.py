import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Get API key from environment
api_key = os.getenv("OPENAI_KEY")
if not api_key:
    raise ValueError("OPENAI_KEY not found in .env file")

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# Make request to the model
try:
    response = client.chat.completions.create(
        model="gpt-4.1",  # O3 model
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain quantum computing in simple terms."}
        ],
        max_tokens=500
    )

    # Print the response
    print(response.choices[0].message.content)

except Exception as e:
    print(f"Error: {e}")
