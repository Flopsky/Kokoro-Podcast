import instructor
from instructor.exceptions import InstructorRetryException
import google.generativeai as genai
import os
from jinja2 import Environment, FileSystemLoader, select_autoescape
from typing import Optional
from dotenv import load_dotenv

# Assuming schemas are in the parent directory src/
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the parent directory (project root) to sys.path
project_root = Path(__file__).resolve().parent.parent

sys.path.append(str(project_root))

from src.schemas.poadcast import Poadcast # Import the Pydantic model

# --- Configuration ---
load_dotenv() # Load environment variables from .env file
# Configure the Gemini API key
# Ensure GOOGLE_API_KEY is set in your environment or .env file
if not os.getenv("GEMINI_API_KEY"):
    raise ValueError("GEMINI_API_KEY environment variable not set.")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

SCRIPTER_MODEL = os.getenv("SCRIPTER_MODEL")

# Configure Jinja
prompt_dir = Path(__file__).parent / "prompts"
env = Environment(
    loader=FileSystemLoader(prompt_dir),
    autoescape=select_autoescape(['html', 'xml']) # Though not strictly needed for text, it's good practice
)

# --- Instructor Client Setup ---
# Patch the genai client with instructor
# Use GEMINI_JSON mode as we want direct JSON output based on the system prompt
# Use gemini-1.5-flash-latest as requested (closest available to 2.0-flash)
safe = [
        {"category": "HARM_CATEGORY_DANGEROUS", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]
client = instructor.from_gemini(
    client=genai.GenerativeModel(
        model_name=SCRIPTER_MODEL,
        safety_settings=safe
    ),
    mode=instructor.Mode.GEMINI_JSON, # Force JSON output mode
)

# --- Core Functions ---

def call_gemini_with_instructor(
    system_prompt: str,
    user_prompt: str,
    max_retries: int = 10,
    temperature: float = 1.0,
    top_p: float = 1.0,
    candidate_count: int = 1,
    max_output_tokens: int = 8000
) -> Poadcast:
    """
    Calls the Gemini model via instructor to get a structured response.

    Args:
        system_prompt: The system prompt defining the AI's role and task.
        user_prompt: The user prompt containing the specific request (e.g., the source text).
        max_retries: Number of retries if validation fails.

    Returns:
        A Poadcast object validated by Pydantic.

    Raises:
        Exception: If the API call fails after retries.
    """
    try:
        logger.info("Calling Gemini API with instructor...")
        response = client.chat.create_with_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            generation_config={
                        "temperature": temperature,
                        "top_p": top_p,
                        "candidate_count": candidate_count,
                        "max_output_tokens": max_output_tokens,
                    },
            response_model=Poadcast,
            max_retries=max_retries,
        )
        logger.info("Gemini API call successful.")
        return response[0]
    except Exception as e:
        logger.error(f"Error calling Gemini API with instructor after {max_retries} retries: {e}")
        if isinstance(e, InstructorRetryException):
            logger.info("last_completion: ", e.last_completion)
            return e.last_completion
            # Handle retry logic here, potentially with exponential backoff

        # You can log the error or perform other error handling steps here.
        # For example, you might want to log the error to a file or send an alert.
        # Depending on requirements, you might re-raise, return None, or a default object


def generate_script(source_text: str) -> Optional[str]:
    """
    Generates a formatted podcast script string from source text using Gemini.

    Args:
        source_text: The raw text to be converted into a podcast script.

    Returns:
        A formatted string containing the podcast script, or None if generation fails.
    """
    try:
        # 1. Load and render prompts
        system_template = env.get_template("system_prompt.jinja2")
        user_template = env.get_template("user_prompt.jinja2")

        system_prompt_rendered = system_template.render() 
        user_prompt_rendered = user_template.render(source_text=source_text)

        # 2. Call Gemini via Instructor
        logger.info("Generating podcast script...")
        poadcast_data = call_gemini_with_instructor(
            system_prompt=system_prompt_rendered,
            user_prompt=user_prompt_rendered,
            temperature=0.0,
            top_p=1.0,
            candidate_count=1,
            max_output_tokens=8000
        )

        # 3. Format the output text
        script_parts = []
        script_parts.append(f"{poadcast_data.title}\n") # Use markdown H1 for title
        if poadcast_data.summary:
            script_parts.append(f"{poadcast_data.summary}\n") # Add summary if present
        for i, segment in enumerate(poadcast_data.script_segments):
            if segment.segment_title:
                script_parts.append(f"{segment.segment_title}\n") # Use markdown H2 for segment title
            for paragraph in segment.paragraphs:
                script_parts.append(paragraph.text ) # Add each paragraph text followed by newline

        logger.info("Script formatted.")
        return "\n".join(script_parts)

    except Exception as e:
        logger.error(f"Failed to generate script: {e}")
        return None

# --- Example Usage ---
# if __name__ == "__main__":
#     example_source_text = """
#     The history of Python dates back to the late 1980s when Guido van Rossum began working on it.
#     He aimed to create a language that was easy to read and write, emphasizing code readability.
#     Python 1.0 was released in 1994, featuring core data types like lists and dictionaries.
#     Its object-oriented features were already present.
#     Python 2.0, released in 2000, introduced list comprehensions and garbage collection.
#     It became hugely popular but introduced some backward incompatibilities.
#     Python 3.0, released in 2008, was a major revision designed to rectify fundamental design flaws.
#     It was intentionally not backward compatible with Python 2. This caused a lengthy migration period.
#     Key changes included Unicode strings by default and the print function.
#     Today, Python 3 is the standard, widely used in web development, data science, AI, and scripting.
#     Its large standard library and active community contribute to its success. Guido van Rossum is often called the BDFL (Benevolent Dictator For Life), though he stepped down from that role in 2018.
#     Readability counts. Explicit is better than implicit. Simple is better than complex.
#     These are parts of the Zen of Python.
#     """

#     generated_script_text = generate_script(example_source_text)

#     if generated_script_text:
#         print("\n--- Generated Podcast Script ---")
#         print(generated_script_text)
#     else:
#         print("\nScript generation failed.")
