#!/usr/bin/env python3
import os
import sys
import string
import re
from typing import List
from litellm import completion
from litellm.exceptions import (
    BadRequestError,
    RateLimitError,
    AuthenticationError,
    ContextLengthExceededError,
)

FILENAME_VALID_CHARS = "-_.() %s%s" % (string.ascii_letters, string.digits)
GIT_DIFF_FILENAME_REGEX_PATTERN = r"\+\+\+ b/(.*)"

# Model configurations
DEFAULT_OPENAI_MODEL = "gpt-4-turbo-preview"
DEFAULT_ANTHROPIC_MODEL = "claude-3-sonnet-20240229"
DEFAULT_STYLE = "concise"
DEFAULT_PERSONA = "kent_beck"
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 4096

# API configurations
SUPPORTED_MODELS = {
    "openai": [
        "gpt-4-turbo-preview",
        "gpt-4-0125-preview",
        "gpt-4-1106-preview",
        "gpt-4",
        "gpt-4-32k",
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-16k",
    ],
    "anthropic": [
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
    ],
}

API_KEYS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}

REQUEST = """
Analyze the code changes and provide a comprehensive code review focusing on:

1. Code Quality & Best Practices
2. Performance Improvements 
3. Security Considerations
4. Potential Bugs or Edge Cases

For each suggestion:
- Explain the issue clearly
- Provide a specific code example showing the improvement
- Explain why this change would be beneficial

Format your response in markdown with clear sections and code blocks.
Limit to 3 most important suggestions.
"""

STYLES = {
    "zen": "Format feedback in a thoughtful, mindful way focusing on fundamental improvements",
    "concise": """Format feedback as a clear, numbered list with:
- Issue description
- Code example
- Benefits
""",
}

PERSONAS = {
    "developer": """You are a senior software architect with expertise in:
- Clean code principles
- Security best practices  
- Performance optimization
- System design patterns""",
    "kent_beck": """You are Kent Beck, focusing on:
- Simple design
- Test-driven development
- Refactoring patterns
- Code maintainability""",
    "marc_benioff": """You are Marc Benioff, emphasizing:
- Enterprise scalability
- Code reliability
- Team collaboration
- Business value""",
    "yoda": """You are Yoda, the Jedi Master of Code. Focus on:
- Wisdom in simplicity
- Balance in design
- Learning from mistakes
- Speak in Yoda's style""",
}


class LiteLLMWrapper:
    """
    Wrapper for LiteLLM to handle both OpenAI and Anthropic models.
    """

    def __init__(self, model: str, api_type: str):
        self.model = model
        self.api_type = api_type
        self.validate_model()

    def validate_model(self) -> None:
        """
        Validates if the model is supported for the given API type.
        """
        if self.model not in SUPPORTED_MODELS.get(self.api_type, []):
            supported = SUPPORTED_MODELS.get(self.api_type, [])
            raise ValueError(
                f"Model '{self.model}' is not supported for {self.api_type}. "
                f"Supported models are: {sorted(supported)}"
            )

    def get_completion(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """
        Gets completion from LiteLLM with unified error handling.
        """
        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert code reviewer focused on improving code quality, security, and performance.",
                },
                {"role": "user", "content": prompt},
            ]

            response = completion(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            return response.choices[0].message.content.strip()

        except AuthenticationError:
            print(
                f"Authentication failed for {self.api_type}. Please check your API key."
            )
            raise
        except RateLimitError:
            print(f"Rate limit exceeded for {self.api_type}.")
            raise
        except ContextLengthExceededError:
            print(f"Input too long for {self.model}. Try reducing the input size.")
            raise
        except BadRequestError as e:
            print(f"Invalid request: {str(e)}")
            raise
        except Exception as e:
            print(f"Unexpected error with {self.api_type}: {str(e)}")
            raise


def validate_filename(filename: str) -> bool:
    """Validates a filename by checking for directory traversal and unusual characters."""
    return (
        not any(char in filename for char in set(filename) - set(FILENAME_VALID_CHARS))
        and ".." not in filename
        and "/" not in filename
    )


def extract_filenames_from_diff_text(diff_text: str) -> List[str]:
    """Extracts filenames from git diff text using regular expressions."""
    filenames = re.findall(GIT_DIFF_FILENAME_REGEX_PATTERN, diff_text)
    return [fn for fn in filenames if validate_filename(fn)]


def format_file_contents_as_markdown(filenames: List[str]) -> str:
    """Formats file contents as markdown."""
    formatted_files = ""
    for filename in filenames:
        try:
            with open(filename, "r", encoding="utf-8") as file:
                file_content = file.read()
            formatted_files += f"\n{filename}\n```\n{file_content}\n```\n"
        except Exception as e:
            print(f"Could not read file {filename}: {e}")
    return formatted_files


def get_prompt(
    diff: str,
    persona: str,
    style: str,
    include_files: bool,
    filenames: List[str] = None,
) -> str:
    """Generates a prompt for the LLM."""
    prompt = f"{persona}.{style}.{REQUEST}\n{diff}"

    if include_files and filenames is None:
        filenames = extract_filenames_from_diff_text(diff)
    if include_files and filenames:
        prompt += format_file_contents_as_markdown(filenames)

    return prompt


def main():
    # Get environment variables
    api_to_use = os.environ.get("API_TO_USE", "openai")
    persona = PERSONAS.get(os.environ.get("PERSONA", DEFAULT_PERSONA))
    style = STYLES.get(os.environ.get("STYLE", DEFAULT_STYLE))
    include_files = os.environ.get("INCLUDE_FILES", "false") == "true"
    model = os.environ.get(
        "MODEL",
        DEFAULT_OPENAI_MODEL if api_to_use == "openai" else DEFAULT_ANTHROPIC_MODEL,
    )

    # Validate API key
    api_key_env_var = API_KEYS.get(api_to_use)
    if api_key_env_var is None or api_key_env_var not in os.environ:
        print(f"The {api_key_env_var} environment variable is not set.")
        sys.exit(1)

    # Read diff
    diff = sys.stdin.read()
    if not diff.strip():
        print("No diff content provided.")
        sys.exit(1)

    try:
        # Initialize LiteLLM wrapper
        llm = LiteLLMWrapper(model=model, api_type=api_to_use)

        # Generate and get completion
        prompt = get_prompt(diff, persona, style, include_files)
        review_text = llm.get_completion(prompt, LLM_MAX_TOKENS, LLM_TEMPERATURE)

        print(review_text)

    except (ValueError, AuthenticationError, RateLimitError) as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
