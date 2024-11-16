#!/usr/bin/env python3
import os
import sys
import openai
import string
import re
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from typing import List

FILENAME_VALID_CHARS = "-_.() %s%s" % (string.ascii_letters, string.digits)
GIT_DIFF_FILENAME_REGEX_PATTERN = r"\+\+\+ b/(.*)"
DEFAULT_OPENAI_MODEL = "gpt-4-turbo-preview"
DEFAULT_ANTHROPIC_MODEL = "claude-3-sonnet-20240229"
DEFAULT_STYLE = "concise"
DEFAULT_PERSONA = "kent_beck"
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 4096

OPENAI_ERROR_NO_RESPONSE = "No response from OpenAI. wtf Error:\n"
OPENAI_ERROR_FAILED = "OpenAI failed to generate a review. Error:\n"

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


class BaseLLM:
    """
    Base class for language learning models.
    """

    def __init__(self, model: str):
        self.model = model

    def prepare_kwargs(self, prompt: str, max_tokens: int, temperature: float) -> dict:
        """
        Prepares the keyword arguments for an LLM API call.
        To be implemented by subclasses.
        """
        raise NotImplementedError

    def call_api(self, kwargs: dict) -> str:
        """
        Calls the LLM API using the provided kwargs.
        To be implemented by subclasses.
        """
        raise NotImplementedError


class OpenAI_LLM(BaseLLM):
    """
    OpenAI LLM implementation using latest API.
    """

    def prepare_kwargs(self, prompt: str, max_tokens: int, temperature: float) -> dict:
        kwargs = {
            "model": self.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert code reviewer focused on improving code quality, security, and performance.",
                },
                {"role": "user", "content": prompt},
            ],
        }
        return kwargs

    def call_api(self, kwargs: dict) -> str:
        try:
            response = openai.chat.completions.create(**kwargs)
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI API call failed with parameters {kwargs}. Error: {e}")
            raise Exception(
                f"OpenAI API call failed with parameters {kwargs}. Error: {e}"
            )


class Anthropic_LLM(BaseLLM):
    """
    Anthropic LLM implementation using latest Claude API.
    """

    def __init__(self, model: str):
        super().__init__(model)
        self.anthropic = Anthropic()

    def prepare_kwargs(self, prompt: str, max_tokens: int, temperature: float) -> dict:
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        return kwargs

    def call_api(self, kwargs: dict) -> str:
        try:
            response = self.anthropic.messages.create(**kwargs)
            return response.content[0].text.strip()
        except Exception as e:
            print(f"Anthropic API call failed with parameters {kwargs}. Error: {e}")
            raise Exception(
                f"Anthropic API call failed with parameters {kwargs}. Error: {e}"
            )


def validate_filename(filename: str) -> bool:
    """
    Validates a filename by checking for directory traversal and unusual characters.

    Args:
      filename: str, filename to be validated

    Returns:
      bool: True if the filename is valid, False otherwise
    """
    return (
        not any(char in filename for char in set(filename) - set(FILENAME_VALID_CHARS))
        and ".." not in filename
        and "/" not in filename
    )


def extract_filenames_from_diff_text(diff_text: str) -> List[str]:
    """
    Extracts filenames from git diff text using regular expressions.

    Args:
      diff_text: str, git diff text

    Returns:
      List of filenames
    """
    filenames = re.findall(GIT_DIFF_FILENAME_REGEX_PATTERN, diff_text)
    sanitized_filenames = [fn for fn in filenames if validate_filename(fn)]
    return sanitized_filenames


def format_file_contents_as_markdown(filenames: List[str]) -> str:
    """
    Iteratively goes through each filename and concatenates
    the filename and its content in a specific markdown format.

    Args:
      filenames: List of filenames

    Returns:
      Formatted string
    """
    formatted_files = ""
    for filename in filenames:
        try:
            with open(filename, "r") as file:
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
    """
    Generates a prompt for use with an LLM

    Args:
      diff: str, the git diff text
      persona: str, the persona to use for the feedback
      style: str, the style of the feedback
      include_files: bool, whether to include file contents in the prompt
      filenames: List[str], optional list of filenames to include in the prompt

    Returns:
      str: The generated prompt
    """

    prompt = f"{persona}.{style}.{REQUEST}\n{diff}"

    # Optionally include files from the diff
    if include_files:
        if filenames is None:
            filenames = extract_filenames_from_diff_text(diff)
        if filenames:
            formatted_files = format_file_contents_as_markdown(filenames)
            prompt += formatted_files

    return prompt


def main():
    # Get environment variables
    api_to_use = os.environ.get(
        "API_TO_USE", "openai"
    )  # Default to OpenAI if not specified
    persona = PERSONAS.get(os.environ.get("PERSONA", DEFAULT_PERSONA))
    style = STYLES.get(os.environ.get("STYLE", DEFAULT_STYLE))
    include_files = os.environ.get("INCLUDE_FILES", "false") == "true"
    model = os.environ.get(
        "MODEL",
        DEFAULT_OPENAI_MODEL if api_to_use == "openai" else DEFAULT_ANTHROPIC_MODEL,
    )
    api_key_env_var = API_KEYS.get(api_to_use)

    # Make sure the necessary environment variable is set
    if api_key_env_var is None or api_key_env_var not in os.environ:
        print(f"The {api_key_env_var} environment variable is not set.")
        sys.exit(1)

    # Read in the diff
    diff = sys.stdin.read()

    # Generate the prompt
    prompt = get_prompt(diff, persona, style, include_files)

    # Instantiate the appropriate LLM class
    if api_to_use == "openai":
        llm = OpenAI_LLM(model)
        openai.api_key = os.environ[api_key_env_var]  # Set the API key for OpenAI
    elif api_to_use == "anthropic":
        llm = Anthropic_LLM(model)
    else:
        raise ValueError(
            f"Invalid API: {api_to_use}. Expected one of ['openai', 'anthropic']."
        )

    # Prepare kwargs for the API call
    kwargs = llm.prepare_kwargs(prompt, LLM_MAX_TOKENS, LLM_TEMPERATURE)

    # Call the API and print the review text
    review_text = llm.call_api(kwargs)

    print(f"{review_text}")


if __name__ == "__main__":
    main()
