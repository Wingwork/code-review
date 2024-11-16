#!/usr/bin/env python3

import argparse
import os
import subprocess
from typing import List, Optional
import sys

DEFAULT_PERSONA = "kent_beck"
DEFAULT_STYLE = "concise"
DEFAULT_MODEL = "gpt-4-turbo-preview"
DEFAULT_ANTHROPIC_MODEL = "claude-3-sonnet-20240229"
DEFAULT_BRANCH = "main"


def get_file(filename: str) -> Optional[str]:
    """
    Get the contents of the specified file.

    Args:
        filename: Path to the file

    Returns:
        Optional[str]: File contents formatted in markdown or None if file cannot be read
    """
    try:
        with open(filename, "r", encoding="utf-8") as file:
            file_str = file.read()
            return f"\n{filename}\n```\n{file_str}\n```"
    except FileNotFoundError:
        print(f"WARNING: File {filename} not found")
        return None
    except UnicodeDecodeError:
        print(f"WARNING: File {filename} could not be read due to encoding issues")
        return None
    except Exception as e:
        print(f"WARNING: Unexpected error reading {filename}: {str(e)}")
        return None


def construct_git_diff_command(branch: str, exclude_files: List[str]) -> str:
    """
    Construct the git diff command with proper escaping.

    Args:
        branch: Branch to compare against
        exclude_files: List of files to exclude

    Returns:
        str: Constructed git diff command
    """
    exclude_str = (
        " ".join(f"':!{file.strip()}'" for file in exclude_files)
        if exclude_files
        else ""
    )
    return f"git diff --merge-base origin/{branch} HEAD -- . {exclude_str}".strip()


def execute_command(command: str) -> str:
    """
    Execute a shell command with proper error handling.

    Args:
        command: Shell command to execute

    Returns:
        str: Command output or error message
    """
    try:
        result = subprocess.run(
            command, shell=True, check=True, capture_output=True, text=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        error_msg = f"Error executing command: {str(e)}\nOutput: {e.output}"
        print(error_msg, file=sys.stderr)
        return error_msg


def get_diff(
    diff_file: Optional[str] = None,
    branch: str = "main",
    exclude_files: Optional[List[str]] = None,
) -> str:
    """
    Get the diff either from git or a file.

    Args:
        diff_file: Optional path to diff file
        branch: Branch to compare against
        exclude_files: List of files to exclude

    Returns:
        str: Diff content
    """
    if diff_file:
        content = get_file(diff_file)
        return content if content else ""
    else:
        return get_diff_from_git(branch, exclude_files or [])


def main():
    parser = argparse.ArgumentParser(
        description="AI-assisted code reviews with multiple model support"
    )
    parser.add_argument(
        "--persona",
        default=DEFAULT_PERSONA,
        choices=["developer", "kent_beck", "marc_benioff", "yoda"],
        help="Review persona to use",
    )
    parser.add_argument(
        "--style",
        default=DEFAULT_STYLE,
        choices=["concise", "zen"],
        help="Output style",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="AI model to use")
    parser.add_argument(
        "--api-to-use",
        default="openai",
        choices=["openai", "anthropic"],
        help="AI API provider",
    )
    parser.add_argument(
        "--branch", default=DEFAULT_BRANCH, help="Branch to compare against"
    )
    parser.add_argument("--filename", help="Optional diff file to use")
    parser.add_argument("--directory", help="Optional directory to review")
    parser.add_argument(
        "--include-files",
        default="false",
        choices=["true", "false"],
        help="Include full file contents",
    )
    parser.add_argument(
        "--exclude-files", default="", help="Comma-separated list of files to exclude"
    )

    args = parser.parse_args()

    # Set default model based on API choice
    if not args.model:
        args.model = (
            DEFAULT_MODEL if args.api_to_use == "openai" else DEFAULT_ANTHROPIC_MODEL
        )

    # Parse exclude files
    exclude_files = (
        [f.strip() for f in args.exclude_files.split(",")]
        if args.exclude_files
        else None
    )

    # Get diff content
    if args.directory:
        if not os.path.isdir(args.directory):
            print(f"Error: Directory {args.directory} does not exist", file=sys.stderr)
            sys.exit(1)
        diff = ""
        for root, _, files in os.walk(args.directory):
            for file in files:
                if exclude_files and file in exclude_files:
                    continue
                file_path = os.path.join(root, file)
                content = get_file(file_path)
                if content:
                    diff += content
    else:
        diff = get_diff(args.filename, args.branch, exclude_files)

    if not diff:
        print("No changes to review", file=sys.stderr)
        sys.exit(1)

    # Set environment variables for action_code_review.py
    os.environ.update(
        {
            "MODEL": args.model,
            "PERSONA": args.persona,
            "STYLE": args.style,
            "INCLUDE_FILES": args.include_files,
            "API_TO_USE": args.api_to_use,
        }
    )

    # Run the review
    script_dir = os.path.dirname(os.path.realpath(__file__))
    review_script = os.path.join(script_dir, "action_code_review.py")

    try:
        result = subprocess.run(
            [sys.executable, review_script],
            input=diff,
            text=True,
            capture_output=True,
            check=True,
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(
            f"Error running code review: {str(e)}\nOutput: {e.stderr}", file=sys.stderr
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
