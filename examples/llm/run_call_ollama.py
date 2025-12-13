#!/bin/env python3
import argparse
import asyncio
from pathlib import Path

from classconfig import Config
from pydantic import BaseModel

from topicer.llm import OllamaAsyncAPI

SCRIPT_DIR = Path(__file__).parent


class StructuredOutput(BaseModel):
    sentiment: str
    topic: str


async def call_run(args):
    """
    Method for running the LLM API example.

    Demonstrates how to process text chunks and obtain both plain and structured responses using the OllamaAsyncAPI.
    :param args: User arguments.
    """

    api = OllamaAsyncAPI.create(SCRIPT_DIR / "ollama_config.yaml")
    res = await api.process_text_chunks(
        text_chunks=["Why is sky blue?", "Explain quantum computing in simple terms.", "What is the capital of France?"],
        instruction="Provide concise answers.",
        model="gpt-oss:20b"
    )

    print("LLM Responses:")
    for idx, answer in enumerate(res):
        print(f"Q{idx + 1}: {answer}")

    # structured output example
    structured_res = await api.process_text_chunks_structured(
        text_chunks=["I love programming.", "I hate chocolate ice cream."],
        instruction="Analyze the sentiment and provide the main topic.",
        output_type=StructuredOutput,
        model="gpt-oss:20b"
    )
    print("\nStructured LLM Responses:")
    for idx, answer in enumerate(structured_res):
        print(f"Text {idx + 1}: Sentiment - {answer.sentiment}, Topic - {answer.topic}")


async def call_create_config(args):
    """
    Method for creating configuration for LLM API.

    :param args: User arguments.
    """
    # Placeholder for actual implementation
    c = Config(OllamaAsyncAPI)
    c.save(SCRIPT_DIR / "ollama_config.yaml")
    c.to_md(SCRIPT_DIR / "ollama_config.md")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Example")
    subparsers = parser.add_subparsers()

    run_parser = subparsers.add_parser('run', help="Run example")
    run_parser.set_defaults(func=call_run)

    create_config_parser = subparsers.add_parser('create_config', help="Create configuration")
    create_config_parser.set_defaults(func=call_create_config)

    return parser.parse_args()


async def main():
    args = parse_arguments()
    if hasattr(args, 'func'):
        await args.func(args)
    else:
        print("No command provided. Use -h for help.")


if __name__ == "__main__":
    asyncio.run(main())
