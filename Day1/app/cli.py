from app.prompt_builder import build_prompt
from app.llm import generate_response
from app.formatter import (
    print_header,
    print_response,
    print_error
)


def run_playground():

    print_header("PROMPT PLAYGROUND")

    while True:

        topic = input(
            "\nEnter topic (or type 'exit' to quit): "
        ).strip()

        if topic.lower() == "exit":
            print("\nThank you for using Prompt Playground!")
            break

        if not topic:
            print_error("Topic cannot be empty.")
            continue

        try:

            print("\nGenerating response...")
            print("Please wait...\n")

            prompt = build_prompt(topic)

            response = generate_response(prompt)

            print_response(response)

        except Exception as error:

            error_message = str(error)

            if "429" in error_message or "RESOURCE_EXHAUSTED" in error_message:

                print_error(
                    "Gemini API quota has been exhausted.\n\n"
                    "Try again after the quota resets or use a "
                    "different model/API project."
                )

            else:

                print_error(error_message)