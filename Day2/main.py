from wikipedia_api import search_topic, get_summary
from formatter import format_summary, format_error


def main():
    """
    Main application function.
    """

    print("=" * 70)
    print("                    WIKIPEDIA CLI TOOL")
    print("=" * 70)

    # Take input from user
    topic = input("\nEnter a topic: ").strip()

    # Validate input
    if not topic:
        print(format_error("Topic cannot be empty."))
        return

    try:

        # Search Wikipedia
        print("\nSearching Wikipedia...")

        title = search_topic(topic)

        if title is None:
            print(format_error("No Wikipedia page found for this topic."))
            return

        print(f"Found page: {title}")

        # Fetch summary
        print("\nFetching summary...")

        summary = get_summary(title)

        if summary is None:
            print(format_error("Summary is not available."))
            return

        # Display formatted result
        print(format_summary(title, summary))

    except Exception as error:

        print(
            format_error(
                f"Something went wrong: {error}"
            )
        )


if __name__ == "__main__":
    main()