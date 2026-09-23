def print_header(title: str) -> None:
    print("\n" + "=" * 70)
    print(title.center(70))
    print("=" * 70)


def print_section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title.center(70))
    print("=" * 70)


def print_response(response: str) -> None:

    print("\n")

    sections = [
        "SUMMARY",
        "EXPLAIN SIMPLY",
        "IMPORTANT POINTS",
        "QUIZ",
        "COMPARISON"
    ]

    current_section = None

    for line in response.splitlines():

        cleaned_line = line.strip()

        if not cleaned_line:
            print()
            continue

        upper_line = cleaned_line.upper()

        # Detect section headings
        if upper_line in sections:

            current_section = upper_line

            print_section(current_section)

            continue

        print(line)

    print("\n" + "=" * 70)
    print("END OF RESPONSE".center(70))
    print("=" * 70)


def print_error(message: str) -> None:

    print("\n" + "!" * 70)
    print("ERROR".center(70))
    print("!" * 70)

    print(message)

    print("!" * 70)