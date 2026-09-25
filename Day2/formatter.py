def format_summary(title, summary):
    """
    Format Wikipedia summary for terminal display.
    """

    line = "=" * 70

    output = f"""
{line}
                    WIKIPEDIA SUMMARY
{line}

Topic: {title}


Summary:
{summary}

{line}
"""

    return output


def format_error(message):
    """
    Format error message.
    """

    return f"""
[ERROR] {message}
"""