import requests


WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"

HEADERS = {
    "User-Agent": "WikipediaCLI/1.0 (learning project)"
}


def search_topic(topic):
    """
    Search Wikipedia for a topic.

    Returns:
        str: First matching Wikipedia page title
        None: If no result is found
    """

    params = {
        "action": "query",
        "list": "search",
        "srsearch": topic,
        "format": "json"
    }

    response = requests.get(
        WIKIPEDIA_API_URL,
        params=params,
        headers=HEADERS,
        timeout=10
    )

    response.raise_for_status()

    data = response.json()

    search_results = data["query"]["search"]

    if not search_results:
        return None

    return search_results[0]["title"]


def get_summary(title):
    """
    Fetch the summary of a Wikipedia page.

    Returns:
        str: Wikipedia summary
        None: If summary is unavailable
    """

    params = {
        "action": "query",
        "prop": "extracts",
        "exintro": True,
        "explaintext": True,
        "titles": title,
        "format": "json"
    }

    response = requests.get(
        WIKIPEDIA_API_URL,
        params=params,
        headers=HEADERS,
        timeout=10
    )

    response.raise_for_status()

    data = response.json()

    pages = data["query"]["pages"]

    page = next(iter(pages.values()))

    if "extract" not in page:
        return None

    return page["extract"]