from app.prompt_templates import PROMPT_PLAYGROUND_TEMPLATE


def build_prompt(topic: str) -> str:

    return PROMPT_PLAYGROUND_TEMPLATE.format(
        topic=topic
    )