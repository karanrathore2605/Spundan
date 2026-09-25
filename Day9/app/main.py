import logging
from .agent import RuleBasedAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

def main() -> None:
    print("=" * 60)
    print("             DAY 9 - RULE-BASED AI AGENT")
    print("=" * 60)
    print("Type your question or 'exit' to quit.\n")

    agent = RuleBasedAgent()

    while True:
        try:
            question = input("Question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        result = agent.run(question)
        print(f"\nRoute: {result['tool']}")
        print(f"Answer: {result['answer']}")

        if result.get("sources"):
            print("Sources:")
            for source in result["sources"]:
                print(f"- {source}")
        print()

if __name__ == "__main__":
    main()
