from backend.rag_pipeline import build_rag_pipeline, ask_question


def main():
    print("\n" + "="*55)
    print("     PDF RAG Chatbot V1 - Smart Routing Test")
    print("="*55 + "\n")

    pipeline = build_rag_pipeline(force_rebuild=False)

    print("\nPipeline ready! Type your questions below.")
    print("Answers come from PDF or Web automatically.")
    print("type exit to quit\n")

    while True:
        question = input("You: ").strip()

        if question.lower() in ["exit", "quit"]:
            print("Bye!")
            break

        if not question:
            continue

        response = ask_question(pipeline, question)

        print(f"\n[Source: {response['source']}]")
        print(f"\nAnswer:\n{response['answer']}")

        print("\nReferences:")
        for item in response["details"]:
            if response["source"] == "PDF":
                print(f"   - Page {item.get('page', '?')} | {item.get('source', 'unknown')}")
            else:
                print(f"   - {item.get('title', '')} -> {item.get('url', '')}")

        print("\n" + "-"*55 + "\n")


if __name__ == "__main__":
    main()