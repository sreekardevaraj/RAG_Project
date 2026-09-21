from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config.prompts import FOLLOWUP_PROMPT
from config.settings import NUM_FOLLOWUP_QUESTIONS


def generate_followup_questions(question: str, answer: str, llm) -> list:
    """
    Generate follow-up question suggestions after every answer.
    Returns a list of question strings.
    """
    try:
        prompt = PromptTemplate(
            input_variables=["question", "answer"],
            template=FOLLOWUP_PROMPT
        )
        chain  = prompt | llm | StrOutputParser()
        result = chain.invoke({"question": question, "answer": answer})

        # Parse numbered list into clean list
        questions = []
        for line in result.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            # Remove numbering like "1." "1)" etc
            if line[0].isdigit():
                line = line.split(".", 1)[-1].strip()
                line = line.split(")", 1)[-1].strip()
            if line:
                questions.append(line)

        # Return only the requested number
        return questions[:NUM_FOLLOWUP_QUESTIONS]

    except Exception as e:
        print(f"[Followup] Error generating follow-up questions: {e}")
        return []