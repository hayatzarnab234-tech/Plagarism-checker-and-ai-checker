
from crewai import Agent


def build_agent(llm):
    return Agent(
        role="Text Similarity and Plagiarism Reviewer",
        goal=(
            "Assess potential text overlap using the supplied reference "
            "documents and similarity findings. Explain evidence and "
            "avoid unsupported accusations."
        ),
        backstory=(
            "You are an academic integrity reviewer. Similarity alone "
            "does not prove plagiarism. You examine wording, context, "
            "attribution, quotations, and limitations."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )
