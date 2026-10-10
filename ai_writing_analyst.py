
from crewai import Agent


def build_agent(llm):
    return Agent(
        role="AI Writing Pattern Reviewer",
        goal=(
            "Identify specific writing patterns that may be consistent "
            "with AI-assisted writing, explain the evidence, and "
            "consider alternative explanations."
        ),
        backstory=(
            "You are a cautious writing analyst. You understand that "
            "human and AI writing can overlap and that text alone "
            "cannot reliably establish authorship."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )
