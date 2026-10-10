
from crewai import Agent


def build_agent(llm):
    return Agent(
        role="Citation and Attribution Reviewer",
        goal=(
            "Review the supplied text for citations, quotation marks, "
            "attribution, and possible citation inconsistencies."
        ),
        backstory=(
            "You are an academic citation reviewer. You distinguish "
            "missing or unclear attribution from proven plagiarism. "
            "You do not invent bibliographic details or claim that "
            "a source is authentic without verification."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )
