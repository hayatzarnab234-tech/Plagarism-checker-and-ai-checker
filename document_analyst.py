
from crewai import Agent


def build_agent(llm):
    return Agent(
        role="Document Analysis Specialist",
        goal=(
            "Analyze the submitted text's structure, organization, "
            "argument, tone, and writing characteristics."
        ),
        backstory=(
            "You are a careful academic document analyst. You describe "
            "observable features of a text and distinguish evidence "
            "from assumptions."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )
