
from crewai import Crew, Process, Task

from llm_setup import create_llm
from document_analyst import build_agent as build_document_agent
from plagiarism_analyst import build_agent as build_plagiarism_agent
from ai_writing_analyst import build_agent as build_ai_agent
from citation_reviewer import build_agent as build_citation_agent
from report_analyst import build_agent as build_report_agent


def run_analysis(submitted_text, matches, api_key=None):
    """Run the multi-agent academic integrity analysis."""
    llm = create_llm(api_key)

    document_agent = build_document_agent(llm)
    plagiarism_agent = build_plagiarism_agent(llm)
    ai_agent = build_ai_agent(llm)
    citation_agent = build_citation_agent(llm)
    report_agent = build_report_agent(llm)

    match_text = "No matches found in the supplied reference documents."

    if matches:
        match_lines = []
        for number, match in enumerate(matches, start=1):
            match_lines.append(
                f"Match {number}:\n"
                f"Submitted text: {match['submitted_sentence']}\n"
                f"Reference: {match['reference_name']}\n"
                f"Reference passage: {match['reference_sentence']}\n"
                f"Similarity score: {match['similarity']}%"
            )
        match_text = "\n\n".join(match_lines)

    document_task = Task(
        description=(
            "Analyze the document's organization, argument, tone, "
            "sentence structure, repetition, and writing consistency. "
            "Do not determine authorship from style alone.\n\n"
            "DOCUMENT:\n{submitted_text}"
        ),
        expected_output=(
            "A concise evidence-based analysis of structure, style, "
            "and writing characteristics."
        ),
        agent=document_agent,
    )

    plagiarism_task = Task(
        description=(
            "Review possible text overlap using only the supplied "
            "similarity results. Discuss close wording, paraphrasing, "
            "quotation, attribution, and context. Do not call a match "
            "plagiarism without considering context.\n\n"
            "SUBMITTED DOCUMENT:\n{submitted_text}\n\n"
            "SIMILARITY FINDINGS:\n{match_text}"
        ),
        expected_output=(
            "A review of potential overlap, the strongest evidence, "
            "and the limitations of the available comparison."
        ),
        agent=plagiarism_agent,
        context=[document_task],
    )

    ai_task = Task(
        description=(
            "Review the submitted text for possible AI-associated "
            "writing patterns, such as unusually uniform sentence "
            "structure, repetitive transitions, generic phrasing, "
            "or inconsistent specificity. Provide examples from the "
            "text. Do not claim that these patterns prove AI use.\n\n"
            "DOCUMENT:\n{submitted_text}"
        ),
        expected_output=(
            "A cautious analysis of observable writing patterns, "
            "with examples and alternative explanations."
        ),
        agent=ai_agent,
        context=[document_task],
    )

    citation_task = Task(
        description=(
            "Review the submitted text for quotation, attribution, "
            "in-text citations, and references where visible. Identify "
            "possible missing or unclear attribution. Do not invent "
            "sources or claim to have verified references externally.\n\n"
            "DOCUMENT:\n{submitted_text}\n\n"
            "SIMILARITY FINDINGS:\n{match_text}"
        ),
        expected_output=(
            "A citation and attribution review with specific "
            "observations and limitations."
        ),
        agent=citation_agent,
        context=[plagiarism_task],
    )

    report_task = Task(
        description=(
            "Create a final Markdown report using the results of the "
            "document, overlap, AI-writing-pattern, and citation reviews. "
            "Include these headings:\n"
            "1. Executive Summary\n"
            "2. Document and Writing Analysis\n"
            "3. Potential Text Overlap\n"
            "4. AI-Writing Pattern Review\n"
            "5. Citation and Attribution Review\n"
            "6. Recommendations\n"
            "7. Limitations\n\n"
            "Distinguish observed evidence from interpretation. Do not "
            "invent a definitive plagiarism percentage or an AI-generated "
            "probability. State that the system checks only the reference "
            "material supplied by the user and that AI-writing indicators "
            "are not proof of authorship.\n\n"
            "DOCUMENT:\n{submitted_text}\n\n"
            "COMPUTED TEXT MATCHES:\n{match_text}"
        ),
        expected_output=(
            "A complete, clearly organized Markdown report with "
            "evidence, recommendations, and explicit limitations."
        ),
        agent=report_agent,
        context=[
            document_task,
            plagiarism_task,
            ai_task,
            citation_task,
        ],
    )

    crew = Crew(
        agents=[
            document_agent,
            plagiarism_agent,
            ai_agent,
            citation_agent,
            report_agent,
        ],
        tasks=[
            document_task,
            plagiarism_task,
            ai_task,
            citation_task,
            report_task,
        ],
        process=Process.sequential,
        verbose=False,
    )

    result = crew.kickoff(
        inputs={
            "submitted_text": submitted_text,
            "match_text": match_text,
        }
    )

    return str(result)
