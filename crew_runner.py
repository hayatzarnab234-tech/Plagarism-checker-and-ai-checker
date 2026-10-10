
import crewai.llms.cache as crew_cache

# Prevent unsupported cache_breakpoint metadata from being added.
crew_cache.mark_cache_breakpoint = lambda msg: msg

from crewai import Crew, Process, Task

from llm_setup import create_llm
from document_analyst import build_agent as build_document_agent
from plagiarism_analyst import build_agent as build_plagiarism_agent
from ai_writing_analyst import build_agent as build_ai_agent
from citation_reviewer import build_agent as build_citation_agent
from report_analyst import build_agent as build_report_agent


# Keep requests within a manageable size for Groq.
MAX_DOCUMENT_CHARS = 6000
MAX_MATCH_CHARS = 2000


def limit_text(text, max_chars):
    """Safely limit the length of text sent to the LLM."""
    if not text:
        return ""
    return str(text)[:max_chars]


def build_match_text(matches):
    """Create a compact summary of supplied similarity matches."""
    if not matches:
        return "No matches found in the supplied reference documents."

    match_lines = []

    for number, match in enumerate(matches, start=1):
        submitted = str(match.get("submitted_sentence", ""))
        reference_name = str(match.get("reference_name", "Unknown"))
        reference = str(match.get("reference_sentence", ""))
        similarity = match.get("similarity", "Unknown")

        match_lines.append(
            f"Match {number}:\n"
            f"Submitted text: {submitted}\n"
            f"Reference: {reference_name}\n"
            f"Reference passage: {reference}\n"
            f"Similarity score: {similarity}%"
        )

    return limit_text(
        "\n\n".join(match_lines),
        MAX_MATCH_CHARS,
    )


def run_analysis(submitted_text, matches, api_key=None):
    """Run the multi-agent academic integrity analysis."""

    submitted_text = limit_text(
        submitted_text,
        MAX_DOCUMENT_CHARS,
    )
    match_text = build_match_text(matches)

    llm = create_llm(api_key)

    document_agent = build_document_agent(llm)
    plagiarism_agent = build_plagiarism_agent(llm)
    ai_agent = build_ai_agent(llm)
    citation_agent = build_citation_agent(llm)
    report_agent = build_report_agent(llm)

    document_task = Task(
        description=(
            "Analyze the supplied document's organization, argument, "
            "tone, sentence structure, repetition, and writing consistency. "
            "Do not determine authorship from style alone.\n\n"
            "DOCUMENT:\n{submitted_text}"
        ),
        expected_output=(
            "A concise, evidence-based analysis of structure, style, "
            "and writing characteristics, with brief examples."
        ),
        agent=document_agent,
    )

    plagiarism_task = Task(
        description=(
            "Review possible text overlap using only the supplied "
            "similarity results. Discuss wording, paraphrasing, quotation, "
            "attribution, and context. Do not call a match plagiarism "
            "without considering context.\n\n"
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
            "Review the document for possible AI-associated writing "
            "patterns, including uniform sentence structure, repetitive "
            "transitions, generic phrasing, and inconsistent specificity. "
            "Give examples when possible. These patterns do not prove "
            "AI authorship.\n\n"
            "DOCUMENT:\n{submitted_text}"
        ),
        expected_output=(
            "A cautious review of observable writing patterns, examples, "
            "and alternative explanations."
        ),
        agent=ai_agent,
        context=[document_task],
    )

    citation_task = Task(
        description=(
            "Review quotation, attribution, in-text citations, and "
            "references visible in the supplied document. Identify "
            "possible missing or unclear attribution. Do not invent "
            "sources or claim external verification.\n\n"
            "SIMILARITY FINDINGS:\n{match_text}"
        ),
        expected_output=(
            "A citation and attribution review with specific observations "
            "and limitations."
        ),
        agent=citation_agent,
        context=[plagiarism_task],
    )

    report_task = Task(
        description=(
            "Create a final Markdown report based on the preceding "
            "analysis tasks. Do not repeat the entire submitted document "
            "or invent evidence.\n\n"
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
            "probability. Explain that comparisons cover only the supplied "
            "reference material and that AI-writing indicators are not "
            "proof of authorship."
        ),
        expected_output=(
            "A complete, clearly organized Markdown report with evidence, "
            "recommendations, and explicit limitations."
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

