
import os
import streamlit as st

from document_reader import extract_text
from similarity import find_matches
from crew_runner import run_analysis


st.set_page_config(
    page_title="Text Integrity Checker",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 Text Integrity Checker")
st.write(
    "A multi-agent tool for reviewing possible text overlap, "
    "citation issues, and AI-associated writing patterns."
)

st.warning(
    "This tool is a screening aid, not a plagiarism verdict. "
    "AI-writing patterns cannot reliably prove whether a human or "
    "AI wrote a text. Similarity checking only compares references "
    "you provide."
)

with st.sidebar:
    st.header("Settings")
    st.write("Model: Groq `openai/gpt-oss-120b`")
    st.write("Agent framework: CrewAI")
    st.write("Supported files: TXT, PDF, DOCX")
    st.caption(
        "Keep your Groq API key in Streamlit Secrets. "
        "Do not put it directly in your code."
    )

st.subheader("1. Submit the document")

input_method = st.radio(
    "Choose how to provide your document",
    ["Paste text", "Upload a document"],
    horizontal=True,
)

submitted_text = ""

if input_method == "Paste text":
    submitted_text = st.text_area(
        "Document text",
        height=280,
        placeholder="Paste your essay, report, or assignment here...",
    )
else:
    uploaded_document = st.file_uploader(
        "Upload the document",
        type=["txt", "pdf", "docx"],
        help="Scanned PDFs may not contain extractable text.",
    )

    if uploaded_document is not None:
        try:
            submitted_text = extract_text(uploaded_document)
            st.success(
                f"Extracted {len(submitted_text):,} characters."
            )
            with st.expander("Preview extracted text"):
                st.text(submitted_text[:5000])
        except Exception as exc:
            st.error(f"Could not read the document: {exc}")

st.subheader("2. Add reference documents")
st.write(
    "Optional: upload or paste sources you want to compare against. "
    "Without references, the app cannot check whether your text "
    "matches other documents."
)

reference_uploads = st.file_uploader(
    "Upload reference documents",
    type=["txt", "pdf", "docx"],
    accept_multiple_files=True,
    key="reference_uploads",
)

reference_texts = []

for reference_file in reference_uploads or []:
    try:
        reference_text = extract_text(reference_file)
        if reference_text.strip():
            reference_texts.append(
                (reference_file.name, reference_text)
            )
    except Exception as exc:
        st.warning(
            f"Could not read reference {reference_file.name}: {exc}"
        )

with st.expander("Or paste reference text"):
    pasted_reference_name = st.text_input(
        "Reference name",
        value="Pasted reference",
    )
    pasted_reference_text = st.text_area(
        "Reference text",
        height=160,
        key="pasted_reference_text",
    )

if pasted_reference_text.strip():
    reference_texts.append(
        (pasted_reference_name.strip() or "Pasted reference",
         pasted_reference_text)
    )

st.caption(f"Reference documents ready: {len(reference_texts)}")

st.subheader("3. Run the analysis")

MAX_CHARACTERS = 18000

if len(submitted_text) > MAX_CHARACTERS:
    st.warning(
        f"The document has {len(submitted_text):,} characters. "
        f"For this version, please reduce it to {MAX_CHARACTERS:,} "
        "characters or fewer."
    )

if st.button("Analyze document", type="primary", use_container_width=True):
    if not submitted_text.strip():
        st.error("Please paste text or upload a document first.")

    elif len(submitted_text) > MAX_CHARACTERS:
        st.error(
            "The document is too long for this version. "
            "Please shorten it and try again."
        )

    else:
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            try:
                api_key = st.secrets.get("GROQ_API_KEY")
            except Exception:
                api_key = None

        if not api_key:
            st.error(
                "GROQ_API_KEY is missing. Add it in your app's "
                "Streamlit Secrets settings, then restart the app."
            )

        else:
            with st.spinner(
                "Running the agents. This may take a few minutes..."
            ):
                try:
                    matches = find_matches(
                        submitted_text,
                        reference_texts,
                    )

                    report = run_analysis(
                        submitted_text=submitted_text,
                        matches=matches,
                        api_key=api_key,
                    )

                    st.session_state["analysis_report"] = report
                    st.session_state["analysis_matches"] = matches

                except Exception as exc:
                    st.error(
                        "The analysis could not be completed. "
                        "Check your Groq API key, package versions, "
                        "and deployment logs."
                    )
                    st.code(str(exc))

if "analysis_report" in st.session_state:
    st.divider()
    st.subheader("4. Results")

    matches = st.session_state.get("analysis_matches", [])

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Potential matches", len(matches))

    with col2:
        st.metric("Reference documents", len(reference_texts))

    if matches:
        st.subheader("Potential text matches")

        for number, match in enumerate(matches, start=1):
            with st.expander(
                f"Match {number}: {match['similarity']}% similarity"
            ):
                st.write("**Submitted passage**")
                st.write(match["submitted_sentence"])

                st.write("**Reference**")
                st.write(match["reference_name"])

                st.write("**Reference passage**")
                st.write(match["reference_sentence"])

                st.caption(
                    "Similarity is a screening signal, not proof "
                    "of plagiarism. Review context and attribution."
                )
    else:
        st.info(
            "No matches passed the current similarity threshold "
            "against the supplied reference documents. This does "
            "not mean the text is plagiarism-free."
        )

    st.subheader("Multi-agent report")
    st.markdown(st.session_state["analysis_report"])

    st.download_button(
        label="Download report as Markdown",
        data=st.session_state["analysis_report"],
        file_name="text_integrity_report.md",
        mime="text/markdown",
        use_container_width=True,
    )
