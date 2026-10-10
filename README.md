
# Text Integrity Checker

A Streamlit application that uses a multi-agent CrewAI system and Groq's LLM API to review academic text.

## Features

- Upload TXT, PDF, and DOCX documents.
- Paste text directly into the application.
- Compare text against user-provided reference documents.
- Review possible plagiarism and close paraphrasing.
- Examine writing patterns sometimes associated with AI-generated text.
- Review citations and attribution.
- Generate a Markdown report.
- Download the report.

## Technology

- Python
- Streamlit
- CrewAI
- Groq API
- PyPDF
- python-docx

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Set your Groq API key as an environment variable named `GROQ_API_KEY`.

Run the application:

```bash
streamlit run app.py
```

## Streamlit Community Cloud deployment

1. Upload all project files to the root of a GitHub repository.
2. Create an app in Streamlit Community Cloud.
3. Select the repository, branch, and `app.py` as the entry point.
4. Add your Groq API key under the app's Secrets settings:

```toml
GROQ_API_KEY = "your_actual_groq_api_key"
```

5. Deploy the app.

## Important limitations

This project compares the submitted document only against references supplied by the user. It does not perform a comprehensive internet-wide plagiarism search.

AI-writing pattern analysis is not proof of AI authorship. The report is intended to assist human review, not make definitive academic misconduct decisions.
