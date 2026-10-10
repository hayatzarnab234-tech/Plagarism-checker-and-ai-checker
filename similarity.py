
import re
from difflib import SequenceMatcher


def split_sentences(text):
    """Split text into approximate sentences."""
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def normalize(text):
    """Normalize text for a basic similarity comparison."""
    return re.sub(r"\W+", " ", text.lower()).strip()


def find_matches(
    submitted_text,
    reference_texts,
    threshold=0.78,
    max_matches=30,
):
    """
    Compare submitted sentences against supplied reference texts.

    The similarity score is a screening signal, not a plagiarism verdict.
    """
    submitted_sentences = split_sentences(submitted_text)
    matches = []

    references = []

    for reference_name, reference_text in reference_texts:
        for reference_sentence in split_sentences(reference_text):
            normalized_reference = normalize(reference_sentence)

            if normalized_reference:
                references.append(
                    (
                        reference_name,
                        reference_sentence,
                        normalized_reference,
                    )
                )

    for sentence in submitted_sentences:
        normalized_sentence = normalize(sentence)

        if len(normalized_sentence.split()) < 5:
            continue

        best_match = None
        best_score = 0.0

        for reference_name, reference_sentence, normalized_reference in references:
            score = SequenceMatcher(
                None,
                normalized_sentence,
                normalized_reference,
            ).ratio()

            if score > best_score:
                best_score = score
                best_match = (reference_name, reference_sentence)

        if best_match and best_score >= threshold:
            matches.append(
                {
                    "submitted_sentence": sentence,
                    "reference_name": best_match[0],
                    "reference_sentence": best_match[1],
                    "similarity": round(best_score * 100, 1),
                }
            )

    matches.sort(key=lambda item: item["similarity"], reverse=True)
    return matches[:max_matches]
