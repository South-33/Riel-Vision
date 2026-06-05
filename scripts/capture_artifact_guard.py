from __future__ import annotations


DERIVED_CAPTURE_ARTIFACT_TOKENS = {
    "annotated",
    "bpmn",
    "contact_sheet",
    "diagram",
    "overlay",
    "prediction",
    "preview",
    "screenshot",
    "synthetic",
}


def derived_capture_tokens(path_text: str) -> list[str]:
    haystack = path_text.replace("\\", "/").lower()
    return sorted(token for token in DERIVED_CAPTURE_ARTIFACT_TOKENS if token in haystack)


def derived_capture_reason(path_text: str) -> str:
    tokens = derived_capture_tokens(path_text)
    if not tokens:
        return ""
    return f"likely derived/non-raw capture artifact ({','.join(tokens)})"
