from __future__ import annotations

ANSWER_CONTRACT_SYSTEM = (
    "You are qq, a local context assistant."
    " Always produce a single JSON object with keys:"
    " answer (string), citations (list of {id, source, score}),"
    " commands (optional list of {step, command, cwd}),"
    " followups (optional list of strings), confidence (0..1)."
    " Be concise and actionable."
)


def build_user_prompt(question: str, contexts: list[dict]) -> str:
    ctx_lines = []
    for c in contexts:
        ctx_lines.append(f"- [{c.get('id')}] ({c.get('score', 0):.3f}) {c.get('source')}: {c.get('text')[:400]}")
    joined = "\n".join(ctx_lines)
    return f"Question: {question}\n\nContext:\n{joined}\n\nRespond with the JSON contract only."

