"""Actions the agent can take beyond answering. Each tool = a declaration + a Python function."""
import json

from .db import conn
from .rag import format_context, retrieve


def search_docs(query: str) -> dict:
    chunks = retrieve(query)
    return {
        "context": format_context(chunks),
        "_chunks": [  # stripped before sending to the model; kept for tracing/citations
            {"chunk_id": c["chunk_id"], "source": c["source"], "page": c["page"],
             "score": round(float(c["score"]), 4), "vec_rank": c["vec_rank"], "kw_rank": c["kw_rank"], "snippet": c["content"][:200],
             "content": c["content"]}
            for c in chunks
        ],
    }


def lookup_record(record_id: str) -> dict:
    with conn() as c:
        row = c.execute("SELECT * FROM records WHERE id = %s", (record_id.upper(),)).fetchone()
    return row or {"error": f"No record found with id {record_id}"}


def create_ticket(subject: str, body: str, record_id: str = "", priority: str = "normal") -> dict:
    with conn() as c:
        row = c.execute(
            "INSERT INTO tickets (record_id, subject, body, priority) VALUES (%s,%s,%s,%s) "
            "RETURNING id, status",
            (record_id or None, subject, body, priority),
        ).fetchone()
    return {"ticket_id": row["id"], "status": row["status"]}


def email_summary(to: str, subject: str, body: str) -> dict:
    # Stub: log instead of sending. Swap for SendGrid/Gmail API later if you want.
    print(f"[EMAIL] to={to} subject={subject}\n{body}")
    return {"sent": True, "to": to, "note": "simulated"}


IMPLS = {
    "search_docs": search_docs,
    "lookup_record": lookup_record,
    "create_ticket": create_ticket,
    "email_summary": email_summary,
}

# Tool definitions in plain JSON Schema, shared by both providers.
TOOL_SPECS = [
    {"name": "search_docs",
     "description": "Search the organisation's official documents (rules, circulars, FAQs). "
                    "Use for ANY policy or factual question. Returns numbered passages.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "lookup_record",
     "description": "Look up a specific applicant/customer record by ID, e.g. APP-1001.",
     "parameters": {"type": "object", "properties": {"record_id": {"type": "string"}}, "required": ["record_id"]}},
    {"name": "create_ticket",
     "description": "Create a support ticket when the user needs human follow-up or the docs don't answer.",
     "parameters": {"type": "object", "properties": {
         "subject": {"type": "string"}, "body": {"type": "string"}, "record_id": {"type": "string"},
         "priority": {"type": "string", "enum": ["low", "normal", "high"]}},
         "required": ["subject", "body"]}},
    {"name": "email_summary",
     "description": "Email a summary of the conversation or answer to an address the user gives.",
     "parameters": {"type": "object", "properties": {
         "to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}},
         "required": ["to", "subject", "body"]}},
]

# OpenAI/Groq format
OPENAI_TOOLS = [{"type": "function", "function": t} for t in TOOL_SPECS]


def gemini_tool():
    """Gemini format, built from the same specs."""
    from google.genai import types

    def schema(d):
        return types.Schema(
            type=types.Type[d["type"].upper()],
            properties={k: schema(v) for k, v in d.get("properties", {}).items()} or None,
            required=d.get("required"), enum=d.get("enum"),
        )
    return types.Tool(function_declarations=[
        types.FunctionDeclaration(name=t["name"], description=t["description"], parameters=schema(t["parameters"]))
        for t in TOOL_SPECS
    ])


def run_tool(name: str, args: dict) -> dict:
    fn = IMPLS.get(name)
    args = {k: v for k, v in (args or {}).items() if v is not None}  # some models send nulls
    if not fn:
        return {"error": f"unknown tool {name}"}
    try:
        return json.loads(json.dumps(fn(**args), default=str))
    except Exception as e:  # surface tool errors to the model instead of crashing
        return {"error": f"{type(e).__name__}: {e}"}
