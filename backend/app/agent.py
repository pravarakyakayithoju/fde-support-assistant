"""Agent loop with manual tool calling, so every step can be traced.
Works with Groq (OpenAI format) or Gemini — set LLM_PROVIDER in .env."""
import json
import re
import time

from .config import BUSINESS_NAME, CHAT_MODEL, LLM_PROVIDER, PRICE_IN_PER_M, PRICE_OUT_PER_M
from .db import conn
from .llm import gemini, groq
from .tools import OPENAI_TOOLS, gemini_tool, run_tool

SYSTEM = f"""You are the support and operations assistant for {BUSINESS_NAME}.

Rules:
- For any question about policies, rules, dates, fees or procedures, call search_docs first.
- Answer ONLY from the passages returned. Cite them inline as [1], [2] matching the passage numbers.
- If the passages don't contain the answer, say "I couldn't find this in the official documents"
  and offer to create a support ticket. Never guess.
- Use lookup_record when the user gives an ID. Use create_ticket / email_summary only when asked
  or clearly needed, and confirm what you did.
- Be complete but brief: include every amount, condition, eligibility rule or exception in the
  passages that affects the answer (e.g. "only from second year", "paid separately", "compulsory").
  Don't add unrelated facts.
- After using a tool, state the key result in your reply (record status, ticket ID, what the email said)."""

MAX_STEPS = 6
GAVE_UP = "Sorry, I couldn't complete that request."


def _exec(name: str, args: dict, trace: dict) -> dict:
    """Run one tool and record it in the trace."""
    ts = time.perf_counter()
    result = run_tool(name, args)
    chunks = result.pop("_chunks", None)
    if chunks:
        trace["retrieved"].extend(chunks)
    trace["tool_calls"].append({
        "name": name, "args": args, "result": json.dumps(result)[:500],
        "ms": round((time.perf_counter() - ts) * 1000),
    })
    return result


def _groq_loop(question: str, trace: dict) -> str:
    client = groq()
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    for _ in range(MAX_STEPS):
        resp = client.chat.completions.create(
            model=CHAT_MODEL, messages=msgs, tools=OPENAI_TOOLS, tool_choice="auto", temperature=0.1,
        )
        if resp.usage:
            trace["in_tok"] += resp.usage.prompt_tokens or 0
            trace["out_tok"] += resp.usage.completion_tokens or 0
        msg = resp.choices[0].message
        if not msg.tool_calls:
            return msg.content or ""
        msgs.append({
            "role": "assistant", "content": msg.content or "",
            "tool_calls": [{"id": tc.id, "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                           for tc in msg.tool_calls],
        })
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            result = _exec(tc.function.name, args, trace)
            msgs.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})
    return GAVE_UP


def _gemini_loop(question: str, trace: dict) -> str:
    from google.genai import types
    client = gemini()
    contents = [types.Content(role="user", parts=[types.Part(text=question)])]
    cfg = types.GenerateContentConfig(
        system_instruction=SYSTEM, tools=[gemini_tool()], temperature=0.1,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    for _ in range(MAX_STEPS):
        resp = client.models.generate_content(model=CHAT_MODEL, contents=contents, config=cfg)
        u = resp.usage_metadata
        if u:
            trace["in_tok"] += u.prompt_token_count or 0
            trace["out_tok"] += u.candidates_token_count or 0
        calls = resp.function_calls or []
        if not calls:
            return resp.text or ""
        contents.append(resp.candidates[0].content)
        parts = [types.Part.from_function_response(name=fc.name, response=_exec(fc.name, dict(fc.args or {}), trace))
                 for fc in calls]
        contents.append(types.Content(role="user", parts=parts))
    return GAVE_UP


def run(question: str) -> dict:
    t0 = time.perf_counter()
    trace = {"retrieved": [], "tool_calls": [], "in_tok": 0, "out_tok": 0}
    error = None
    try:
        loop = _groq_loop if LLM_PROVIDER == "groq" else _gemini_loop
        answer = loop(question, trace)
        # Some models cite as 【1】 or 【1†L4-L7】; normalise to [1]
        answer = re.sub(r"【(\d+)[^】]*】", r" [\1]", answer).replace("\u202f", " ").strip()
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        answer = "Something went wrong. Please try again."
        print("agent error:", error)  # also shows in the uvicorn window

    latency = round((time.perf_counter() - t0) * 1000)
    cost = trace["in_tok"] / 1e6 * PRICE_IN_PER_M + trace["out_tok"] / 1e6 * PRICE_OUT_PER_M
    trace_id = _save_trace(question, answer, trace, latency, cost, error)

    return {
        "answer": answer,
        "sources": trace["retrieved"],
        "tool_calls": trace["tool_calls"],
        "trace_id": trace_id,
        "latency_ms": latency,
        "tokens": {"input": trace["in_tok"], "output": trace["out_tok"]},
        "cost_usd": round(cost, 6),
        "error": error,
    }


def _save_trace(q, a, tr, latency, cost, error) -> int | None:
    try:
        with conn() as c:
            return c.execute(
                """INSERT INTO traces (question, answer, retrieved, tool_calls, latency_ms,
                                       input_tokens, output_tokens, cost_usd, error)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
                (q, a, json.dumps(tr["retrieved"]), json.dumps(tr["tool_calls"]), latency,
                 tr["in_tok"], tr["out_tok"], cost, error),
            ).fetchone()["id"]
    except Exception as e:
        print("trace save failed:", e)
        return None
