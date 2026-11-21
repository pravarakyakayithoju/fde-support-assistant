"""Score the assistant on a fixed question set.

Each line in questions.jsonl:
  {"id": "q01", "type": "answerable|unanswerable|tool",
   "question": "...", "expected": "key facts the answer must contain",
   "expected_tool": "lookup_record"   # only for type=tool}

Metrics:
  accuracy           – judge says the answer matches `expected`
  hallucination rate – answer states facts not supported by the retrieved passages
                       (or answers an unanswerable question instead of declining)
  citation rate      – answerable answers that contain at least one [n] citation
  tool accuracy      – type=tool questions where the expected tool was called

Usage (from backend/):  python ../evals/run_evals.py [--limit N]
"""
import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app import agent  # noqa: E402
from app.db import pool  # noqa: E402
from app.llm import complete_json  # noqa: E402

JUDGE = """You are grading a support assistant. Return JSON only:
{"correct": true|false, "hallucinated": true|false, "reason": "<one sentence>"}

- correct: the answer contains the key facts in EXPECTED (wording can differ). For an
  UNANSWERABLE question, correct means the assistant declined / said it couldn't find it.
- hallucinated: the answer states a specific fact (number, date, rule) that is NOT supported
  by the RETRIEVED PASSAGES or the TOOL RESULTS. Declining is never a hallucination.

QUESTION TYPE: {qtype}
QUESTION: {question}
EXPECTED: {expected}
RETRIEVED PASSAGES:
{passages}
TOOL RESULTS:
{tools}
ANSWER:
{answer}"""


def judge(item, out) -> dict:
    prompt = (JUDGE.replace("{qtype}", item["type"])
                   .replace("{question}", item["question"])
                   .replace("{expected}", item.get("expected", ""))
                   .replace("{passages}", "\n\n".join(s.get("content", s["snippet"]) for s in out["sources"]) or "(none)")
                   .replace("{tools}", json.dumps(out["tool_calls"])[:3000])
                   .replace("{answer}", out["answer"]))
    text = complete_json(prompt)
    try:
        return json.loads(text)
    except Exception:
        return {"correct": False, "hallucinated": False, "reason": f"judge parse error: {text[:100]}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(Path(__file__).with_name("questions.jsonl")))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--ids", help="comma-separated question ids to run, e.g. q06,q16,q17")
    args = ap.parse_args()

    items = [json.loads(l) for l in open(args.file, encoding="utf-8") if l.strip()]
    if args.ids:
        wanted = {i.strip() for i in args.ids.split(",")}
        items = [it for it in items if it["id"] in wanted]
    items = items[: args.limit]
    results = []
    for it in items:
        out = agent.run(it["question"])
        g = judge(it, out)
        tools_called = [t["name"] for t in out["tool_calls"]]
        r = {
            "id": it["id"], "type": it["type"],
            "correct": bool(g.get("correct")),
            "hallucinated": bool(g.get("hallucinated")),
            "cited": bool(re.search(r"\[\d+\]", out["answer"])),
            "tool_ok": it.get("expected_tool") in tools_called if it.get("expected_tool") else None,
            "latency_ms": out["latency_ms"], "cost": out["cost_usd"],
            "reason": g.get("reason", ""), "answer": out["answer"], "error": out.get("error"),
            "tools_called": tools_called,
            "retrieved": [f'{Path(x["source"]).name} ({x["score"]}): {x["snippet"][:120]}' for x in out["sources"]],
        }
        results.append(r)
        mark = "✓" if r["correct"] else "✗"
        print(f"{mark} {it['id']:<5} {'HALLUC ' if r['hallucinated'] else ''}{r['reason'][:90]}")
        time.sleep(2)  # stay under free-tier rate limits

    n = len(results)
    if not results:
        print("no questions selected"); return
    ans = [r for r in results if r["type"] == "answerable"]
    tool = [r for r in results if r["tool_ok"] is not None]
    summary = {
        "n": n,
        "accuracy": round(sum(r["correct"] for r in results) / n, 3),
        "hallucination_rate": round(sum(r["hallucinated"] for r in results) / n, 3),
        "citation_rate": round(sum(r["cited"] for r in ans) / len(ans), 3) if ans else None,
        "tool_accuracy": round(sum(r["tool_ok"] for r in tool) / len(tool), 3) if tool else None,
        "p50_latency_ms": statistics.median(r["latency_ms"] for r in results),
        "total_cost_usd": round(sum(r["cost"] for r in results), 4),
    }
    print("\n" + json.dumps(summary, indent=2))

    out_dir = Path(__file__).with_name("results")
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"run_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps({"summary": summary, "results": results}, indent=2))
    print(f"saved {path}")


if __name__ == "__main__":
    try:
        main()
    finally:
        pool.close()  # stops the "couldn't stop thread" warnings
