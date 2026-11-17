"""Asks the language model every golden question and checks the answers.
Uses OpenAI (OPENAI_API_KEY) or Anthropic (ANTHROPIC_API_KEY), see rag/answer.py. Costs well under one cent.
The key can also sit in a .env file next to this script (it is in .gitignore).

  python eval_answers.py --chunker headings --mode hybrid --current-only
  python eval_answers.py --only 13                                    # one question
  python eval_answers.py ... --defend                                 # bonus 1: hidden-instruction defence
  python eval_answers.py ... --strict-quotes                          # exact-quote rules in the prompt
  python eval_answers.py ... --decompose --only 12                    # bonus 2: one search per model
"""
import argparse
import sys
import json
import os
from pathlib import Path

from eval_retrieval import build_index, needles
from rag.answer import PRICES, Usage, ask, decompose, model_name, provider, strip_html_comments, verify_citations
from rag.loader import load_docs
from rag.search import search, search_decomposed


def load_dotenv(path: str = ".env") -> None:
    """Reads KEY=value lines from .env into the environment (values already set are kept)."""
    if not Path(path).exists():
        return
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        if sep and key.strip() and not key.strip().startswith("#"):
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def check(q: dict, answer, bad_citations, chunks) -> tuple[bool, str]:
    if q["expect"] == "insufficient":
        ok = answer.insufficient_context and not answer.citations
        return ok, "" if ok else "should have said there is not enough information"
    if answer.insufficient_context:
        return False, "said there is not enough information"
    if bad_citations:
        return False, f"{len(bad_citations)} citation(s) not found in the chunks"
    quotes = "\n".join(c.quote for c in answer.citations)
    if not all(n in quotes for n in needles(q)):
        return False, "the quotes do not contain the answer"
    for word in q.get("forbidden_in_answer", []):
        if word in answer.answer:
            return False, f"the answer contains {word!r}"
    return True, ""


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # the ✓/✗ marks crash the default Windows console encoding
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunker", choices=["fixed", "headings"], default="fixed")
    ap.add_argument("--mode", choices=["vector", "hybrid"], default="vector")
    ap.add_argument("--current-only", action="store_true")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--only", type=int, help="run a single question id")
    ap.add_argument("--defend", action="store_true", help="strip HTML comments and mark chunks as data")
    ap.add_argument("--strict-quotes", action="store_true", help="add exact-quote rules to the prompt")
    ap.add_argument("--decompose", action="store_true", help="split multi-model questions into one search per model")
    ap.add_argument("--k-per-query", type=int, default=3, help="chunks per sub-query with --decompose")
    args = ap.parse_args()

    load_dotenv()
    golden = json.loads(Path("golden.json").read_text(encoding="utf-8"))
    if args.only:
        golden = [q for q in golden if q["id"] == args.only]
    index = build_index(args.chunker)
    models = [d.title for d in load_docs() if d.meta.get("type") == "model"]
    print(f"provider: {provider()} · model: {model_name()}")
    passed, total = 0, Usage(0, 0)
    for q in golden:
        queries = [q["question"]]
        if args.decompose:
            queries, usage = decompose(q["question"], models)
            total = Usage(total.input_tokens + usage.input_tokens, total.output_tokens + usage.output_tokens)
        if len(queries) > 1:
            chunks = search_decomposed(index, queries, args.mode, args.k_per_query, args.current_only)
        else:
            chunks = search(index, q["question"], args.mode, args.k, args.current_only)
        if args.defend:
            chunks = strip_html_comments(chunks)
        answer, usage = ask(q["question"], chunks, defend=args.defend, strict_quotes=args.strict_quotes)
        total = Usage(total.input_tokens + usage.input_tokens, total.output_tokens + usage.output_tokens)
        bad = verify_citations(answer, chunks)
        ok, why = check(q, answer, bad, chunks)
        passed += ok
        print(f"\n{q['id']:>2} {'✓' if ok else '✗'} [{q['type']}] {q['question']}")
        if len(queries) > 1:
            print(f"   sub-queries: {' | '.join(queries)}")
            print(f"   chunks: {', '.join(c.id for c in chunks)}")
        print(f"   {answer.answer}")
        for c in answer.citations:
            mark = "✗" if c in bad else "✓"
            print(f"   {mark} {c.chunk_id}: \"{c.quote}\"")
        if why:
            print(f"   → {why}")

    price = PRICES.get(model_name())
    cost = (f"${total.input_tokens / 1e6 * price[0] + total.output_tokens / 1e6 * price[1]:.4f}" if price
            else "cost unknown (add the model to PRICES)")
    print(f"\npassed {passed}/{len(golden)} · {total.input_tokens:,} input + {total.output_tokens:,} output tokens · {cost}")


if __name__ == "__main__":
    main()
