import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

import ollama

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from weaviate_v3.grounded.grounded_rag import ChatRAG
from weaviate_v3.unified.unified_rag import UnifiedChatRAG


CHOICE_RE = re.compile(r"(?<!\d)([1-4])(?!\d)")


def resolve_output_path(output: str) -> Path:
    candidate = Path(output)
    if candidate.is_absolute():
        return candidate
    if candidate.parent == Path('.'):
        return REPO_ROOT / 'eval' / candidate.name
    return candidate


class EvalRunner(Protocol):
    def answer_question(self, question: str, options: List[str]) -> Dict[str, Any]:
        ...

    def close(self) -> None:
        ...


def load_eval_dataset(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open() as handle:
        for idx, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            question, options, answer = json.loads(line)
            records.append(
                {
                    "id": idx,
                    "question": question,
                    "options": options,
                    "answer": str(answer),
                }
            )
    return records


def build_mcq_prompt(question: str, options: List[str]) -> str:
    options_block = "\n".join(options)
    return (
        "You are answering a multiple-choice exam. "
        "Your response must be a single character: 1, 2, 3, or 4. "
        "You are an O-RAN expert. Use the provided context to answer the question.\n"
        "Do not write anything else. No explanation. No punctuation. No words. Just the number.\n\n"
        f"Question: {question}\n"
        f"Options:\n{options_block}\n\n"
        "Respond with only the number of the correct option.\n"
        "Answer:"
    )


def extract_choice(text: str) -> Optional[str]:
    if not text:
        return None
    stripped = text.strip()
    if stripped in {"1", "2", "3", "4"}:
        return stripped
    match = CHOICE_RE.search(stripped)
    if match:
        return match.group(1)
    return None


class NoRAGRunner:
    def __init__(self, model: str, ollama_host: str, ollama_port: int):
        self.model = model
        self.client = ollama.Client(host=f"http://{ollama_host}:{ollama_port}")

    def answer_question(self, question: str, options: List[str]) -> Dict[str, Any]:
        prompt = build_mcq_prompt(question, options)
        response = self.client.generate(model=self.model, prompt=prompt)
        raw_answer = response.get("response", "")
        return {
            "raw_answer": raw_answer,
            "predicted": extract_choice(raw_answer),
            "sources": [],
        }

    def close(self) -> None:
        return None


class GroundedRunner:
    def __init__(self, args: argparse.Namespace):
        self.rag = ChatRAG(
            collection_name=args.grounded_collection,
            weaviate_host=args.grounded_weaviate_host,
            weaviate_port=args.grounded_weaviate_port,
            weaviate_grpc_port=args.grounded_weaviate_grpc_port,
            ollama_host=args.ollama_host,
            ollama_port=args.ollama_port,
            ollama_model=args.model,
            enable_phoenix=not args.disable_phoenix,
        )
        self.top_k = args.top_k
        self.query_alpha = args.grounded_query_alpha
        self.block_filter = args.block_filter

    def answer_question(self, question: str, options: List[str]) -> Dict[str, Any]:
        self.rag.clear_history()
        prompt = build_mcq_prompt(question, options)
        result = self.rag.chat(
            user_message=prompt,
            use_rag=True,
            top_k=self.top_k,
            query_alpha=self.query_alpha,
            return_sources=True,
            block_filter=self.block_filter,
        )
        raw_answer = result.get("answer", "")
        return {
            "raw_answer": raw_answer,
            "predicted": extract_choice(raw_answer),
            "sources": result.get("sources", []),
        }

    def close(self) -> None:
        self.rag.close()


class UnifiedRunner:
    def __init__(self, args: argparse.Namespace):
        self.rag = UnifiedChatRAG(
            collection_name=args.unified_collection,
            weaviate_host=args.unified_weaviate_host,
            weaviate_port=args.unified_weaviate_port,
            weaviate_grpc_port=args.unified_weaviate_grpc_port,
            ollama_host=args.ollama_host,
            ollama_port=args.ollama_port,
            ollama_model=args.model,
            multi2vec_host=args.multi2vec_host,
            multi2vec_port=args.multi2vec_port,
            summarize_threshold_tokens=args.unified_summarize_threshold,
            multimodal=False,
        )
        self.top_k = args.top_k
        self.query_alpha = args.unified_query_alpha
        self.modality_balance = args.unified_modality_balance
        self.block_filter = args.block_filter

    def answer_question(self, question: str, options: List[str]) -> Dict[str, Any]:
        self.rag.clear_history()
        prompt = build_mcq_prompt(question, options)
        result = self.rag.chat(
            user_message=prompt,
            use_rag=True,
            top_k=self.top_k,
            query_alpha=self.query_alpha,
            modality_balance=self.modality_balance,
            return_sources=True,
            block_filter=self.block_filter,
            system_prompt="Return only the number 1, 2, 3, or 4. No other text.",
        )
        raw_answer = result.get("answer", "")
        return {
            "raw_answer": raw_answer,
            "predicted": extract_choice(raw_answer),
            "sources": result.get("sources", []),
            "summary_info": result.get("summary_info", {}),
            "retrieval_mode": result.get("retrieval_mode"),
            "retrieval_warning": result.get("retrieval_warning"),
        }

    def close(self) -> None:
        self.rag.close()


def build_runner(args: argparse.Namespace) -> EvalRunner:
    if args.pipeline == "none":
        return NoRAGRunner(model=args.model, ollama_host=args.ollama_host, ollama_port=args.ollama_port)
    if args.pipeline == "grounded":
        return GroundedRunner(args)
    if args.pipeline == "unified":
        return UnifiedRunner(args)
    raise ValueError(f"Unsupported pipeline: {args.pipeline}")


def evaluate(records: List[Dict[str, Any]], runner: EvalRunner, limit: Optional[int]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    subset = records[:limit] if limit else records

    correct = 0
    for record in subset:
        result = runner.answer_question(record["question"], record["options"])
        is_correct = result.get("predicted") == record["answer"]
        correct += int(is_correct)

        rows.append(
            {
                "id": record["id"],
                "question": record["question"],
                "options": record["options"],
                "gold": record["answer"],
                "predicted": result.get("predicted"),
                "correct": is_correct,
                "raw_answer": result.get("raw_answer", ""),
                "sources": result.get("sources", []),
                "summary_info": result.get("summary_info"),
                "retrieval_mode": result.get("retrieval_mode"),
                "retrieval_warning": result.get("retrieval_warning"),
            }
        )

    total = len(rows)
    return {
        "total": total,
        "correct": correct,
        "accuracy": (correct / total) if total else 0.0,
        "details": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate MCQ performance with a selectable pipeline: none, grounded, or unified."
    )
    parser.add_argument("--dataset", default="dataset/fin_E.json")
    parser.add_argument("--pipeline", choices=["none", "grounded", "unified"], default="grounded")
    parser.add_argument("--model", default="llama3.2")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default=None, help="Optional path to write detailed JSON results.")
    parser.add_argument("--ollama-host", default="172.17.0.6")
    parser.add_argument("--ollama-port", type=int, default=11434)
    parser.add_argument("--disable-phoenix", action="store_true")
    parser.add_argument(
        "--block-filter",
        choices=["chunks"],
        default=None,
        help="Optional server-side Weaviate block filter preset applied to grounded or unified retrieval.",
    )

    parser.add_argument("--grounded-collection", default="Grounded_nomic_full")
    parser.add_argument("--grounded-weaviate-host", default="172.17.0.5")
    parser.add_argument("--grounded-weaviate-port", type=int, default=8080)
    parser.add_argument("--grounded-weaviate-grpc-port", type=int, default=50051)
    parser.add_argument("--grounded-query-alpha", type=float, default=0.7)

    parser.add_argument("--unified-collection", default="Unified_embedding")
    parser.add_argument("--unified-weaviate-host", default="172.17.0.5")
    parser.add_argument("--unified-weaviate-port", type=int, default=8080)
    parser.add_argument("--unified-weaviate-grpc-port", type=int, default=50051)
    parser.add_argument("--multi2vec-host", default="172.17.0.7")
    parser.add_argument("--multi2vec-port", type=int, default=8080)
    parser.add_argument("--unified-query-alpha", type=float, default=0.7)
    parser.add_argument("--unified-modality-balance", type=float, default=0.5)
    parser.add_argument("--unified-summarize-threshold", type=int, default=70)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_eval_dataset(Path(args.dataset))
    runner = build_runner(args)

    try:
        summary = evaluate(records, runner=runner, limit=args.limit)
    finally:
        runner.close()

    report = {
        "dataset": args.dataset,
        "pipeline": args.pipeline,
        "model": args.model,
        "top_k": args.top_k,
        "block_filter": args.block_filter,
        "total": summary["total"],
        "correct": summary["correct"],
        "accuracy": summary["accuracy"],
    }
    print(json.dumps(report, indent=2))

    if args.output:
        output_path = resolve_output_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "config": vars(args),
                    **summary,
                },
                indent=2,
            )
        )
        print(f"Wrote detailed results to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
