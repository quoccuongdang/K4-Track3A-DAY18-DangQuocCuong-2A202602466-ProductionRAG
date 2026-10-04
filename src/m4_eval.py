from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    from config import OPENAI_API_KEY, OPENAI_BASE_URL, LLM_MODEL

    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from datasets import Dataset

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })

        kwargs = {}
        if OPENAI_API_KEY:
            try:
                from langchain_openai import ChatOpenAI
                kwargs["llm"] = ChatOpenAI(
                    model=LLM_MODEL,
                    api_key=OPENAI_API_KEY,
                    base_url=OPENAI_BASE_URL if OPENAI_BASE_URL else None
                )
            except Exception:
                pass
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    from langchain_community.embeddings import HuggingFaceEmbeddings
                    kwargs["embeddings"] = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            except Exception:
                pass

        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            **kwargs
        )
        df = result.to_pandas()
        per_question = [
            EvalResult(
                question=str(row["question"]),
                answer=str(row["answer"]),
                contexts=list(row["contexts"]) if isinstance(row["contexts"], (list, tuple)) else [str(row["contexts"])],
                ground_truth=str(row["ground_truth"]),
                faithfulness=float(row.get("faithfulness", 0.0) or 0.0),
                answer_relevancy=float(row.get("answer_relevancy", 0.0) or 0.0),
                context_precision=float(row.get("context_precision", 0.0) or 0.0),
                context_recall=float(row.get("context_recall", 0.0) or 0.0)
            )
            for _, row in df.iterrows()
        ]

        def _get_metric(key):
            val = result.get(key)
            if val is not None and not (isinstance(val, float) and (val != val)):
                return float(val)
            col_val = df[key].mean() if key in df else 0.0
            return float(col_val) if col_val == col_val else 0.0

        return {
            "faithfulness": _get_metric("faithfulness"),
            "answer_relevancy": _get_metric("answer_relevancy"),
            "context_precision": _get_metric("context_precision"),
            "context_recall": _get_metric("context_recall"),
            "per_question": per_question
        }
    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed: {e}")
        per_question = [
            EvalResult(
                question=q,
                answer=a,
                contexts=c,
                ground_truth=gt,
                faithfulness=0.0,
                answer_relevancy=0.0,
                context_precision=0.0,
                context_recall=0.0
            )
            for q, a, c, gt in zip(questions, answers, contexts, ground_truths)
        ]
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "per_question": per_question
        }


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    if not eval_results:
        return []

    diagnostic_tree = {
        "faithfulness": (
            "LLM tự bịa câu trả lời ngoài tài liệu (LLM hallucinating)",
            "Thắt chặt system prompt, giảm nhiệt độ (temperature) về 0 (Tighten prompt, lower temperature)"
        ),
        "context_recall": (
            "Hệ thống tìm kiếm bỏ sót đoạn văn đúng (Missing relevant chunks)",
            "Cải thiện lại bước cắt đoạn hoặc bổ sung từ khóa BM25 (Improve chunking or add BM25)"
        ),
        "context_precision": (
            "Đoạn văn không liên quan bị xếp lên đầu (Too many irrelevant chunks)",
            "Bổ sung tầng Cross-Encoder reranking hoặc lọc theo metadata (Add reranking or metadata filter)"
        ),
        "answer_relevancy": (
            "Câu trả lời bị lệch trọng tâm câu hỏi (Answer doesn't match question)",
            "Viết lại prompt hướng dẫn mô hình trả lời trực tiếp hơn (Improve prompt template)"
        ),
    }

    scored_items = []
    for r in eval_results:
        metrics = {
            "faithfulness": r.faithfulness,
            "answer_relevancy": r.answer_relevancy,
            "context_precision": r.context_precision,
            "context_recall": r.context_recall,
        }
        avg_score = sum(metrics.values()) / len(metrics)
        worst_metric = min(metrics, key=metrics.get)
        diag, fix = diagnostic_tree.get(worst_metric, ("Unknown issue", "Investigate manually"))

        scored_items.append({
            "question": r.question,
            "worst_metric": worst_metric,
            "score": round(float(metrics[worst_metric]), 4),
            "avg_score": round(float(avg_score), 4),
            "diagnosis": diag,
            "suggested_fix": fix,
        })

    scored_items.sort(key=lambda x: x["avg_score"])
    return scored_items[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
