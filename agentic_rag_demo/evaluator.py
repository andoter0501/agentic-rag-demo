from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class EvaluationMetrics:
    hit_rate: float = 0.0
    mrr: float = 0.0
    ndcg: float = 0.0
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_relevancy: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "hit_rate": self.hit_rate,
            "mrr": self.mrr,
            "ndcg": self.ndcg,
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_relevancy": self.context_relevancy,
        }

    def summary(self) -> str:
        return (
            f"Hit Rate: {self.hit_rate:.2%} | "
            f"MRR: {self.mrr:.3f} | "
            f"NDCG: {self.ndcg:.3f} | "
            f"Faithfulness: {self.faithfulness:.2%} | "
            f"Answer Relevancy: {self.answer_relevancy:.2%}"
        )


@dataclass
class RetrievalMetrics:
    retrieved_ids: list[str] = field(default_factory=list)
    relevant_ids: list[str] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)

    @property
    def precision_at_k(self) -> float:
        if not self.retrieved_ids:
            return 0.0
        k = len(self.retrieved_ids)
        relevant = len(set(self.retrieved_ids) & set(self.relevant_ids))
        return relevant / k

    @property
    def recall_at_k(self) -> float:
        if not self.relevant_ids:
            return 0.0
        relevant = len(set(self.retrieved_ids) & set(self.relevant_ids))
        return relevant / len(self.relevant_ids)

    @property
    def hit_at_k(self) -> bool:
        return len(set(self.retrieved_ids) & set(self.relevant_ids)) > 0

    @property
    def mrr(self) -> float:
        for i, doc_id in enumerate(self.retrieved_ids, start=1):
            if doc_id in self.relevant_ids:
                return 1.0 / i
        return 0.0

    @property
    def ndcg_at_k(self) -> float:
        k = len(self.retrieved_ids)
        dcg = 0.0
        for i, doc_id in enumerate(self.retrieved_ids, start=1):
            if doc_id in self.relevant_ids:
                dcg += 1.0 / (i.bit_length())
            else:
                dcg += 0.0

        idcg = sum(
            1.0 / (i.bit_length()) for i in range(1, min(k, len(self.relevant_ids)) + 1)
        )
        return dcg / idcg if idcg > 0 else 0.0


@dataclass
class GenerationMetrics:
    answer: str
    reference: str
    evidence_chunks: list[str]

    @property
    def answer_length(self) -> int:
        return len(self.answer)

    @property
    def word_count(self) -> int:
        return len(self.answer.split())

    def faithfulness_score(self) -> float:
        evidence_words = set(" ".join(self.evidence_chunks).lower().split())
        answer_words = set(self.answer.lower().split())
        common = evidence_words & answer_words
        if not answer_words:
            return 0.0
        return len(common) / len(answer_words)

    def citation_coverage(self) -> float:
        if not self.evidence_chunks:
            return 0.0
        cited = sum(1 for chunk in self.evidence_chunks if chunk in self.answer)
        return cited / len(self.evidence_chunks)


class Evaluator:
    def __init__(
        self,
        llm: Any | None = None,
        use_llm_judgment: bool = False,
    ) -> None:
        self.llm = llm
        self.use_llm_judgment = use_llm_judgment

    def evaluate_retrieval(
        self,
        retrieved_chunks: list[str],
        relevant_chunks: list[str],
        scores: list[float] | None = None,
    ) -> RetrievalMetrics:
        return RetrievalMetrics(
            retrieved_ids=retrieved_chunks,
            relevant_ids=relevant_chunks,
            scores=scores or [],
        )

    def evaluate_generation(
        self,
        answer: str,
        question: str,
        evidence_chunks: list[str],
        reference_answer: str | None = None,
    ) -> GenerationMetrics:
        return GenerationMetrics(
            answer=answer,
            reference=reference_answer or "",
            evidence_chunks=evidence_chunks,
        )

    def evaluate_rag(
        self,
        question: str,
        answer: str,
        retrieved_chunks: list[str],
        relevant_chunks: list[str],
        evidence_chunks: list[str],
        reference_answer: str | None = None,
    ) -> EvaluationMetrics:
        retrieval_metrics = self.evaluate_retrieval(
            retrieved_chunks=retrieved_chunks,
            relevant_chunks=relevant_chunks,
        )

        generation_metrics = self.evaluate_generation(
            answer=answer,
            question=question,
            evidence_chunks=evidence_chunks,
            reference_answer=reference_answer,
        )

        hit_rate = 1.0 if retrieval_metrics.hit_at_k else 0.0
        mrr = retrieval_metrics.mrr
        ndcg = retrieval_metrics.ndcg_at_k
        faithfulness = generation_metrics.faithfulness_score()
        answer_relevancy = self._estimate_answer_relevancy(question, answer)
        context_relevancy = retrieval_metrics.precision_at_k

        return EvaluationMetrics(
            hit_rate=hit_rate,
            mrr=mrr,
            ndcg=ndcg,
            faithfulness=faithfulness,
            answer_relevancy=answer_relevancy,
            context_relevancy=context_relevancy,
        )

    def _estimate_answer_relevancy(self, question: str, answer: str) -> float:
        if not answer or not question:
            return 0.0

        if self.llm and self.use_llm_judgment:
            return self._llm_judged_relevancy(question, answer)

        q_words = set(question.lower().split())
        a_words = set(answer.lower().split())

        overlap = q_words & a_words
        if not q_words:
            return 0.0

        return len(overlap) / len(q_words)

    def _llm_judged_relevancy(self, question: str, answer: str) -> float:
        if not self.llm:
            return 0.0

        prompt = (
            f"问题: {question}\n\n"
            f"回答: {answer}\n\n"
            "请评估回答与问题的相关性(0-1分)。考虑: "
            "1) 回答是否针对问题 2) 回答是否完整 3) 是否有无关内容\n"
            "返回格式: 只需返回数字分数(如 0.85)"
        )

        try:
            result = self.llm.generate(prompt, [])
            score = float(result.strip())
            return max(0.0, min(1.0, score))
        except Exception:
            return 0.5


class BatchEvaluator:
    def __init__(self, evaluator: Evaluator | None = None) -> None:
        self.evaluator = evaluator or Evaluator()

    def evaluate_batch(
        self,
        test_cases: list[dict[str, Any]],
    ) -> dict[str, Any]:
        results: list[EvaluationMetrics] = []
        detailed_results: list[dict[str, Any]] = []

        for case in test_cases:
            metrics = self.evaluator.evaluate_rag(
                question=case.get("question", ""),
                answer=case.get("answer", ""),
                retrieved_chunks=case.get("retrieved_chunks", []),
                relevant_chunks=case.get("relevant_chunks", []),
                evidence_chunks=case.get("evidence_chunks", []),
                reference_answer=case.get("reference_answer"),
            )
            results.append(metrics)
            detailed_results.append(
                {
                    "question": case.get("question", ""),
                    "metrics": metrics.to_dict(),
                }
            )

        avg_metrics = self._compute_averages(results)

        return {
            "total_cases": len(results),
            "average_metrics": avg_metrics.to_dict(),
            "detailed_results": detailed_results,
        }

    def _compute_averages(self, results: list[EvaluationMetrics]) -> EvaluationMetrics:
        if not results:
            return EvaluationMetrics()

        n = len(results)
        return EvaluationMetrics(
            hit_rate=sum(r.hit_rate for r in results) / n,
            mrr=sum(r.mrr for r in results) / n,
            ndcg=sum(r.ndcg for r in results) / n,
            faithfulness=sum(r.faithfulness for r in results) / n,
            answer_relevancy=sum(r.answer_relevancy for r in results) / n,
            context_relevancy=sum(r.context_relevancy for r in results) / n,
        )

    def print_summary(self, results: dict[str, Any]) -> None:
        print("\n" + "=" * 60)
        print("RAG 评估结果摘要")
        print("=" * 60)

        avg = results.get("average_metrics", {})
        print(f"\n总测试用例: {results.get('total_cases', 0)}")
        print(f"\n平均指标:")
        print(f"  Hit Rate:      {avg.get('hit_rate', 0):.2%}")
        print(f"  MRR:           {avg.get('mrr', 0):.4f}")
        print(f"  NDCG:          {avg.get('ndcg', 0):.4f}")
        print(f"  Faithfulness:  {avg.get('faithfulness', 0):.2%}")
        print(f"  Answer relevancy: {avg.get('answer_relevancy', 0):.2%}")
        print(f"  Context relevancy: {avg.get('context_relevancy', 0):.2%}")

        print("\n详细结果:")
        for detail in results.get("detailed_results", [])[:5]:
            q = detail.get("question", "")[:50]
            m = detail.get("metrics", {})
            print(f"\n  Q: {q}...")
            print(
                f"    Hit: {m.get('hit_rate', 0):.2%} | MRR: {m.get('mrr', 0):.3f} | Faith: {m.get('faithfulness', 0):.2%}"
            )

        print("\n" + "=" * 60)


def create_test_cases_from_answers(
    questions: list[str],
    answers: list[str],
    retrieved_chunks: list[list[str]],
    relevant_chunks: list[list[str]] | None = None,
) -> list[dict[str, Any]]:
    if len(questions) != len(answers) or len(questions) != len(retrieved_chunks):
        raise ValueError("所有列表长度必须一致")

    test_cases: list[dict[str, Any]] = []

    for q, a, retrieved in zip(questions, answers, retrieved_chunks):
        test_cases.append(
            {
                "question": q,
                "answer": a,
                "retrieved_chunks": retrieved,
                "relevant_chunks": relevant_chunks[i] if relevant_chunks else retrieved,
                "evidence_chunks": retrieved,
            }
        )

    return test_cases
