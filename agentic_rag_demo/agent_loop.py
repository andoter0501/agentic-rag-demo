from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


class ActionType(Enum):
    RETRIEVE = "retrieve"
    CALCULATE = "calculate"
    LOOKUP = "lookup"
    GENERATE = "generate"
    REFLECT = "reflect"
    ASK_CLARIFICATION = "ask_clarification"
    FINISH = "finish"


@dataclass
class Action:
    action_type: ActionType
    thought: str
    tool: str | None = None
    input_data: dict[str, Any] = field(default_factory=dict)
    output_data: Any = None
    success: bool = True
    error: str | None = None


@dataclass
class PlanStep:
    step_id: int
    description: str
    expected_output: str
    completed: bool = False
    actual_output: Any = None


@dataclass
class AgentState:
    question: str
    plan: list[PlanStep] = field(default_factory=list)
    actions: list[Action] = field(default_factory=list)
    evidence: list[Any] = field(default_factory=list)
    reflections: list[str] = field(default_factory=list)
    iterations: int = 0
    max_iterations: int = 5
    finished: bool = False
    final_answer: str | None = None


class AgentLoop:
    def __init__(
        self,
        retriever: Any,
        reranker: Any | None = None,
        llm: Any | None = None,
        max_iterations: int = 5,
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm
        self.max_iterations = max_iterations

    def run(self, question: str, context: list[Any] | None = None) -> AgentState:
        state = AgentState(question=question)
        state.evidence = context or []

        self._planning(state)

        while not state.finished and state.iterations < state.max_iterations:
            state.iterations += 1
            action = self._decide_next_action(state)

            self._execute_action(state, action)

            self._reflect(state)

            if action.action_type == ActionType.FINISH:
                state.finished = True

        if not state.finished:
            self._force_finish(state)

        return state

    def _planning(self, state: AgentState) -> None:
        if not self.llm:
            state.plan = [
                PlanStep(1, "理解问题意图", "清晰的意图描述"),
                PlanStep(2, "检索相关证据", "检索结果列表"),
                PlanStep(3, "评估证据质量", "高质量证据子集"),
                PlanStep(4, "生成最终答案", "完整回答"),
            ]
            return

        planning_prompt = self._build_planning_prompt(state.question)
        try:
            plan_text = self.llm.generate(planning_prompt, state.evidence)
            state.plan = self._parse_plan(plan_text)
        except Exception:
            state.plan = [
                PlanStep(1, "理解问题意图", "清晰的意图描述"),
                PlanStep(2, "检索相关证据", "检索结果列表"),
                PlanStep(3, "评估证据质量", "高质量证据子集"),
                PlanStep(4, "生成最终答案", "完整回答"),
            ]

    def _decide_next_action(self, state: AgentState) -> Action:
        remaining_steps = [s for s in state.plan if not s.completed]

        if not remaining_steps:
            return Action(
                action_type=ActionType.FINISH,
                thought="所有计划步骤已完成",
            )

        next_step = remaining_steps[0]

        if "检索" in next_step.description:
            return Action(
                action_type=ActionType.RETRIEVE,
                thought=f"执行步骤: {next_step.description}",
                tool="retriever",
                input_data={"query": state.question},
            )
        elif "评估" in next_step.description or "反思" in next_step.description:
            return Action(
                action_type=ActionType.REFLECT,
                thought=f"执行步骤: {next_step.description}",
                tool="reflection",
            )
        elif "生成" in next_step.description or "回答" in next_step.description:
            return Action(
                action_type=ActionType.GENERATE,
                thought=f"执行步骤: {next_step.description}",
                tool="llm",
                input_data={"question": state.question},
            )
        elif "澄清" in next_step.description:
            return Action(
                action_type=ActionType.ASK_CLARIFICATION,
                thought=f"执行步骤: {next_step.description}",
            )
        else:
            return Action(
                action_type=ActionType.RETRIEVE,
                thought=f"执行步骤: {next_step.description}",
                tool="retriever",
                input_data={"query": state.question},
            )

    def _execute_action(self, state: AgentState, action: Action) -> None:
        try:
            if action.action_type == ActionType.RETRIEVE:
                self._execute_retrieve(state, action)
            elif action.action_type == ActionType.REFLECT:
                self._execute_reflect(state, action)
            elif action.action_type == ActionType.GENERATE:
                self._execute_generate(state, action)
            elif action.action_type == ActionType.ASK_CLARIFICATION:
                self._execute_ask_clarification(state, action)
            elif action.action_type == ActionType.FINISH:
                pass

            if action.success:
                self._mark_step_completed(state)

        except Exception as exc:
            action.success = False
            action.error = str(exc)
            state.actions.append(action)

    def _execute_retrieve(self, state: AgentState, action: Action) -> None:
        query = action.input_data.get("query", state.question)

        results = self.retriever.search(query, top_k=5)

        if self.reranker and results:
            reranked = self.reranker.rerank(query, results, top_k=3)
            for r in reranked:
                if r.chunk not in [
                    e.get("chunk") if isinstance(e, dict) else e for e in state.evidence
                ]:
                    state.evidence.append(
                        {"chunk": r.chunk, "rerank_score": r.rerank_score}
                    )
        else:
            for result in results:
                if result.chunk not in [
                    e.get("chunk") if isinstance(e, dict) else e for e in state.evidence
                ]:
                    state.evidence.append(
                        {"chunk": result.chunk, "score": result.score}
                    )

        action.output_data = results
        action.success = True
        state.actions.append(action)

    def _execute_reflect(self, state: AgentState, action: Action) -> None:
        if not self.llm:
            evidence_scores = []
            for e in state.evidence:
                chunk = e.get("chunk") if isinstance(e, dict) else e
                score = e.get("rerank_score", e.get("score", 0.0))
                evidence_scores.append((chunk.chunk_id, score))

            weak_evidence = [cid for cid, score in evidence_scores if score < 0.5]

            if weak_evidence:
                reflection = (
                    f"发现 {len(weak_evidence)} 条弱证据: {weak_evidence}，考虑补充检索"
                )
                state.reflections.append(reflection)
                action.thought = reflection
            else:
                reflection = "证据质量良好，可进入生成阶段"
                state.reflections.append(reflection)
                action.thought = reflection

            action.output_data = state.reflections[-1]
            action.success = True
            state.actions.append(action)
            return

        prompt = self._build_reflection_prompt(state)
        try:
            reflection = self.llm.generate(prompt, [])
            state.reflections.append(reflection)
            action.output_data = reflection
            action.success = True
        except Exception as exc:
            action.success = False
            action.error = str(exc)

        state.actions.append(action)

    def _execute_generate(self, state: AgentState, action: Action) -> None:
        if not self.llm:
            from .llm import LocalHeuristicLLM

            evidence_text = "\n".join(
                f"- {e.get('chunk').chunk_id}: {e.get('chunk').content}"
                for e in state.evidence
                if isinstance(e, dict) and "chunk" in e
            )
            answer = f"基于以下证据回答:\n{evidence_text}\n\n问题: {state.question}"
            state.final_answer = answer
            action.output_data = answer
            action.success = True
            state.actions.append(action)
            return

        from .llm import Evidence

        evidence_list = [
            Evidence(
                chunk_id=e.get("chunk").chunk_id if isinstance(e, dict) else "unknown",
                content=e.get("chunk").content if isinstance(e, dict) else str(e),
                score=e.get("rerank_score", e.get("score", 0.0)),
            )
            for e in state.evidence
            if isinstance(e, dict) and "chunk" in e
        ]

        try:
            answer = self.llm.generate(state.question, evidence_list)
            state.final_answer = answer
            action.output_data = answer
            action.success = True
        except Exception as exc:
            action.success = False
            action.error = str(exc)

        state.actions.append(action)

    def _execute_ask_clarification(self, state: AgentState, action: Action) -> None:
        action.output_data = "请提供更具体的问题或补充关键词以便准确检索"
        action.success = True
        state.actions.append(action)

    def _reflect(self, state: AgentState) -> None:
        if state.iterations >= state.max_iterations:
            if not state.final_answer:
                state.final_answer = "已达到最大迭代次数，无法生成完整回答"
            state.finished = True

        recent_action = state.actions[-1] if state.actions else None

        if recent_action and not recent_action.success:
            if state.iterations < state.max_iterations:
                pass

    def _mark_step_completed(self, state: AgentState) -> None:
        for step in state.plan:
            if not step.completed:
                step.completed = True
                step.actual_output = (
                    state.evidence
                    if step.description == "检索相关证据"
                    else state.final_answer
                )
                break

    def _force_finish(self, state: AgentState) -> None:
        if not state.final_answer:
            evidence_parts = []
            for e in state.evidence:
                if isinstance(e, dict) and "chunk" in e:
                    evidence_parts.append(e["chunk"].content)
                elif hasattr(e, "content"):
                    evidence_parts.append(e.content)

            if evidence_parts:
                state.final_answer = "基于检索到的证据: " + " ".join(evidence_parts[:3])
            else:
                state.final_answer = "无法生成回答，证据不足"

        state.finished = True

    def _build_planning_prompt(self, question: str) -> str:
        return (
            f"问题: {question}\n\n"
            "请分析这个问题并制定解决步骤。考虑是否需要: "
            "1) 检索外部知识 2) 计算 3) 澄清问题\n"
            "返回格式: 步骤1: xxx | 步骤2: xxx | ..."
        )

    def _build_reflection_prompt(self, state: AgentState) -> str:
        evidence_summary = "\n".join(
            f"- {e.get('chunk').chunk_id}: {e.get('chunk').content[:100]}..."
            for e in state.evidence[:5]
            if isinstance(e, dict)
        )
        return (
            f"问题: {state.question}\n\n"
            f"当前证据:\n{evidence_summary}\n\n"
            "请评估: 1) 证据是否足够回答问题 2) 是否需要补充检索 3) 回答质量预期\n"
            "返回: 足够/不足 + 原因"
        )

    def _parse_plan(self, plan_text: str) -> list[PlanStep]:
        steps: list[PlanStep] = []
        lines = plan_text.split("|")

        for i, line in enumerate(lines, start=1):
            line = line.strip()
            if line:
                desc = re.sub(r"^步骤?\d+[.:]", "", line).strip()
                steps.append(
                    PlanStep(
                        step_id=i,
                        description=desc or f"步骤 {i}",
                        expected_output="待确定",
                    )
                )

        return (
            steps
            if steps
            else [
                PlanStep(1, "理解问题意图", "清晰的意图描述"),
                PlanStep(2, "检索相关证据", "检索结果列表"),
                PlanStep(3, "评估证据质量", "高质量证据子集"),
                PlanStep(4, "生成最终答案", "完整回答"),
            ]
        )


def build_agent_loop(
    retriever: Any,
    reranker: Any | None = None,
    llm: Any | None = None,
    max_iterations: int = 5,
) -> AgentLoop:
    return AgentLoop(
        retriever=retriever,
        reranker=reranker,
        llm=llm,
        max_iterations=max_iterations,
    )
