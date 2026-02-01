"""LLM-based graph planner for task decomposition."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from .graph_types import GraphCandidate, Subtask, SubtaskType, TaskGraph
from .prompts import (
    CROSSOVER_PROMPT,
    FIX_FAILURES_PROMPT,
    INITIAL_GRAPH_PROMPT,
    MUTATION_PROMPT,
    NOVEL_GRAPH_PROMPT,
)

logger = logging.getLogger(__name__)


class LLMGraphPlanner:
    """
    LLM-based graph generator for RWARE task decomposition.

    Supports both OpenAI API and local models via vLLM.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        use_local: bool = False,
        max_retries: int = 3,
    ):
        """
        Initialize the LLM planner.

        Args:
            model: Model name (OpenAI model or HuggingFace model for local)
            temperature: Sampling temperature
            use_local: Whether to use local vLLM instead of OpenAI API
            max_retries: Number of retries on parsing failures
        """
        self.model = model
        self.temperature = temperature
        self.use_local = use_local
        self.max_retries = max_retries

        if use_local:
            try:
                from vllm import LLM, SamplingParams

                self.llm = LLM(model=model)
                self.sampling_params = SamplingParams(temperature=temperature, max_tokens=2048)
                self.client = None
            except ImportError:
                raise ImportError("vLLM not installed. Install with: pip install vllm")
        else:
            try:
                from openai import OpenAI

                self.client = OpenAI()
                self.llm = None
            except ImportError:
                raise ImportError("OpenAI not installed. Install with: pip install openai")

        # Try to import weave for tracing
        try:
            import weave

            self._weave_available = True
        except ImportError:
            self._weave_available = False
            logger.warning("Weave not available, tracing disabled")

    def _generate(self, prompt: str) -> str:
        """Generate response from LLM."""
        if self.client:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=2048,
            )
            return response.choices[0].message.content
        else:
            outputs = self.llm.generate([prompt], self.sampling_params)
            return outputs[0].outputs[0].text

    def _format_env_state(self, env_state: Dict[str, Any]) -> str:
        """Format environment state for prompt."""
        lines = [
            f"Warehouse size: {env_state.get('grid_size', 'unknown')}",
            f"Number of robots: {env_state.get('n_agents', 'unknown')}",
        ]

        if env_state.get("agent_positions"):
            lines.append(f"Robot positions: {env_state['agent_positions']}")

        if env_state.get("requested_shelves"):
            lines.append(f"Shelves to deliver: {env_state['requested_shelves']}")

        if env_state.get("workstation_position"):
            lines.append(f"Workstation location: {env_state['workstation_position']}")

        return "\n".join(lines)

    def _parse_graph_response(self, response: str, n_agents: int) -> TaskGraph:
        """Parse LLM response into TaskGraph."""
        graph = TaskGraph()

        # Try JSON parsing first
        try:
            # Find JSON block in response
            json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find raw JSON
                json_match = re.search(r"\{[\s\S]*\}", response)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    raise ValueError("No JSON found in response")

            data = json.loads(json_str)

            for subtask_data in data.get("subtasks", []):
                # Validate agent ID
                agent_id = subtask_data.get("agent", 0)
                if agent_id >= n_agents:
                    agent_id = agent_id % n_agents

                # Parse task type
                task_type_str = subtask_data.get("type", "navigate").lower()
                try:
                    task_type = SubtaskType(task_type_str)
                except ValueError:
                    task_type = SubtaskType.NAVIGATE

                subtask = Subtask(
                    id=subtask_data["id"],
                    task_type=task_type,
                    agent_id=agent_id,
                    target=subtask_data.get("target"),
                    dependencies=subtask_data.get("dependencies", []),
                )
                graph.add_subtask(subtask)

            return graph

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"JSON parsing failed: {e}, falling back to regex")

        # Fallback: regex parsing
        pattern = r"-\s*(\w+):\s*Agent_(\d+)\s+(\w+)\s*(\S*)\s*(?:\(after:\s*([^)]+)\))?"
        for match in re.finditer(pattern, response):
            subtask_id = match.group(1)
            agent_id = int(match.group(2))
            task_type_str = match.group(3).lower()
            target = match.group(4) if match.group(4) else None
            deps_str = match.group(5)
            deps = [d.strip() for d in deps_str.split(",")] if deps_str else []

            # Validate agent ID
            if agent_id >= n_agents:
                agent_id = agent_id % n_agents

            # Parse task type
            try:
                task_type = SubtaskType(task_type_str)
            except ValueError:
                task_type = SubtaskType.NAVIGATE

            subtask = Subtask(
                id=subtask_id,
                task_type=task_type,
                agent_id=agent_id,
                target=target,
                dependencies=deps,
            )
            graph.add_subtask(subtask)

        return graph

    def generate_initial_graphs(
        self,
        env_state: Dict[str, Any],
        task: str,
        n_candidates: int = 8,
    ) -> List[GraphCandidate]:
        """
        Generate diverse initial graph candidates.

        Args:
            env_state: Current environment state
            task: Task description
            n_candidates: Number of candidates to generate

        Returns:
            List of GraphCandidate objects
        """
        strategies = [
            "sequential",
            "parallel",
            "pipelined",
            "zone_based",
            "nearest_first",
            "load_balanced",
            "hierarchical",
            "adaptive",
        ]

        n_agents = env_state.get("n_agents", 4)
        candidates = []

        for i in range(n_candidates):
            strategy = strategies[i % len(strategies)]

            prompt = INITIAL_GRAPH_PROMPT.format(
                env_state=self._format_env_state(env_state),
                task=task,
                n_agents=n_agents,
                n_agents_minus_1=n_agents - 1,
                strategy=strategy,
            )

            # Retry loop for parsing failures
            for attempt in range(self.max_retries):
                try:
                    response = self._generate(prompt)
                    graph = self._parse_graph_response(response, n_agents)

                    # Validate graph
                    is_valid, errors = graph.validate()
                    if not is_valid:
                        logger.warning(f"Invalid graph (attempt {attempt + 1}): {errors}")
                        if attempt < self.max_retries - 1:
                            continue

                    candidates.append(
                        GraphCandidate(
                            graph=graph,
                            origin=f"initial_{strategy}",
                            generation=0,
                        )
                    )
                    break

                except Exception as e:
                    logger.error(f"Graph generation failed (attempt {attempt + 1}): {e}")
                    if attempt == self.max_retries - 1:
                        # Create a simple fallback graph
                        fallback = self._create_fallback_graph(env_state, task)
                        candidates.append(
                            GraphCandidate(
                                graph=fallback,
                                origin=f"fallback_{strategy}",
                                generation=0,
                            )
                        )

        return candidates

    def crossover(
        self,
        parent1: GraphCandidate,
        parent1_perf: Dict[str, Any],
        parent2: GraphCandidate,
        parent2_perf: Dict[str, Any],
        insights: Dict[str, Any],
    ) -> GraphCandidate:
        """
        Combine best features from two parent graphs.

        Args:
            parent1: First parent candidate
            parent1_perf: Performance metrics for parent1
            parent2: Second parent candidate
            parent2_perf: Performance metrics for parent2
            insights: Aggregate insights from all candidates

        Returns:
            New GraphCandidate from crossover
        """
        n_agents = insights.get("n_agents", 4)

        prompt = CROSSOVER_PROMPT.format(
            parent1_graph=parent1.graph.to_prompt_string(),
            parent1_perf=parent1_perf.get("success_rate", 0),
            parent1_strengths=parent1_perf.get("strengths", "N/A"),
            parent2_graph=parent2.graph.to_prompt_string(),
            parent2_perf=parent2_perf.get("success_rate", 0),
            parent2_strengths=parent2_perf.get("strengths", "N/A"),
            successful_patterns=insights.get("successful_patterns", "N/A"),
            failure_patterns=insights.get("failure_patterns", "N/A"),
        )

        for attempt in range(self.max_retries):
            try:
                response = self._generate(prompt)
                graph = self._parse_graph_response(response, n_agents)

                is_valid, errors = graph.validate()
                if not is_valid and attempt < self.max_retries - 1:
                    logger.warning(f"Invalid crossover graph: {errors}")
                    continue

                return GraphCandidate(
                    graph=graph,
                    origin="crossover",
                    parent_ids=[parent1.candidate_id, parent2.candidate_id],
                    generation=max(parent1.generation, parent2.generation) + 1,
                )

            except Exception as e:
                logger.error(f"Crossover failed (attempt {attempt + 1}): {e}")

        # Fallback: return copy of better parent
        better_parent = parent1 if parent1_perf.get("success_rate", 0) >= parent2_perf.get("success_rate", 0) else parent2
        return GraphCandidate(
            graph=TaskGraph.from_dict(better_parent.graph.to_dict()),
            origin="crossover_fallback",
            parent_ids=[parent1.candidate_id, parent2.candidate_id],
            generation=max(parent1.generation, parent2.generation) + 1,
        )

    def mutate(
        self,
        parent: GraphCandidate,
        parent_perf: Dict[str, Any],
        insights: Dict[str, Any],
    ) -> GraphCandidate:
        """
        Make targeted improvements to a graph.

        Args:
            parent: Parent candidate to mutate
            parent_perf: Performance metrics for parent
            insights: Aggregate insights from all candidates

        Returns:
            New mutated GraphCandidate
        """
        n_agents = insights.get("n_agents", 4)

        failure_analysis = parent_perf.get("failure_analysis", {})
        failure_str = "\n".join([
            f"- Failure points: {failure_analysis.get('common_failure_points', [])}",
            f"- Coordination issues: {failure_analysis.get('coordination_failures', [])}",
            f"- Bottlenecks: {failure_analysis.get('bottleneck_subtasks', [])}",
        ])

        prompt = MUTATION_PROMPT.format(
            parent_graph=parent.graph.to_prompt_string(),
            parent_perf=parent_perf.get("success_rate", 0),
            failure_analysis=failure_str,
            successful_patterns=insights.get("successful_patterns", "N/A"),
        )

        for attempt in range(self.max_retries):
            try:
                response = self._generate(prompt)
                graph = self._parse_graph_response(response, n_agents)

                is_valid, errors = graph.validate()
                if not is_valid and attempt < self.max_retries - 1:
                    logger.warning(f"Invalid mutation graph: {errors}")
                    continue

                return GraphCandidate(
                    graph=graph,
                    origin="mutation",
                    parent_ids=[parent.candidate_id],
                    generation=parent.generation + 1,
                )

            except Exception as e:
                logger.error(f"Mutation failed (attempt {attempt + 1}): {e}")

        # Fallback: return copy of parent
        return GraphCandidate(
            graph=TaskGraph.from_dict(parent.graph.to_dict()),
            origin="mutation_fallback",
            parent_ids=[parent.candidate_id],
            generation=parent.generation + 1,
        )

    def fix_failures(
        self,
        failing_candidate: GraphCandidate,
        failure_analysis: Dict[str, Any],
        successful_patterns: str,
    ) -> GraphCandidate:
        """
        Aggressively fix a poorly performing graph.

        Args:
            failing_candidate: The poorly performing candidate
            failure_analysis: Detailed failure analysis
            successful_patterns: Patterns that work in other graphs

        Returns:
            New fixed GraphCandidate
        """
        n_agents = failure_analysis.get("n_agents", 4)

        prompt = FIX_FAILURES_PROMPT.format(
            failing_graph=failing_candidate.graph.to_prompt_string(),
            failure_points=failure_analysis.get("common_failure_points", []),
            coordination_failures=failure_analysis.get("coordination_failures", []),
            bottlenecks=failure_analysis.get("bottleneck_subtasks", []),
            successful_patterns=successful_patterns,
        )

        for attempt in range(self.max_retries):
            try:
                response = self._generate(prompt)
                graph = self._parse_graph_response(response, n_agents)

                is_valid, errors = graph.validate()
                if not is_valid and attempt < self.max_retries - 1:
                    logger.warning(f"Invalid fixed graph: {errors}")
                    continue

                return GraphCandidate(
                    graph=graph,
                    origin="fix_failures",
                    parent_ids=[failing_candidate.candidate_id],
                    generation=failing_candidate.generation + 1,
                )

            except Exception as e:
                logger.error(f"Fix failures failed (attempt {attempt + 1}): {e}")

        # Fallback
        return GraphCandidate(
            graph=TaskGraph.from_dict(failing_candidate.graph.to_dict()),
            origin="fix_failures_fallback",
            parent_ids=[failing_candidate.candidate_id],
            generation=failing_candidate.generation + 1,
        )

    def generate_novel(
        self,
        env_state: Dict[str, Any],
        task: str,
        insights: Dict[str, Any],
    ) -> GraphCandidate:
        """
        Generate a novel graph informed by all learnings.

        Args:
            env_state: Current environment state
            task: Task description
            insights: Aggregate insights from all generations

        Returns:
            New novel GraphCandidate
        """
        n_agents = env_state.get("n_agents", 4)

        prompt = NOVEL_GRAPH_PROMPT.format(
            env_state=self._format_env_state(env_state),
            task=task,
            successful_patterns=insights.get("successful_patterns", "N/A"),
            failure_patterns=insights.get("failure_patterns", "N/A"),
            generation_scores=insights.get("generation_best_scores", []),
        )

        for attempt in range(self.max_retries):
            try:
                response = self._generate(prompt)
                graph = self._parse_graph_response(response, n_agents)

                is_valid, errors = graph.validate()
                if not is_valid and attempt < self.max_retries - 1:
                    logger.warning(f"Invalid novel graph: {errors}")
                    continue

                return GraphCandidate(
                    graph=graph,
                    origin="novel",
                    generation=insights.get("current_generation", 0) + 1,
                )

            except Exception as e:
                logger.error(f"Novel generation failed (attempt {attempt + 1}): {e}")

        # Fallback
        fallback = self._create_fallback_graph(env_state, task)
        return GraphCandidate(
            graph=fallback,
            origin="novel_fallback",
            generation=insights.get("current_generation", 0) + 1,
        )

    def _create_fallback_graph(self, env_state: Dict[str, Any], task: str) -> TaskGraph:
        """Create a simple fallback graph when LLM fails."""
        graph = TaskGraph()
        n_agents = env_state.get("n_agents", 4)
        requested_shelves = env_state.get("requested_shelves", [f"shelf_{i}" for i in range(n_agents)])

        # Simple round-robin assignment
        for i, shelf in enumerate(requested_shelves):
            agent_id = i % n_agents

            fetch_id = f"fetch_{shelf}"
            deliver_id = f"deliver_{shelf}"

            # Add fetch subtask
            graph.add_subtask(
                Subtask(
                    id=fetch_id,
                    task_type=SubtaskType.FETCH,
                    agent_id=agent_id,
                    target=shelf,
                    dependencies=[],
                )
            )

            # Add deliver subtask (depends on fetch)
            graph.add_subtask(
                Subtask(
                    id=deliver_id,
                    task_type=SubtaskType.DELIVER,
                    agent_id=agent_id,
                    target="workstation",
                    dependencies=[fetch_id],
                )
            )

        return graph
