"""LLM-based graph planner specifically for Overcooked environment."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

try:
    import weave
    WEAVE_AVAILABLE = True
except ImportError:
    WEAVE_AVAILABLE = False

from .graph_types import GraphCandidate, Subtask, SubtaskType, TaskGraph
from .prompts import (
    OVERCOOKED_INITIAL_GRAPH_PROMPT,
    OVERCOOKED_CROSSOVER_PROMPT,
    OVERCOOKED_MUTATION_PROMPT,
    OVERCOOKED_FIX_FAILURES_PROMPT,
    OVERCOOKED_NOVEL_GRAPH_PROMPT,
)

CRITIC_PROMPT = """You are a critic evaluating task graphs for the Overcooked cooking game.

A VALID graph for making soup must have these task types:
- get_ingredient: Pick up onion (need 3 total)
- put_in_pot: Place onion in pot (need 3 total)
- wait_cooking: Wait for soup to cook (need at least 1)
- get_dish: Pick up a dish (need at least 1)
- plate_soup: Take soup from pot onto dish (need at least 1)
- serve: Deliver to serving location (need at least 1)

A DEGENERATE graph has:
- Only "navigate" tasks
- Missing critical steps (no serve, no plate_soup, no get_ingredient)
- Less than 6 meaningful tasks

## GRAPH TO EVALUATE
{graph}

## VERDICT
Respond with ONLY valid JSON:
```json
{{"valid": true/false, "reason": "brief explanation"}}
```
"""

logger = logging.getLogger(__name__)


class OvercookedGraphPlanner:
    """
    LLM-based graph generator for Overcooked task decomposition.
    """

    # Overcooked-specific strategies
    STRATEGIES = [
        "role_based",       # One chef ingredients, one plating/serving
        "parallel_soups",   # Both work on separate soups
        "pipeline",         # Assembly line
        "zone_based",       # Each chef owns area
        "helper",           # Main cook + helper
        "alternating",      # Take turns to avoid collision
    ]

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        use_local: bool = False,
        max_retries: int = 5,
        critic_retries: int = 3,
    ):
        self.model = model
        self.temperature = temperature
        self.use_local = use_local
        self.max_retries = max_retries
        self.critic_retries = critic_retries  # Extra retries if critic rejects

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

    def _generate(self, prompt: str, operation: str = "generate") -> str:
        """Generate response from LLM. Weave auto-tracks OpenAI calls."""
        # Wrap with weave.op if available for explicit tracing
        if WEAVE_AVAILABLE and not hasattr(self, '_generate_traced'):
            self._generate_traced = weave.op()(self._generate_impl)

        if hasattr(self, '_generate_traced'):
            return self._generate_traced(prompt, operation)
        return self._generate_impl(prompt, operation)

    def _generate_impl(self, prompt: str, operation: str = "generate") -> str:
        """Actual LLM generation logic."""
        if self.client:
            is_new_model = any(x in self.model for x in ["gpt-4.1", "gpt-5", "o1", "o3"])

            params = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
            }

            if is_new_model:
                params["max_completion_tokens"] = 2048
            else:
                params["max_tokens"] = 2048

            # Weave automatically tracks OpenAI calls when initialized
            response = self.client.chat.completions.create(**params)
            return response.choices[0].message.content
        else:
            outputs = self.llm.generate([prompt], self.sampling_params)
            return outputs[0].outputs[0].text

    def _format_env_state(self, env_state: Dict[str, Any]) -> str:
        """Format Overcooked environment state for prompt."""
        terrain = env_state.get("terrain", [])
        terrain_str = "\n".join(terrain)

        orders = env_state.get("orders", [["onion", "onion", "onion"]])
        orders_str = ", ".join([f"Soup ({', '.join(o)})" for o in orders])

        return {
            "layout": env_state.get("layout", "cramped_room"),
            "terrain": terrain_str,
            "orders": orders_str,
        }

    def _critic_validate(self, graph: TaskGraph) -> bool:
        """Use LLM critic to reject degenerate graphs."""
        graph_str = graph.to_prompt_string()

        # Quick heuristic check first - reject obviously bad graphs
        task_types = [s.task_type.value for s in graph.subtasks.values()]
        navigate_count = task_types.count("navigate")
        if navigate_count > len(task_types) * 0.5:
            logger.warning(f"Critic rejected: >50% navigate tasks ({navigate_count}/{len(task_types)})")
            return False

        # Check for essential task types
        essential = {"get_ingredient", "put_in_pot", "serve"}
        present = set(task_types)
        if not essential.intersection(present):
            logger.warning(f"Critic rejected: missing essential tasks (has: {present})")
            return False

        # LLM critic for more nuanced validation
        prompt = CRITIC_PROMPT.format(graph=graph_str)
        try:
            response = self._generate(prompt, operation="critic")

            # Parse response
            json_match = re.search(r'\{[^}]+\}', response)
            if json_match:
                result = json.loads(json_match.group(0))
                is_valid = result.get("valid", False)
                reason = result.get("reason", "unknown")
                if not is_valid:
                    logger.warning(f"Critic rejected: {reason}")
                return is_valid
        except Exception as e:
            logger.warning(f"Critic failed, using heuristic: {e}")

        # Fallback: accept if heuristics passed
        return True

    def _parse_graph_response(self, response: str, n_agents: int = 2) -> TaskGraph:
        """Parse LLM response into TaskGraph."""
        graph = TaskGraph()

        try:
            # Find JSON block
            json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_match = re.search(r"\{[\s\S]*\}", response)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    raise ValueError("No JSON found in response")

            data = json.loads(json_str)

            for i, subtask_data in enumerate(data.get("subtasks", [])):
                # Handle agent_id - could be string or int
                agent_id = subtask_data.get("agent", 0)
                if isinstance(agent_id, str):
                    agent_id = int(agent_id) if agent_id.isdigit() else 0
                if agent_id >= n_agents:
                    agent_id = agent_id % n_agents

                task_type_str = subtask_data.get("type", "navigate").lower()
                try:
                    task_type = SubtaskType(task_type_str)
                except ValueError:
                    # Default to navigate for unknown types
                    task_type = SubtaskType.NAVIGATE

                # Handle missing id - generate one
                subtask_id = subtask_data.get("id") or subtask_data.get("name") or f"task_{i}"

                subtask = Subtask(
                    id=subtask_id,
                    task_type=task_type,
                    agent_id=agent_id,
                    target=subtask_data.get("target"),
                    dependencies=subtask_data.get("dependencies", []),
                )
                graph.add_subtask(subtask)

            return graph

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"JSON parsing failed: {e}")
            raise

    def generate_initial_graphs(
        self,
        env_state: Dict[str, Any],
        task: str = "Make and serve soup",
        n_candidates: int = 6,
    ) -> List[GraphCandidate]:
        """Generate diverse initial graph candidates for Overcooked."""
        n_agents = env_state.get("n_agents", 2)
        env_info = self._format_env_state(env_state)
        candidates = []

        for i in range(n_candidates):
            strategy = self.STRATEGIES[i % len(self.STRATEGIES)]

            prompt = OVERCOOKED_INITIAL_GRAPH_PROMPT.format(
                layout=env_info["layout"],
                terrain=env_info["terrain"],
                orders=env_info["orders"],
                strategy=strategy,
            )

            for attempt in range(self.max_retries):
                try:
                    response = self._generate(prompt)
                    graph = self._parse_graph_response(response, n_agents)

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
                    logger.info(f"Generated graph with strategy={strategy}, {len(graph)} subtasks")
                    break

                except Exception as e:
                    logger.error(f"Graph generation failed (attempt {attempt + 1}): {e}")
                    if attempt == self.max_retries - 1:
                        fallback = self._create_fallback_graph(env_state, strategy)
                        candidates.append(
                            GraphCandidate(
                                graph=fallback,
                                origin=f"fallback_{strategy}",
                                generation=0,
                            )
                        )

        return candidates

    def _create_fallback_graph(self, env_state: Dict[str, Any], strategy: str) -> TaskGraph:
        """Create fallback graph for Overcooked when LLM fails."""
        graph = TaskGraph()

        if strategy in ["role_based", "pipeline"]:
            # Chef 0: get ingredients, Chef 1: plate and serve
            # Chef 0 gets all 3 onions
            for i in range(3):
                get_id = f"get_onion_{i}"
                put_id = f"put_pot_{i}"
                deps = [f"put_pot_{i-1}"] if i > 0 else []

                graph.add_subtask(Subtask(
                    id=get_id,
                    task_type=SubtaskType.GET_INGREDIENT,
                    agent_id=0,
                    target="onion",
                    dependencies=deps,
                ))
                graph.add_subtask(Subtask(
                    id=put_id,
                    task_type=SubtaskType.PUT_IN_POT,
                    agent_id=0,
                    target="pot_0",
                    dependencies=[get_id] + deps,
                ))

            # Chef 1: wait, get dish, plate, serve
            graph.add_subtask(Subtask(
                id="wait_cook",
                task_type=SubtaskType.WAIT_COOKING,
                agent_id=1,
                target="pot_0",
                dependencies=["put_pot_2"],
            ))
            graph.add_subtask(Subtask(
                id="get_dish",
                task_type=SubtaskType.GET_DISH,
                agent_id=1,
                target="dish",
                dependencies=["wait_cook"],
            ))
            graph.add_subtask(Subtask(
                id="plate_soup",
                task_type=SubtaskType.PLATE_SOUP,
                agent_id=1,
                target="pot_0",
                dependencies=["get_dish"],
            ))
            graph.add_subtask(Subtask(
                id="serve",
                task_type=SubtaskType.SERVE,
                agent_id=1,
                target="serving",
                dependencies=["plate_soup"],
            ))
        else:
            # Default: alternating strategy
            # Both chefs alternate on getting onions
            for i in range(3):
                agent = i % 2
                get_id = f"get_onion_{i}"
                put_id = f"put_pot_{i}"
                deps = [f"put_pot_{i-1}"] if i > 0 else []

                graph.add_subtask(Subtask(
                    id=get_id,
                    task_type=SubtaskType.GET_INGREDIENT,
                    agent_id=agent,
                    target="onion",
                    dependencies=deps,
                ))
                graph.add_subtask(Subtask(
                    id=put_id,
                    task_type=SubtaskType.PUT_IN_POT,
                    agent_id=agent,
                    target="pot_0",
                    dependencies=[get_id] + deps,
                ))

            # Chef 0: wait, plate, serve
            graph.add_subtask(Subtask(
                id="wait_cook",
                task_type=SubtaskType.WAIT_COOKING,
                agent_id=0,
                target="pot_0",
                dependencies=["put_pot_2"],
            ))
            graph.add_subtask(Subtask(
                id="get_dish",
                task_type=SubtaskType.GET_DISH,
                agent_id=1,
                target="dish",
                dependencies=["wait_cook"],
            ))
            graph.add_subtask(Subtask(
                id="plate_soup",
                task_type=SubtaskType.PLATE_SOUP,
                agent_id=0,
                target="pot_0",
                dependencies=["get_dish", "wait_cook"],
            ))
            graph.add_subtask(Subtask(
                id="serve",
                task_type=SubtaskType.SERVE,
                agent_id=0,
                target="serving",
                dependencies=["plate_soup"],
            ))

        return graph

    def crossover(
        self,
        parent1: GraphCandidate,
        parent1_perf: Dict[str, Any],
        parent2: GraphCandidate,
        parent2_perf: Dict[str, Any],
        insights: Dict[str, Any],
    ) -> GraphCandidate:
        """Combine best features from two parent graphs."""
        n_agents = insights.get("n_agents", 2)
        env_state = insights.get("env_state", {})

        prompt = OVERCOOKED_CROSSOVER_PROMPT.format(
            parent1_perf=parent1_perf.get("avg_reward", 0),
            parent1_graph=parent1.graph.to_prompt_string(),
            parent1_strengths=parent1_perf.get("analysis", {}).get("strengths", "None identified"),
            parent2_perf=parent2_perf.get("avg_reward", 0),
            parent2_graph=parent2.graph.to_prompt_string(),
            parent2_strengths=parent2_perf.get("analysis", {}).get("strengths", "None identified"),
            successful_patterns=insights.get("successful_patterns", "None yet"),
            failure_patterns=insights.get("failure_patterns", "None yet"),
        )

        total_attempts = self.max_retries + self.critic_retries
        for attempt in range(total_attempts):
            try:
                orig_temp = self.temperature
                if attempt > self.max_retries:
                    self.temperature = min(1.0, self.temperature + 0.1)

                response = self._generate(prompt)
                self.temperature = orig_temp

                graph = self._parse_graph_response(response, n_agents)

                is_valid, errors = graph.validate()
                if not is_valid:
                    logger.warning(f"Crossover invalid (attempt {attempt + 1}): {errors}")
                    continue

                # Critic validation
                if not self._critic_validate(graph):
                    logger.warning(f"Crossover rejected by critic (attempt {attempt + 1}/{total_attempts})")
                    continue

                logger.info(f"Crossover accepted after {attempt + 1} attempts")
                return GraphCandidate(
                    graph=graph,
                    origin="crossover",
                    parent_ids=[parent1.candidate_id, parent2.candidate_id],
                    generation=max(parent1.generation, parent2.generation) + 1,
                )

            except Exception as e:
                logger.error(f"Crossover failed (attempt {attempt + 1}): {e}")

        # Fallback: use known-good fallback graph
        logger.warning(f"Crossover exhausted all {total_attempts} attempts, using fallback")
        fallback = self._create_fallback_graph(env_state, "pipeline")
        return GraphCandidate(
            graph=fallback,
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
        """Make targeted improvements to a graph."""
        n_agents = insights.get("n_agents", 2)
        env_state = insights.get("env_state", {})

        prompt = OVERCOOKED_MUTATION_PROMPT.format(
            parent_perf=parent_perf.get("avg_reward", 0),
            parent_graph=parent.graph.to_prompt_string(),
            failure_analysis=parent_perf.get("analysis", {}).get("failures", "None identified"),
        )

        total_attempts = self.max_retries + self.critic_retries
        for attempt in range(total_attempts):
            try:
                # Increase temperature slightly on retries to get more diverse outputs
                orig_temp = self.temperature
                if attempt > self.max_retries:
                    self.temperature = min(1.0, self.temperature + 0.1)

                response = self._generate(prompt)
                self.temperature = orig_temp

                graph = self._parse_graph_response(response, n_agents)

                is_valid, errors = graph.validate()
                if not is_valid:
                    logger.warning(f"Mutation invalid (attempt {attempt + 1}): {errors}")
                    continue

                # Critic validation - reject degenerate graphs
                if not self._critic_validate(graph):
                    logger.warning(f"Mutation rejected by critic (attempt {attempt + 1}/{total_attempts})")
                    continue

                logger.info(f"Mutation accepted after {attempt + 1} attempts")
                return GraphCandidate(
                    graph=graph,
                    origin="mutation",
                    parent_ids=[parent.candidate_id],
                    generation=parent.generation + 1,
                )

            except Exception as e:
                logger.error(f"Mutation failed (attempt {attempt + 1}): {e}")

        # Fallback: use a known-good fallback graph
        logger.warning(f"Mutation exhausted all {total_attempts} attempts, using fallback")
        fallback = self._create_fallback_graph(env_state, "alternating")
        return GraphCandidate(
            graph=fallback,
            origin="mutation_fallback",
            parent_ids=[parent.candidate_id],
            generation=parent.generation + 1,
        )

    def generate_novel(
        self,
        env_state: Dict[str, Any],
        task: str,
        insights: Dict[str, Any],
    ) -> GraphCandidate:
        """Generate a novel graph informed by learnings."""
        n_agents = env_state.get("n_agents", 2)
        env_info = self._format_env_state(env_state)

        prompt = OVERCOOKED_NOVEL_GRAPH_PROMPT.format(
            layout=env_info["layout"],
            terrain=env_info["terrain"],
            orders=env_info["orders"],
            successful_patterns=insights.get("successful_patterns", "None yet"),
            failure_patterns=insights.get("failure_patterns", "None yet"),
        )

        for attempt in range(self.max_retries):
            try:
                response = self._generate(prompt)
                graph = self._parse_graph_response(response, n_agents)

                is_valid, errors = graph.validate()
                if not is_valid and attempt < self.max_retries - 1:
                    continue

                return GraphCandidate(
                    graph=graph,
                    origin="novel",
                    generation=insights.get("current_generation", 0) + 1,
                )

            except Exception as e:
                logger.error(f"Novel generation failed (attempt {attempt + 1}): {e}")

        # Fallback
        fallback = self._create_fallback_graph(env_state, "alternating")
        return GraphCandidate(
            graph=fallback,
            origin="novel_fallback",
            generation=insights.get("current_generation", 0) + 1,
        )
