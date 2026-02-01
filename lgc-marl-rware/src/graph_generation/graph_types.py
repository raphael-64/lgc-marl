"""Graph data structures for task decomposition."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

import networkx as nx


class SubtaskType(Enum):
    """Types of subtasks in multi-agent environments."""

    # RWARE subtasks
    FETCH = "fetch"  # Go to shelf and pick up
    DELIVER = "deliver"  # Bring shelf to workstation
    NAVIGATE = "navigate"  # Move to position
    WAIT = "wait"  # Wait for dependency
    COORDINATE = "coordinate"  # Sync with other agent
    RETURN = "return"  # Return shelf to original position

    # Overcooked subtasks
    GET_INGREDIENT = "get_ingredient"  # Pick up onion/tomato from dispenser
    PUT_IN_POT = "put_in_pot"  # Place ingredient in pot
    WAIT_COOKING = "wait_cooking"  # Wait for soup to cook
    GET_DISH = "get_dish"  # Pick up dish from dispenser
    PLATE_SOUP = "plate_soup"  # Get soup from pot onto dish
    SERVE = "serve"  # Deliver dish to serving location
    PUT_ON_COUNTER = "put_on_counter"  # Place item on counter
    GET_FROM_COUNTER = "get_from_counter"  # Pick up item from counter


@dataclass
class Subtask:
    """A single subtask in the task decomposition graph."""

    id: str
    task_type: SubtaskType
    agent_id: int
    target: Optional[str] = None  # shelf_id, position, etc.
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "type": self.task_type.value,
            "agent": self.agent_id,
            "target": self.target,
            "dependencies": self.dependencies,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Subtask:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            task_type=SubtaskType(data["type"]),
            agent_id=data["agent"],
            target=data.get("target"),
            dependencies=data.get("dependencies", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class TaskGraph:
    """
    Directed acyclic graph representing task decomposition.

    Nodes = subtasks
    Edges = dependencies (must complete before)
    """

    subtasks: Dict[str, Subtask] = field(default_factory=dict)

    def add_subtask(self, subtask: Subtask) -> None:
        """Add a subtask to the graph."""
        self.subtasks[subtask.id] = subtask

    def remove_subtask(self, subtask_id: str) -> None:
        """Remove a subtask and update dependencies."""
        if subtask_id in self.subtasks:
            del self.subtasks[subtask_id]
            # Remove from other subtasks' dependencies
            for subtask in self.subtasks.values():
                if subtask_id in subtask.dependencies:
                    subtask.dependencies.remove(subtask_id)

    def to_networkx(self) -> nx.DiGraph:
        """Convert to NetworkX directed graph."""
        G = nx.DiGraph()
        for subtask_id, subtask in self.subtasks.items():
            G.add_node(
                subtask_id,
                task_type=subtask.task_type.value,
                agent_id=subtask.agent_id,
                target=subtask.target,
            )
            for dep in subtask.dependencies:
                if dep in self.subtasks:  # Only add edge if dependency exists
                    G.add_edge(dep, subtask_id)
        return G

    def get_ready_subtasks(self, completed: Set[str]) -> List[Subtask]:
        """Get subtasks whose dependencies are all completed."""
        ready = []
        for subtask_id, subtask in self.subtasks.items():
            if subtask_id not in completed:
                if all(dep in completed for dep in subtask.dependencies):
                    ready.append(subtask)
        return ready

    def get_subtasks_for_agent(self, agent_id: int) -> List[Subtask]:
        """Get all subtasks assigned to an agent."""
        return [s for s in self.subtasks.values() if s.agent_id == agent_id]

    def validate(self) -> tuple[bool, List[str]]:
        """
        Check graph validity.

        Returns:
            (is_valid, list of error messages)
        """
        errors = []

        # Check for empty graph
        if not self.subtasks:
            errors.append("Graph has no subtasks")
            return False, errors

        G = self.to_networkx()

        # Check for cycles
        if not nx.is_directed_acyclic_graph(G):
            cycles = list(nx.simple_cycles(G))
            errors.append(f"Graph contains cycles: {cycles}")

        # Check for missing dependencies
        for subtask in self.subtasks.values():
            for dep in subtask.dependencies:
                if dep not in self.subtasks:
                    errors.append(f"Subtask {subtask.id} has missing dependency: {dep}")

        # Note: Disconnected components are OK for parallel execution
        # (each agent can work independently)

        return len(errors) == 0, errors

    def get_parallelism_score(self) -> float:
        """
        Measure how parallelizable the graph is.

        Returns ratio of total tasks to critical path length.
        Higher = more parallel.
        """
        if not self.subtasks:
            return 0.0

        G = self.to_networkx()

        try:
            critical_path = nx.dag_longest_path_length(G)
            total_tasks = len(self.subtasks)
            return total_tasks / (critical_path + 1)
        except nx.NetworkXError:
            return 1.0  # If not a DAG, return 1.0

    def get_agent_load_balance(self) -> Dict[int, int]:
        """Get number of subtasks per agent."""
        load = {}
        for subtask in self.subtasks.values():
            load[subtask.agent_id] = load.get(subtask.agent_id, 0) + 1
        return load

    def get_critical_path(self) -> List[str]:
        """Get the critical path (longest path) through the graph."""
        G = self.to_networkx()
        try:
            return nx.dag_longest_path(G)
        except nx.NetworkXError:
            return []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "subtasks": [s.to_dict() for s in self.subtasks.values()],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TaskGraph:
        """Create from dictionary."""
        graph = cls()
        for subtask_data in data.get("subtasks", []):
            subtask = Subtask.from_dict(subtask_data)
            graph.add_subtask(subtask)
        return graph

    def to_prompt_string(self) -> str:
        """Convert to string for LLM prompts."""
        if not self.subtasks:
            return "(empty graph)"

        lines = []

        # Sort by topological order if possible
        G = self.to_networkx()
        try:
            order = list(nx.topological_sort(G))
        except nx.NetworkXError:
            order = list(self.subtasks.keys())

        for subtask_id in order:
            subtask = self.subtasks[subtask_id]
            deps = f" (after: {', '.join(subtask.dependencies)})" if subtask.dependencies else ""
            target = f" {subtask.target}" if subtask.target else ""
            lines.append(
                f"- {subtask_id}: Agent_{subtask.agent_id} {subtask.task_type.value}{target}{deps}"
            )

        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.subtasks)

    def __repr__(self) -> str:
        return f"TaskGraph(n_subtasks={len(self.subtasks)})"


@dataclass
class GraphCandidate:
    """A graph candidate with metadata for evolution."""

    graph: TaskGraph
    origin: str = "initial"  # initial, crossover, mutation, novel, fix_failures
    parent_ids: List[str] = field(default_factory=list)
    generation: int = 0
    candidate_id: str = ""

    # Filled after evaluation
    performance: Optional[Dict[str, Any]] = None
    trajectory: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        if not self.candidate_id:
            import uuid

            self.candidate_id = str(uuid.uuid4())[:8]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "candidate_id": self.candidate_id,
            "graph": self.graph.to_dict(),
            "origin": self.origin,
            "parent_ids": self.parent_ids,
            "generation": self.generation,
            "performance": self.performance,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GraphCandidate:
        """Create from dictionary."""
        return cls(
            candidate_id=data["candidate_id"],
            graph=TaskGraph.from_dict(data["graph"]),
            origin=data["origin"],
            parent_ids=data.get("parent_ids", []),
            generation=data.get("generation", 0),
            performance=data.get("performance"),
        )

    def __repr__(self) -> str:
        perf_str = ""
        if self.performance:
            perf_str = f", success_rate={self.performance.get('success_rate', 0):.2%}"
        return f"GraphCandidate(id={self.candidate_id}, origin={self.origin}, gen={self.generation}{perf_str})"
