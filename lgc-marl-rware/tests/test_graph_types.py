"""Tests for graph types."""

import pytest
from src.graph_generation.graph_types import (
    GraphCandidate,
    Subtask,
    SubtaskType,
    TaskGraph,
)


class TestSubtask:
    def test_create_subtask(self):
        subtask = Subtask(
            id="fetch_1",
            task_type=SubtaskType.FETCH,
            agent_id=0,
            target="shelf_1",
            dependencies=[],
        )
        assert subtask.id == "fetch_1"
        assert subtask.task_type == SubtaskType.FETCH
        assert subtask.agent_id == 0
        assert subtask.target == "shelf_1"
        assert subtask.dependencies == []

    def test_subtask_to_dict(self):
        subtask = Subtask(
            id="deliver_1",
            task_type=SubtaskType.DELIVER,
            agent_id=1,
            target="workstation",
            dependencies=["fetch_1"],
        )
        d = subtask.to_dict()
        assert d["id"] == "deliver_1"
        assert d["type"] == "deliver"
        assert d["agent"] == 1
        assert d["dependencies"] == ["fetch_1"]

    def test_subtask_from_dict(self):
        d = {
            "id": "nav_1",
            "type": "navigate",
            "agent": 2,
            "target": "pos_1",
            "dependencies": ["fetch_1", "fetch_2"],
        }
        subtask = Subtask.from_dict(d)
        assert subtask.id == "nav_1"
        assert subtask.task_type == SubtaskType.NAVIGATE
        assert subtask.agent_id == 2
        assert len(subtask.dependencies) == 2


class TestTaskGraph:
    def test_create_empty_graph(self):
        graph = TaskGraph()
        assert len(graph) == 0

    def test_add_subtask(self):
        graph = TaskGraph()
        subtask = Subtask(
            id="fetch_1",
            task_type=SubtaskType.FETCH,
            agent_id=0,
            target="shelf_1",
        )
        graph.add_subtask(subtask)
        assert len(graph) == 1
        assert "fetch_1" in graph.subtasks

    def test_remove_subtask(self):
        graph = TaskGraph()
        graph.add_subtask(
            Subtask(id="s1", task_type=SubtaskType.FETCH, agent_id=0)
        )
        graph.add_subtask(
            Subtask(id="s2", task_type=SubtaskType.DELIVER, agent_id=0, dependencies=["s1"])
        )

        graph.remove_subtask("s1")
        assert len(graph) == 1
        assert "s1" not in graph.subtasks
        # Dependency should also be removed
        assert "s1" not in graph.subtasks["s2"].dependencies

    def test_validate_valid_graph(self):
        graph = TaskGraph()
        graph.add_subtask(
            Subtask(id="fetch", task_type=SubtaskType.FETCH, agent_id=0)
        )
        graph.add_subtask(
            Subtask(id="deliver", task_type=SubtaskType.DELIVER, agent_id=0, dependencies=["fetch"])
        )

        is_valid, errors = graph.validate()
        assert is_valid
        assert len(errors) == 0

    def test_validate_empty_graph(self):
        graph = TaskGraph()
        is_valid, errors = graph.validate()
        assert not is_valid
        assert "no subtasks" in errors[0].lower()

    def test_validate_missing_dependency(self):
        graph = TaskGraph()
        graph.add_subtask(
            Subtask(id="deliver", task_type=SubtaskType.DELIVER, agent_id=0, dependencies=["missing"])
        )

        is_valid, errors = graph.validate()
        assert not is_valid
        assert any("missing dependency" in e.lower() for e in errors)

    def test_get_ready_subtasks(self):
        graph = TaskGraph()
        graph.add_subtask(
            Subtask(id="fetch1", task_type=SubtaskType.FETCH, agent_id=0)
        )
        graph.add_subtask(
            Subtask(id="fetch2", task_type=SubtaskType.FETCH, agent_id=1)
        )
        graph.add_subtask(
            Subtask(id="deliver", task_type=SubtaskType.DELIVER, agent_id=0, dependencies=["fetch1"])
        )

        # Initially, fetch1 and fetch2 are ready
        ready = graph.get_ready_subtasks(completed=set())
        ready_ids = [s.id for s in ready]
        assert "fetch1" in ready_ids
        assert "fetch2" in ready_ids
        assert "deliver" not in ready_ids

        # After fetch1 completes, deliver becomes ready
        ready = graph.get_ready_subtasks(completed={"fetch1"})
        ready_ids = [s.id for s in ready]
        assert "deliver" in ready_ids

    def test_parallelism_score(self):
        # Linear graph: low parallelism
        linear = TaskGraph()
        linear.add_subtask(Subtask(id="s1", task_type=SubtaskType.FETCH, agent_id=0))
        linear.add_subtask(Subtask(id="s2", task_type=SubtaskType.FETCH, agent_id=0, dependencies=["s1"]))
        linear.add_subtask(Subtask(id="s3", task_type=SubtaskType.FETCH, agent_id=0, dependencies=["s2"]))

        # Parallel graph: high parallelism
        parallel = TaskGraph()
        parallel.add_subtask(Subtask(id="s1", task_type=SubtaskType.FETCH, agent_id=0))
        parallel.add_subtask(Subtask(id="s2", task_type=SubtaskType.FETCH, agent_id=1))
        parallel.add_subtask(Subtask(id="s3", task_type=SubtaskType.FETCH, agent_id=2))

        assert linear.get_parallelism_score() < parallel.get_parallelism_score()

    def test_to_prompt_string(self):
        graph = TaskGraph()
        graph.add_subtask(
            Subtask(id="fetch_1", task_type=SubtaskType.FETCH, agent_id=0, target="shelf_1")
        )
        graph.add_subtask(
            Subtask(id="deliver_1", task_type=SubtaskType.DELIVER, agent_id=0, target="workstation", dependencies=["fetch_1"])
        )

        prompt = graph.to_prompt_string()
        assert "Agent_0" in prompt
        assert "fetch" in prompt
        assert "deliver" in prompt
        assert "fetch_1" in prompt

    def test_serialization(self):
        graph = TaskGraph()
        graph.add_subtask(
            Subtask(id="s1", task_type=SubtaskType.FETCH, agent_id=0, target="shelf_1")
        )
        graph.add_subtask(
            Subtask(id="s2", task_type=SubtaskType.DELIVER, agent_id=0, dependencies=["s1"])
        )

        # Serialize
        d = graph.to_dict()

        # Deserialize
        graph2 = TaskGraph.from_dict(d)

        assert len(graph2) == len(graph)
        assert "s1" in graph2.subtasks
        assert "s2" in graph2.subtasks
        assert graph2.subtasks["s2"].dependencies == ["s1"]


class TestGraphCandidate:
    def test_create_candidate(self):
        graph = TaskGraph()
        graph.add_subtask(
            Subtask(id="s1", task_type=SubtaskType.FETCH, agent_id=0)
        )

        candidate = GraphCandidate(
            graph=graph,
            origin="initial_parallel",
            generation=0,
        )

        assert candidate.origin == "initial_parallel"
        assert candidate.generation == 0
        assert len(candidate.candidate_id) > 0

    def test_candidate_serialization(self):
        graph = TaskGraph()
        graph.add_subtask(
            Subtask(id="s1", task_type=SubtaskType.FETCH, agent_id=0)
        )

        candidate = GraphCandidate(
            graph=graph,
            origin="crossover",
            parent_ids=["p1", "p2"],
            generation=2,
        )
        candidate.performance = {"success_rate": 0.8}

        # Serialize
        d = candidate.to_dict()

        # Deserialize
        candidate2 = GraphCandidate.from_dict(d)

        assert candidate2.origin == "crossover"
        assert candidate2.generation == 2
        assert candidate2.parent_ids == ["p1", "p2"]
        assert candidate2.performance["success_rate"] == 0.8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
