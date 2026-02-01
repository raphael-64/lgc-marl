"""Prompt templates for LLM graph generation."""

INITIAL_GRAPH_PROMPT = """You are a multi-robot warehouse task planner. Generate a task decomposition graph for coordinating robots to complete warehouse deliveries.

## ENVIRONMENT
{env_state}

## TASK
{task}

## NUMBER OF ROBOTS
{n_agents}

## STRATEGY HINT: {strategy}
Strategy descriptions:
- sequential: Robots take turns, one completes before next starts
- parallel: Maximize parallel execution, all robots work simultaneously
- pipelined: Division of labor - some robots fetch, others deliver
- zone_based: Each robot owns a section of the warehouse
- nearest_first: Assign tasks greedily by proximity
- load_balanced: Distribute work evenly across all robots
- hierarchical: One robot coordinates, others execute
- adaptive: Mix strategies based on task requirements

## RULES
1. Each subtask must be assigned to exactly ONE robot (agent_id 0 to {n_agents_minus_1})
2. Use dependencies to indicate ordering constraints
3. Consider collision avoidance - robots shouldn't block each other
4. Maximize parallelism where safe
5. A robot must FETCH a shelf before it can DELIVER it

## SUBTASK TYPES
- fetch: Robot goes to shelf location and picks it up
- deliver: Robot brings held shelf to workstation
- navigate: Robot moves to a position (for coordination)
- wait: Robot waits for another task to complete
- coordinate: Explicit synchronization point between robots

## OUTPUT FORMAT
Respond with ONLY a JSON block in this exact format:
```json
{{
  "subtasks": [
    {{"id": "fetch_shelf1", "type": "fetch", "agent": 0, "target": "shelf_1", "dependencies": []}},
    {{"id": "fetch_shelf2", "type": "fetch", "agent": 1, "target": "shelf_2", "dependencies": []}},
    {{"id": "deliver_shelf1", "type": "deliver", "agent": 0, "target": "workstation", "dependencies": ["fetch_shelf1"]}},
    {{"id": "deliver_shelf2", "type": "deliver", "agent": 1, "target": "workstation", "dependencies": ["fetch_shelf2", "deliver_shelf1"]}}
  ]
}}
```

Generate the task graph now:
"""


CROSSOVER_PROMPT = """You are evolving task decomposition graphs for multi-robot warehouse coordination. Combine the best features from two high-performing parent graphs.

## PARENT 1 (success rate: {parent1_perf:.1%})
{parent1_graph}

Strengths observed: {parent1_strengths}

## PARENT 2 (success rate: {parent2_perf:.1%})
{parent2_graph}

Strengths observed: {parent2_strengths}

## PATTERNS THAT WORK (from successful runs)
{successful_patterns}

## PATTERNS TO AVOID (from failures)
{failure_patterns}

## CROSSOVER GUIDELINES
1. Keep structures that BOTH parents share (likely important)
2. When parents differ, prefer the structure from the higher performer
3. Combine complementary strengths (e.g., Parent 1's fetch ordering + Parent 2's delivery coordination)
4. Ensure the result is a valid DAG (no cycles)
5. Maintain valid dependencies

## OUTPUT FORMAT
Respond with ONLY a JSON block:
```json
{{
  "subtasks": [
    {{"id": "...", "type": "...", "agent": 0, "target": "...", "dependencies": []}}
  ]
}}
```

Generate the crossover child graph:
"""


MUTATION_PROMPT = """You are refining a task decomposition graph for multi-robot warehouse coordination. Make targeted improvements based on observed performance.

## CURRENT GRAPH (success rate: {parent_perf:.1%})
{parent_graph}

## FAILURE ANALYSIS FROM THIS GRAPH
{failure_analysis}

## SUCCESSFUL PATTERNS FROM OTHER GRAPHS
{successful_patterns}

## MUTATION OPTIONS
Choose 1-2 targeted changes:
1. Remove unnecessary dependencies that create bottlenecks
2. Add missing coordination edges where robots collide
3. Reorder subtasks to reduce waiting time
4. Reassign subtasks to balance load across robots
5. Split a complex dependency chain
6. Merge redundant synchronization points

## OUTPUT FORMAT
Respond with ONLY a JSON block:
```json
{{
  "subtasks": [
    {{"id": "...", "type": "...", "agent": 0, "target": "...", "dependencies": []}}
  ]
}}
```

Generate the mutated graph:
"""


FIX_FAILURES_PROMPT = """This task decomposition graph performed poorly. Fix the specific failure modes aggressively.

## FAILING GRAPH
{failing_graph}

## SPECIFIC FAILURES OBSERVED
- Common failure points: {failure_points}
- Coordination breakdowns: {coordination_failures}
- Bottleneck subtasks: {bottlenecks}

## WHAT WORKS IN SUCCESSFUL GRAPHS
{successful_patterns}

## FIX GUIDELINES
This graph needs SIGNIFICANT changes. Be aggressive:
1. Completely restructure problematic sections
2. Rebalance work if one robot is overloaded
3. Add explicit coordination where collisions occurred
4. Remove unnecessary sequential dependencies
5. Consider a completely different strategy if current approach is fundamentally flawed

## OUTPUT FORMAT
Respond with ONLY a JSON block:
```json
{{
  "subtasks": [
    {{"id": "...", "type": "...", "agent": 0, "target": "...", "dependencies": []}}
  ]
}}
```

Generate the fixed graph:
"""


NOVEL_GRAPH_PROMPT = """Generate a NOVEL task decomposition graph for multi-robot warehouse coordination, informed by learnings from previous attempts.

## ENVIRONMENT
{env_state}

## TASK
{task}

## LEARNINGS FROM PREVIOUS GENERATIONS
Patterns that lead to SUCCESS:
{successful_patterns}

Patterns that lead to FAILURE:
{failure_patterns}

Performance trend across generations: {generation_scores}

## NOVELTY GUIDELINES
1. Incorporate ALL successful patterns discovered
2. Avoid ALL known failure patterns
3. Try something genuinely NEW that hasn't been explored yet
4. Consider unconventional strategies:
   - Staged execution (all fetches, then all delivers)
   - Buddy system (pairs of robots working together)
   - Wave-based (batches of parallel tasks)
   - Priority lanes (dedicate paths to specific robots)

## OUTPUT FORMAT
Respond with ONLY a JSON block:
```json
{{
  "subtasks": [
    {{"id": "...", "type": "...", "agent": 0, "target": "...", "dependencies": []}}
  ]
}}
```

Generate a novel graph:
"""


# ============================================================================
# OVERCOOKED PROMPTS
# ============================================================================

OVERCOOKED_INITIAL_GRAPH_PROMPT = """You are a task planner for the Overcooked cooking game. Two chefs (Agent 0 and Agent 1) must coordinate to prepare and serve soups.

## ENVIRONMENT - {layout}
{terrain}
Legend: O=onion dispenser, P=pot, S=serving location, D=dish dispenser, X=counter, space=walkable

## ORDER: {orders}
Recipe: 3 onions → pot → cook → plate → serve

## YOUR STRATEGY: {strategy}
YOU MUST FOLLOW THIS STRATEGY. Each strategy requires DIFFERENT task assignments:

- role_based: Agent 0 does ALL ingredient work (get onions, put in pot). Agent 1 does ALL finishing work (get dish, plate, serve). Clear role separation.
- parallel_soups: If 2 pots exist, each agent works their own pot independently. Agent 0 owns pot_0, Agent 1 owns pot_1.
- pipeline: Assembly line. Agent 0 does steps 1-3 (get onions, put in pot, wait). Agent 1 does steps 4-6 (get dish, plate, serve).
- zone_based: Agent 0 works left side of kitchen only. Agent 1 works right side only. Use counter handoffs between zones.
- helper: Agent 0 is primary cook doing main tasks. Agent 1 only assists (fetches extra onions, clears path, handles overflow).
- alternating: Agents take strict turns. Agent 0 does task 1, Agent 1 does task 2, Agent 0 does task 3, etc. Avoids collisions.

## SUBTASK TYPES
- get_ingredient: Pick up onion from dispenser
- put_in_pot: Place onion in pot
- wait_cooking: Wait for soup to cook
- get_dish: Pick up dish from dispenser
- plate_soup: Take soup from pot onto dish
- serve: Deliver to serving location
- put_on_counter: Place item on counter (for handoffs)
- get_from_counter: Pick up item from counter

## OUTPUT FORMAT
```json
{{
  "subtasks": [
    {{"id": "unique_id", "type": "subtask_type", "agent": 0, "target": "target", "dependencies": ["dep_id"]}},
    ...
  ]
}}
```

IMPORTANT:
- Generate 8-12 subtasks for a complete soup
- Agent assignments MUST reflect the {strategy} strategy
- DO NOT just assign tasks round-robin - think about the strategy!
- Dependencies should reflect real task ordering (can't plate before cooking, etc.)

Generate the task graph:
"""

OVERCOOKED_CROSSOVER_PROMPT = """Combine the best features from two cooking coordination strategies.

## PARENT 1 (avg reward: {parent1_perf:.1f})
{parent1_graph}
Observed: {parent1_strengths}

## PARENT 2 (avg reward: {parent2_perf:.1f})
{parent2_graph}
Observed: {parent2_strengths}

## WHAT WORKS
{successful_patterns}

## WHAT FAILS
{failure_patterns}

Keep structures both parents share. Combine complementary strengths.

## SUBTASK TYPES (use ONLY these)
- get_ingredient: Pick up onion from dispenser
- put_in_pot: Place onion in pot
- wait_cooking: Wait for soup to cook
- get_dish: Pick up dish from dispenser
- plate_soup: Take soup from pot onto dish
- serve: Deliver to serving location

## OUTPUT FORMAT - MUST use this exact structure
```json
{{
  "subtasks": [
    {{"id": "get_onion_1", "type": "get_ingredient", "agent": 0, "target": "onion", "dependencies": []}},
    {{"id": "put_pot_1", "type": "put_in_pot", "agent": 0, "target": "pot", "dependencies": ["get_onion_1"]}},
    {{"id": "get_onion_2", "type": "get_ingredient", "agent": 1, "target": "onion", "dependencies": []}},
    ...more subtasks with same structure...
  ]
}}
```

Generate the crossover (8-12 subtasks for a complete soup):
"""

OVERCOOKED_MUTATION_PROMPT = """Improve this cooking coordination strategy based on observed failures.

## CURRENT STRATEGY (avg reward: {parent_perf:.1f})
{parent_graph}

## PROBLEMS OBSERVED
{failure_analysis}

## MUTATION IDEAS
1. Reduce collisions by staggering chef movements
2. Add handoff via counter if chefs are blocking each other
3. Rebalance workload if one chef is idle
4. Change role assignments
5. Add explicit wait/coordination points

## SUBTASK TYPES (use ONLY these)
- get_ingredient: Pick up onion from dispenser
- put_in_pot: Place onion in pot
- wait_cooking: Wait for soup to cook
- get_dish: Pick up dish from dispenser
- plate_soup: Take soup from pot onto dish
- serve: Deliver to serving location

## OUTPUT FORMAT - MUST use this exact structure
```json
{{
  "subtasks": [
    {{"id": "get_onion_1", "type": "get_ingredient", "agent": 0, "target": "onion", "dependencies": []}},
    {{"id": "put_pot_1", "type": "put_in_pot", "agent": 0, "target": "pot", "dependencies": ["get_onion_1"]}},
    ...more subtasks with same structure...
  ]
}}
```

Generate the mutated graph (8-12 subtasks for a complete soup):
"""

OVERCOOKED_FIX_FAILURES_PROMPT = """This cooking strategy is failing badly. Make aggressive changes.

## FAILING STRATEGY
{failing_graph}

## FAILURES
- Collision locations: {failure_points}
- Missed timing: {coordination_failures}
- Idle time: {bottlenecks}

## WORKING PATTERNS
{successful_patterns}

Make SIGNIFICANT changes. Consider completely different role assignments.

## SUBTASK TYPES (use ONLY these)
- get_ingredient: Pick up onion from dispenser
- put_in_pot: Place onion in pot
- wait_cooking: Wait for soup to cook
- get_dish: Pick up dish from dispenser
- plate_soup: Take soup from pot onto dish
- serve: Deliver to serving location

## OUTPUT FORMAT - MUST use this exact structure
```json
{{
  "subtasks": [
    {{"id": "get_onion_1", "type": "get_ingredient", "agent": 0, "target": "onion", "dependencies": []}},
    {{"id": "put_pot_1", "type": "put_in_pot", "agent": 0, "target": "pot", "dependencies": ["get_onion_1"]}},
    ...more subtasks with same structure...
  ]
}}
```

Generate the fixed graph (8-12 subtasks for a complete soup):
"""

OVERCOOKED_NOVEL_GRAPH_PROMPT = """Generate a NOVEL cooking coordination strategy.

## ENVIRONMENT - {layout}
{terrain}

## ORDER: {orders}

## LEARNINGS
SUCCESS patterns: {successful_patterns}
FAILURE patterns: {failure_patterns}

## NOVEL IDEAS TO TRY
- Designated "runner" who only moves items between stations
- Batch mode: both chefs do same task type at once
- Counter relay: pass items via counter chain
- One-pot focus: both chefs on same soup, max speed
- Speculative prep: start next soup while serving current

## SUBTASK TYPES (use ONLY these)
- get_ingredient: Pick up onion from dispenser
- put_in_pot: Place onion in pot
- wait_cooking: Wait for soup to cook
- get_dish: Pick up dish from dispenser
- plate_soup: Take soup from pot onto dish
- serve: Deliver to serving location

## OUTPUT FORMAT - MUST use this exact structure
```json
{{
  "subtasks": [
    {{"id": "get_onion_1", "type": "get_ingredient", "agent": 0, "target": "onion", "dependencies": []}},
    {{"id": "put_pot_1", "type": "put_in_pot", "agent": 0, "target": "pot", "dependencies": ["get_onion_1"]}},
    ...more subtasks with same structure...
  ]
}}
```

Generate a novel graph (8-12 subtasks for a complete soup):
"""

# ============================================================================
# RWARE PROMPTS (existing)
# ============================================================================

REWARD_FUNCTION_PROMPT = """Generate a reward function for training multi-agent reinforcement learning policies on warehouse tasks.

## ENVIRONMENT
{env_description}

## TASK GOALS
{task_goals}

## TASK GRAPH STRUCTURE
{graph_structure}

## REWARD GUIDELINES
The reward function should:
1. Encourage task completion (delivering shelves to workstation)
2. Encourage coordination based on the graph dependencies
3. Penalize collisions and deadlocks
4. Provide intermediate rewards for progress (subtask completion)
5. Be shaped to guide learning (not too sparse)

## OUTPUT FORMAT
Respond with a Python function:
```python
def compute_reward(info: dict, graph: TaskGraph, completed_subtasks: set) -> list[float]:
    '''
    Compute rewards for each agent.

    Args:
        info: Environment info dict with keys like 'agent_positions', 'collisions', etc.
        graph: The task decomposition graph
        completed_subtasks: Set of completed subtask IDs

    Returns:
        List of reward values, one per agent
    '''
    rewards = [0.0] * info['n_agents']

    # Your reward logic here

    return rewards
```

Generate the reward function:
"""
