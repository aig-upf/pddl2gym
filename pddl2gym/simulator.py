from pddl2gym.pyperplan_planner.grounding import ground
from collections import defaultdict
from pddl2gym.utils import to_tuple, to_string, get_objects_by_type


class PDDLProblemSimulator:
    def __init__(self, problem):
        self.problem = problem
        self.task = ground(self.problem)
        self.operators = {to_tuple(op.name): op for op in self.task.operators}

    def get_atoms(self, state):
        return state

    def reset(self):
        return self.task.initial_state

    def get_goal(self):
        return self.task.goals

    def apply(self, state, action):
        if isinstance(action, str):  # Str action
            action = to_tuple(action)
        try:
            op = self.operators[action]
        except KeyError:
            raise Exception(f"Action {action} not in possible operators (grounded actions): {self.operators.keys()}")
        if not op.applicable(state):
            raise Exception(f"Action {action} not applicable in state {state}")
        return op.apply(state)

    def goal_reached(self, state):
        return self.task.goal_reached(state)

    def get_applicable_actions(self, state):
        applicable_actions = defaultdict(list)
        for (action, params), op in self.operators.items():
            if op.applicable(state):
                applicable_actions[action].append(params)
        return applicable_actions

    def get_applicable_str_actions(self, state):
        actions = self.get_applicable_actions(state)
        return [to_string(a, params) for a, params_list in actions.items() for params in params_list]


class PDDLDomainSimulator:
    def __init__(self, domain, problem_generator):
        self.domain = domain
        self.problem_generator = problem_generator
        self.problem_id = -1
        self.reset()

    @property
    def problem(self):
        return self.problem_simulator.problem

    def get_atoms(self, state):
        return state[1]

    def reset(self):
        problem = next(self.problem_generator)
        assert problem.domain is self.domain
        problem.objects_by_type = get_objects_by_type(problem)
        self.problem_simulator = PDDLProblemSimulator(problem)
        s = self.problem_simulator.reset()
        self.problem_id += 1
        return (self.problem_id, s)

    def get_goal(self):
        return self.problem_simulator.get_goal()

    def apply(self, state, action):
        prob_id, s = state
        assert prob_id == self.problem_id
        return (prob_id, self.problem_simulator.apply(s, action))

    def goal_reached(self, state):
        prob_id, s = state
        assert prob_id == self.problem_id
        return self.problem_simulator.goal_reached(s)

    def get_applicable_actions(self, state):
        prob_id, s = state
        assert prob_id == self.problem_id
        return self.problem_simulator.get_applicable_actions(s)

    def get_applicable_str_actions(self, state):
        prob_id, s = state
        assert prob_id == self.problem_id
        return self.problem_simulator.get_applicable_actions(s)

