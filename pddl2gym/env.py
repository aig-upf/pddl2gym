import sys

from pyperplan_planner.pyperplan import _parse, _ground
from collections import defaultdict
import gym


class PDDLEnv(gym.Env): #TODO: use gym.GoalEnv?
    def __init__(self, domain_file, instance_file):
        assert instance_file is not None
        self.problem = _parse(domain_file, instance_file)
        self.actions = list(self.problem.domain.actions.values())
        self.task = _ground(self.problem)
        self.operators = {(op.name[1:-1].split(' ')[0], tuple(op.name[1:-1].split(' ')[1:])): op for op in self.task.operators}
        self.state = self.task.initial_state

    def step(self, action, params=None):
        if params is None: # Grounded action
            res = action.split(" ")
            action = res[0]
            params = tuple(res[1:])
        elif type(params) is not tuple:
            params = tuple([params])
        assert " " not in action

        try:
            op = self.operators[(action, params)]
        except KeyError:
            raise Exception(f"Action {action} with params {params} not in possible operators (grounded actions).")
        if not op.applicable(self.state):
            raise Exception(f"Action {action}({params}) not applicable in state {self.state.as_atoms()}")
        self.state = op.apply(self.state)
        done = self.task.goal_reached(self.state)
        reward = float(done)
        return self.state, reward, done, {}

    def reset(self):
        self.state = self.task.initial_state
        return self.state

    def clone_state(self):
        return self.state # TODO: do we need copy.deepcopy here? Also in initial_state.

    def restore_state(self, state):
        self.state = state

    def get_applicable_actions(self, state=None):
        if state is None:
            state = self.state
        applicable_actions = defaultdict(list)
        for (action, params), op in self.operators.items():
            if op.applicable(state):
                applicable_actions[action].append(params)
        return applicable_actions

    def get_applicable_grounded_actions(self, state=None):
        actions = self.get_applicable_actions(state)
        return [" ".join([a, " ".join(params)])for a, params_list in actions.items() for params in params_list]

if __name__ == "__main__":
    import os

    path = "../aibasel-downward-benchmarks/blocks/"
    domain = "domain.pddl"
    instance = "probBLOCKS-4-0.pddl"

    env = PDDLEnv(os.path.join(path, domain), os.path.join(path, instance))
    initial_atoms = env.reset()
    print(initial_atoms)
