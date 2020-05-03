import sys
sys.path.append('/home/mjunyent/repos/pddl2gym') # so that pyperplan imports work  # TODO: look into this
sys.path.append('/home/mjunyent/repos/pddl2gym/pddl2gym') # so that pyperplan imports work  # TODO: look into this
sys.path.append('/home/mjunyent/repos/pddl2gym/pddl2gym/pyperplan_planner') # so that pyperplan imports work  # TODO: look into this

from pyperplan_planner.pyperplan import _parse, _ground
from collections import defaultdict
import gym
from pddl2gym.utils import to_tuple, to_string, get_objects_by_type

class PDDLSimulator:
    def __init__(self, domain_file, instance_file):
        self.problem = _parse(domain_file, instance_file)
        self.task = _ground(self.problem)
        self.operators = {to_tuple(op.name): op for op in self.task.operators}
        self.objects_by_type = get_objects_by_type(self.problem)

    def get_initial_state(self):
        return self.task.initial_state

    def apply(self, state, action):
        if type(action) is str:  # Str action
            action = to_tuple(action)
        try:
            op = self.operators[action]
        except KeyError:
            raise Exception(f"Action {action} not in possible operators (grounded actions).")
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


class PDDLEnv(gym.Env): #TODO: use gym.GoalEnv?
    def __init__(self, domain_file, instance_file):
        self.pddl = PDDLSimulator(domain_file, instance_file)
        self.state = self.pddl.get_initial_state()

    def step(self, action):
        self.state = self.pddl.apply(self.state, action)
        done = self.pddl.goal_reached(self.state)
        reward = float(done)
        return self.state, reward, done, {}

    def reset(self):
        self.state = self.pddl.get_initial_state()
        return self.state

    def clone_state(self):
        return self.state # TODO: do we need copy.deepcopy here? Also after get_initial_state().

    def restore_state(self, state):
        self.state = state


if __name__ == "__main__":
    import os

    # path = "../aibasel-downward-benchmarks/blocks/"
    path = "/home/mjunyent/repos/GP-learn-feats/pddl-encoding/domains/blocks"
    domain = "domain.pddl"
    instance = "probBLOCKS-4-0.pddl"

    env = PDDLEnv(os.path.join(path, domain), os.path.join(path, instance))
    initial_atoms = env.reset()
    print(initial_atoms)
