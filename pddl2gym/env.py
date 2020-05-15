import sys
import os
module_path = os.path.dirname(__file__)
sys.path.append(os.path.join(module_path, 'pyperplan_planner')) # so that pyperplan imports work  # TODO: look into this

from pddl2gym.pyperplan_planner.grounding import ground
from collections import defaultdict
import gym
from pddl2gym.utils import to_tuple, to_string, get_objects_by_type, parse_problem
import numpy as np
from gridenvs.env import GridEnv


class PDDLSimulator:
    def __init__(self, problem):
        self.problem = problem
        self.task = ground(problem)

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
        self.pddl = PDDLSimulator(parse_problem(domain_file, instance_file))
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
        return self.state

    def restore_state(self, state):
        self.state = state


class PDDLGridEnv(GridEnv):
    def __init__(self, pddl, **kwargs):
        self.pddl = pddl
        super(PDDLGridEnv, self).__init__(using_immutable_states=True,
                                          fixed_init_state=True,
                                          **kwargs)

    def get_init_state(self):
        atoms = self.pddl.get_initial_state()
        grid_objects = self.get_grid_objects(atoms)
        return {"atoms": atoms, "grid_objects": grid_objects}

    def get_next_state(self, state, action):
        if np.issubdtype(type(action), np.integer):
            actions = self.get_reduced_actions(state["atoms"])
            assert action < len(actions), f"Action index {action} exceeds the number of actions ({len(actions)})"
            action = actions[action]

        if action is None:
            return state, 0.0, False, {}

        new_atoms = self.pddl.apply(state["atoms"], action)
        done = self.pddl.goal_reached(new_atoms)
        reward = float(done)

        next_state = {"atoms": new_atoms,
                      "grid_objects": self.get_grid_objects(new_atoms)}
        return next_state, reward, done, {}

    def get_objects_to_render(self, state):
        return state["grid_objects"]

    def get_goal_obs(self):
        goal_atoms = self.pddl.task.goals
        complete_goal_atoms = self.get_atoms_from_reduced_set(goal_atoms)
        print(goal_atoms)
        print(complete_goal_atoms)
        raise
        grid_objects = self.get_grid_objects(complete_goal_atoms)
        goal_obs = self.world.render(grid_objects, size=self.pixel_size)
        return goal_obs

    def get_indexed_actions(self, state=None):
        if state is None:
            state = self._state["state"]
        return self.get_reduced_actions(state["atoms"])

    def get_applicable_actions(self, state=None):
        if state is None:
            state = self._state["state"]
        return self.pddl.get_applicable_actions(state["atoms"])

    def get_reduced_actions(self, atoms):
        raise NotImplementedError()

    def get_grid_objects(self, atoms):
        raise NotImplementedError()

    def get_atoms_from_reduced_set(self, atoms):
        raise NotImplementedError()

if __name__ == "__main__":
    path = os.path.join(module_path, "pddl/blocks")
    domain = "domain.pddl"
    instance = "probBLOCKS-4-0.pddl"

    env = PDDLEnv(os.path.join(path, domain), os.path.join(path, instance))
    initial_atoms = env.reset()
    print(initial_atoms)
