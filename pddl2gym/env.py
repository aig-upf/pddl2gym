import gym
import numpy as np
from gridenvs.env import GridEnv
from gridenvs.utils import Colors


class PDDLEnv(gym.Env):  # TODO: use gym.GoalEnv?
    def __init__(self, simulator):
        self.simulator = simulator

    def step(self, action):
        self.state = self.simulator.apply(self.state, action)
        atoms = self.simulator.get_atoms(self.state)
        done = self.simulator.goal_reached(self.state)
        reward = float(done)
        return atoms, reward, done, {}

    def reset(self):
        self.state = self.simulator.reset()
        return self.simulator.get_atoms(self.state)

    def clone_state(self):
        return self.state

    def restore_state(self, state):
        self.state = state


class PDDLGridEnv(GridEnv):
    def __init__(self, simulator, representation, **kwargs):
        self.simulator = simulator
        self.representation = representation
        super(PDDLGridEnv, self).__init__(n_actions=representation.get_n_actions(self.simulator.problem),
                                          using_immutable_states=True,
                                          **kwargs)

    def get_init_state(self):
        simulator_state = self.simulator.reset()
        atoms = self.simulator.get_atoms(simulator_state)
        grid_state = self.representation.get_gridstate(self.simulator.problem, atoms)
        self._goal_obs = self._get_goal_obs()
        return {"simulator_state": simulator_state, "grid_state": grid_state}

    def get_next_state(self, state, action):
        if np.issubdtype(type(action), np.integer):
            atoms = self.simulator.get_atoms(state["simulator_state"])
            actions = self.representation.get_reduced_actions(self.simulator.problem, atoms)
            assert action < len(actions), f"Action index {action} exceeds the number of actions ({len(actions)})"
            action = actions[action]

        if action is None:
            return state, 0.0, False, {}

        simulator_state = self.simulator.apply(state["simulator_state"], action)
        atoms = self.simulator.get_atoms(simulator_state)
        grid_state = self.representation.get_gridstate(self.simulator.problem, atoms)
        done = self.simulator.goal_reached(simulator_state)
        reward = float(done)

        next_state = {"simulator_state": simulator_state,
                      "grid_state": grid_state}
        return next_state, reward, done, {}

    def get_gridstate(self, state):
        return state["grid_state"]

    def get_goal_obs(self):
        return self._goal_obs

    def _get_goal_obs(self):
        goal_atoms = self.simulator.get_goal()
        complete_goal_atoms = self.representation.get_atoms_from_subset(self.simulator.problem, goal_atoms)
        grid_objects = self.representation.get_gridstate(self.simulator.problem, complete_goal_atoms)
        goal_obs = self.world.render(grid_objects, size=self.pixel_size)
        return goal_obs

    def get_indexed_actions(self, state=None):
        if state is None:
            state = self._state["state"]
        atoms = self.simulator.get_atoms(state["simulator_state"])
        return self.representation.get_reduced_actions(self.simulator.problem, atoms)

    def get_applicable_actions(self, state=None):
        if state is None:
            state = self._state["state"]
        atoms = self.simulator.get_atoms(state["simulator_state"])
        return self.simulator.get_applicable_actions(self.simulator.problem, atoms)


class PDDLRepresentation:
    def __init__(self):
        self.colors = [Colors.hex_to_rgb(c) for c in Colors.distinguishable_hex]

    def get_n_actions(self, problem):
        raise NotImplementedError()

    def get_reduced_actions(self, problem, atoms):
        raise NotImplementedError()

    def get_gridstate(self, problem, atoms):
        raise NotImplementedError()

    def get_atoms_from_subset(self, problem, atoms):
        raise NotImplementedError()


if __name__ == "__main__":
    import os
    from pddl2gym.utils import parse_problem
    from pddl2gym.simulator import PDDLProblemSimulator

    module_path = os.path.dirname(__file__)
    path = os.path.join(module_path, "pddl/Blocks/Track1/Untyped")
    domain = "domain.pddl"
    instance = "probBLOCKS-4-0.pddl"

    pddl_simulator = PDDLProblemSimulator(parse_problem(domain, instance, path))
    env = PDDLEnv(pddl_simulator)
    initial_atoms = env.reset()
    print(initial_atoms)
