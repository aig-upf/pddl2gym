import sys
sys.path.append('/home/mjunyent/repos/pddl2gym') # so that pyperplan imports work  # TODO: look into this
sys.path.append('/home/mjunyent/repos/pddl2gym/pddl2gym') # so that pyperplan imports work  # TODO: look into this
sys.path.append('/home/mjunyent/repos/pddl2gym/pddl2gym/pyperplan_planner') # so that pyperplan imports work  # TODO: look into this


from gridenvs.world import GridObject, GridWorld
from gridenvs.utils import Point, Colors
from gridenvs.env import GridEnv
from pddl2gym.env import PDDLSimulator
from pddl2gym.utils import to_tuple, state_to_atoms_dict, get_atom_fixed_param
from collections import defaultdict
import numpy as np
import gym
import os


class Blocksworld(GridEnv):
    def __init__(self, domain_file, instance_file):
        self.pddl = PDDLSimulator(domain_file, instance_file)
        assert list(self.pddl.objects_by_type.keys()) == ["object"], "Wrong domain file?"
        self.n_blocks = len(self.pddl.objects_by_type["object"])
        super(Blocksworld, self).__init__(n_actions=self.n_blocks, max_moves=100)

    def get_reduced_actions(self, state=None):
        if state is None:
            state = self.state["atoms"]
        applicable_actions = self.pddl.get_applicable_actions(state)

        atoms = state_to_atoms_dict(state)

        # one action per column
        actions = []
        for b in self.pddl.objects_by_type["object"]:
            bp = self._get_block_pile(atoms, b)
            if "holding" in atoms:
                hb = atoms["holding"][0][0]
                if len(bp) == 0:
                    action = ("put-down", (hb,))  # empty column
                else:
                    action = ("stack", (hb, bp[-1])) # column has at least one block, stack on top one
            else:
                assert "handempty" in atoms
                if len(bp) == 0:
                    action = None  # empty column, we cannot pick any block from there
                elif len(bp) == 1:
                    action = ("pick-up", (bp[-1],))  # only one block
                else:
                    action = ("unstack", (bp[-1], bp[-2]))

            assert action is None or action[1] in applicable_actions[action[0]], f"\n{self.state['world']}\nAction {action} not in {applicable_actions} for state {self.state['atoms']}"
            actions.append(action)
        return actions

    def get_init_state(self):
        state = {}

        # Get grid size
        grid_size = (self.n_blocks + 1, self.n_blocks + 1)

        # Create a gridenvs object for each block
        state["world"] = GridWorld(grid_size)
        colors = iter(Colors.distinguishable.items())
        for b in self.pddl.objects_by_type["object"]:
            color_name, rgb = next(colors)
            state["world"].add_object(GridObject(name=b, pos=(0,0), rgb=rgb))

        # Assign positions to gridenvs object according to the initial state
        state["atoms"] = self.pddl.get_initial_state()
        self._update_objects(state["atoms"], state["world"])
        return state

    def update_environment(self, action):
        if np.issubdtype(type(action), np.integer):
            actions = self.get_reduced_actions()
            assert action < len(actions), f"Action index {action} exceeds the number of actions ({len(actions)})"
            action = actions[action]

        if action is not None:
            self.state["atoms"] = self.pddl.apply(self.state["atoms"], action)
            self._update_objects(self.state["atoms"], self.state["world"])
        done = self.pddl.goal_reached(self.state["atoms"])
        reward = float(done)
        return reward, done, {}

    def _get_block_pile(self, atoms_dict, b):  #ordered bottom to top
        blocks = []
        if "ontable" in atoms_dict and (b,) in atoms_dict["ontable"]:
            blocks.append(b)

            while True:
                params = get_atom_fixed_param(atoms_dict, name="on", param_idx=1, param_value=blocks[-1])
                if params is None:
                    break
                blocks.append(params[0])
        return blocks

    def _update_objects(self, atoms, world):
        on = dict()
        ontable = defaultdict(bool)
        holding_block = None
        for a in atoms:
            name, signature = to_tuple(a)

            if name == "on":
                b_top, b_bottom = signature
                on[b_bottom] = b_top
            elif name == "clear":
                b = signature[0]
                on[b] = None
            elif name == "ontable":
                b = signature[0]
                ontable[b] = True
            elif name == "holding":
                holding_block = signature[0]
            elif name == "handempty":
                assert holding_block is None

        for i, block in enumerate(self.pddl.objects_by_type["object"]):
            if ontable[block]:
                j = 0
                b = block
                while b is not None:
                    objs = world.get_objects_by_names(b)
                    assert len(objs) == 1
                    objs[0].pos = Point(i, world.grid_size[1]-j-1)
                    b = on[b]
                    j += 1

        if holding_block is not None:
            objs = world.get_objects_by_names(holding_block)
            assert len(objs) == 1
            objs[0].pos = Point(world.grid_size[0] - 1, 0)



def blocksworld(problem_file):
    path = "/home/mjunyent/repos/pddl2gym/pddl2gym/pddl/blocks"
    domain_file = "domain.pddl"
    env = Blocksworld(os.path.join(path, domain_file), os.path.join(path, problem_file))
    return env


try:
    gym.register(id=f'PDDL_Blocksworld_17_0-v0',
                 entry_point=f'pddl2gym.blocksworld:blocksworld',
                 nondeterministic=False,
                 kwargs={"problem_file": "probBLOCKS-17-0.pddl"})
except gym.error.Error:
    pass

if __name__ == "__main__":
    env = gym.make('PDDL_Blocksworld_17_0-v0')