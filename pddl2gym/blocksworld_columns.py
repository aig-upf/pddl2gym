import sys
sys.path.append('/home/mjunyent/repos/pddl2gym') # so that pyperplan imports work  # TODO: look into this
sys.path.append('/home/mjunyent/repos/pddl2gym/pddl2gym') # so that pyperplan imports work  # TODO: look into this
sys.path.append('/home/mjunyent/repos/pddl2gym/pddl2gym/pyperplan_planner') # so that pyperplan imports work  # TODO: look into this


from gridenvs.world import GridObject, GridWorld
from gridenvs.utils import Point, Colors
from gridenvs.env import GridEnv
from pddl2gym.env import PDDLSimulator
from pddl2gym.utils import to_tuple, state_to_atoms_dict, get_atom_fixed_param
import numpy as np
import gym
import os


class BlocksworldColumns(GridEnv):
    def __init__(self, domain_file, instance_file):
        self.pddl = PDDLSimulator(domain_file, instance_file)
        assert all(k in ["salient", "block", "column"] for k in self.pddl.objects_by_type.keys()), "Wrong domain file?"
        super(BlocksworldColumns, self).__init__(n_actions=len(self.pddl.objects_by_type["column"]), max_moves=100)

    def get_reduced_actions(self, state=None):
        if state is None:
            state = self.state["atoms"]
        applicable_actions = self.pddl.get_applicable_actions(state)

        atoms = state_to_atoms_dict(state)

        # one action per column
        actions = []
        for col in self.pddl.objects_by_type["column"]:
            bp = self._get_block_pile(atoms, col)
            if "holding" in atoms:
                hb = atoms["holding"][0][0]
                if len(bp) == 0:
                    action = ("putdown", (hb, col))  # empty column
                else:
                    action = ("stack", (hb, bp[-1])) # column has at least one block, stack on top one
            else:
                assert "hand-free" in atoms
                if len(bp) == 0:
                    action = None  # empty column, we cannot pick any block from there
                elif len(bp) == 1:
                    action = ("pickup", (bp[-1], col))  # only one block
                else:
                    action = ("unstack", (bp[-1], bp[-2]))

            assert action is None or action[1] in applicable_actions[action[0]], f"\n{self.state['world']}\nAction {action} not in {applicable_actions} for state {self.state['atoms']}"
            actions.append(action)
        return actions

    def get_init_state(self):
        state = {}

        # Get grid size
        all_blocks = self.pddl.objects_by_type["salient"] + self.pddl.objects_by_type["block"]
        grid_size = (len(all_blocks), len(self.pddl.objects_by_type["column"]) + 1)

        # Create a gridenvs object for each block
        state["world"] = GridWorld(grid_size)
        s = self.pddl.objects_by_type["salient"][0]
        state["world"].add_object(GridObject(name=s, pos=(0, 0), rgb=Colors.red))
        colors = iter(Colors.distinguishable.items())
        for b in self.pddl.objects_by_type["block"]:
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

    def _get_block_pile(self, atoms_dict, col):  #ordered bottom to top
        blocks = []
        if not ("empty" in atoms_dict and (col,) in atoms_dict["empty"]):
            params = get_atom_fixed_param(atoms_dict, name="bottom", param_idx=1, param_value=col)
            assert params is not None
            blocks.append(params[0])

            while params is not None:
                params = get_atom_fixed_param(atoms_dict, name="on", param_idx=1, param_value=blocks[-1])
                if params is None:
                    break
                blocks.append(params[0])
        return blocks

    def _update_objects(self, atoms, world):
        on = dict()
        col_bottom = dict()
        holding_block = None
        for a in atoms:
            name, signature = to_tuple(a)

            if name == "on":
                b_top, b_bottom = signature
                on[b_bottom] = b_top
            elif name == "clear":
                b = signature[0]
                on[b] = None
            elif name == "bottom":
                b, c = signature
                col_bottom[c] = b
            elif name == "empty":
                c = signature[0]
                col_bottom[c] = None
            elif name == "holding":
                holding_block = signature[0]
            elif name == "hand-free":
                assert holding_block is None

        assert len(col_bottom.keys()) == len(self.pddl.objects_by_type["column"])

        for i, c in enumerate(self.pddl.objects_by_type["column"]):
            j = 0
            b = col_bottom[c]
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


def blocksworld_columns_clear(problem_file):
    path = "/home/mjunyent/repos/pddl2gym/pddl2gym/pddl/blocks_columns"
    domain_file = "domain.pddl"
    env = BlocksworldColumns(os.path.join(path, domain_file), os.path.join(path, problem_file))
    return env


try:
    for blocks in [5, 15]:
        gym.register(id=f'PDDL_BlocksworldColumns{blocks}-v0',
                     entry_point=f'pddl2gym.blocksworld_columns:blocksworld_columns_clear',
                     nondeterministic=False,
                     kwargs={"problem_file": f"instance_{blocks}_clear_x_1.pddl"})
except gym.error.Error:
    pass

if __name__ == "__main__":
    env = gym.make('PDDL_BlocksworldColumns5-v0')