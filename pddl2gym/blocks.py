from gridenvs.world import GridObject, GridWorld
from gridenvs.utils import Point, Colors
from gridenvs.env import GridEnv
from pddl2gym.env import PDDLSimulator
from pddl2gym.utils import to_tuple, state_to_atoms_dict, get_atom_fixed_param, files_in_dir
from collections import defaultdict
import numpy as np
import gym
import os


class Blocks(GridEnv):
    def __init__(self, domain_file, instance_file):
        self.pddl = PDDLSimulator(domain_file, instance_file)
        assert list(self.pddl.objects_by_type.keys()) == ["object"], "Wrong domain file?"
        self.n_blocks = len(self.pddl.objects_by_type["object"])
        super(Blocks, self).__init__(n_actions=self.n_blocks, max_moves=100)

    # def _new_get_reduced_actions(self, state=None):
    #     if state is None:
    #         state = self.state["atoms"]
    #     applicable_actions = self.pddl.get_applicable_actions(state)
    #
    #     meta_action_variables = [('?x', 'object')]
    #     meta_action_OR = {'pick-up': ('?x',), 'stack': ('?y', '?x'), 'put-down': ('?x',), 'unstack': ('?y', '?x')}
    #
    #     def get_idx(variable_values):
    #         # TODO: combinations for more than one object
    #         b = variable_values[0]
    #         return self.pddl.objects_by_type['object'].index(b)
    #
    #     def get_variable_values(action_name, sig):
    #         # TODO: combinations for more than one object
    #         b = meta_action_variables[0][0]
    #         param_idx = meta_action_OR[action_name].index(b)
    #         return (sig[param_idx],) # TODO: return None if combination of variable values not in sig
    #
    #
    #     indexed_actions = [None]*self.action_space.n
    #     for action_name, signatures in applicable_actions.items():
    #         if action_name in meta_action_OR:
    #             for sig in signatures:
    #                 v = get_variable_values(action_name, sig)
    #                 if v is not None:
    #                     idx = get_idx(v)
    #                     assert indexed_actions[idx] is None, f"Trying to assign a grounded action to a meta-action that already has a value: {indexed_actions[idx]} <-- {(action_name, sig)}"
    #                     indexed_actions[idx] = (action_name, sig)
    #     return indexed_actions

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
                    if hb == b:
                        action = ("put-down", (hb,))  # empty column
                    else:
                        action = None
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
        colors = iter(Colors.distinguishable_hex)
        for b in self.pddl.objects_by_type["object"]:
            rgb = Colors.hex_to_rgb(next(colors))
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



# Register environments

def register_envs():
    registered_envs = {}
    for p in ["pddl/Blocks/Track1/Untyped", "pddl/Blocks/Track1/Untyped/Additional"]:
        path = os.path.join(os.path.dirname(__file__), p)
        domain_path = os.path.join(path, "domain.pddl")
        for problem_file in sorted(files_in_dir(path)):
            if problem_file.endswith(".pddl") and problem_file != "domain.pddl":
                if problem_file.startswith("probBLOCKS") or problem_file.startswith("probblocks"):
                    _, i, j = problem_file.split(".")[0].split("-")  # output example: ['probBLOCKS', '4', '0']
                    env_id = f'PDDL_Blocks_{i}_{j}-v0'
                else:
                    raise NotImplementedError(f"Environment id not specified for problem file {problem_file}")

                try:
                    instance_path = os.path.join(path, problem_file)
                    gym.register(id=env_id,
                                 entry_point='pddl2gym.blocks:Blocks',
                                 nondeterministic=False,
                                 kwargs={"domain_file": domain_path,
                                         "instance_file": instance_path})
                    registered_envs[env_id] = (domain_path, instance_path)
                except gym.error.Error:
                    pass

    return registered_envs


registered_envs = register_envs()



if __name__ == "__main__":
    print("Registered environments:")
    for env_id in registered_envs:
        print(f"\t{env_id}")
        env = gym.make(env_id)