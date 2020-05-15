from gridenvs.world import GridObject
from gridenvs.utils import Colors
from pddl2gym.env import PDDLSimulator, PDDLGridEnv
from pddl2gym.utils import parse_problem, to_tuple, to_string, to_atoms_dict, get_atom_fixed_param, files_in_dir
from collections import defaultdict
import gym
import os


class Blocks(PDDLGridEnv):
    def __init__(self, domain_file, instance_file, max_moves=100):
        pddl = PDDLSimulator(parse_problem(domain_file, instance_file))
        assert list(pddl.objects_by_type.keys()) == ["object"], "Wrong domain file?"
        blocks = pddl.objects_by_type["object"]
        self.n_blocks = len(blocks)

        colors_iter = iter(Colors.distinguishable_hex)
        self.block_colors = {}
        for b in blocks:
            self.block_colors[b] = Colors.hex_to_rgb(next(colors_iter))

        super(Blocks, self).__init__(pddl=pddl,
                                     size=(self.n_blocks + 1, self.n_blocks + 1),
                                     n_actions=self.n_blocks,
                                     max_moves=max_moves)

    def get_reduced_actions(self, atoms):
        atoms_dict = to_atoms_dict(atoms)

        # one action per column
        actions = []
        for b in self.pddl.objects_by_type["object"]:
            bp = self._get_block_pile(atoms_dict, b)
            if "holding" in atoms_dict:
                hb = atoms_dict["holding"][0][0]
                if len(bp) == 0:
                    if hb == b:
                        action = ("put-down", (hb,))  # empty column
                    else:
                        action = None
                else:
                    action = ("stack", (hb, bp[-1])) # column has at least one block, stack on top one
            else:
                assert "handempty" in atoms_dict
                if len(bp) == 0:
                    action = None  # empty column, we cannot pick any block from there
                elif len(bp) == 1:
                    action = ("pick-up", (bp[-1],))  # only one block
                else:
                    action = ("unstack", (bp[-1], bp[-2]))

            actions.append(action)
        return actions

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

    def _read_atoms(self, atoms):
        ontable = defaultdict(bool)
        on = dict()  # on[y] = x <-> on x y
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
        return ontable, on, holding_block

    def get_grid_objects(self, atoms):
        ontable, on, holding_block = self._read_atoms(atoms)

        blocks = self.pddl.objects_by_type["object"]
        objects = []
        for i, block in enumerate(blocks):
            if ontable[block]:
                j = 0
                b = block
                while b is not None:
                    objects.append(GridObject(name=b,
                                              pos=(i, self.world.size[1]-j-1),
                                              rgb=self.block_colors[b]))
                    b = on[b]
                    j += 1
            else:
                assert block in on.values() or block == holding_block

        if holding_block is not None:
            objects.append(GridObject(name=holding_block,
                                      pos=(self.world.size[0] - 1, 0),
                                      rgb=self.block_colors[holding_block]))

        return objects

    def get_atoms_from_reduced_set(self, atoms):
        # We asume that, if not stated otherwise, all blocks are on the table and clear, and the hand is empty
        ontable, on, holding_block = self._read_atoms(atoms)

        deduced_atoms = []
        blocks = self.pddl.objects_by_type["object"]
        for b in blocks:
            # b is ontable if it's not on a block or in the hand
            if b not in on.values() and b != holding_block:
                deduced_atoms.append(to_string('ontable', [b]))
            # b is clear if it's not under a block or in the hand
            if b not in on.keys() and b != holding_block:
                deduced_atoms.append(to_string('clear', [b]))
            if holding_block is None:
                deduced_atoms.append(to_string('handempty', []))

        return atoms.union(deduced_atoms)


def random_column_instance(domain, n_blocks):
    from pddl2gym.pyperplan_planner.pddl.pddl import Problem, Predicate
    from random import shuffle
    assert list(domain.types.keys()) == ['object']
    assert n_blocks <= 26

    blocks = [chr(97+i) for i in range(n_blocks)]
    block_type = domain.types['object']

    column = blocks.copy()
    shuffle(column)

    init = []
    for b in blocks:
        init.append(Predicate('ontable', [(b, block_type)]))
        init.append(Predicate('clear', [(b, block_type)]))
    init.append(Predicate('handempty', []))

    goal = []
    for i in range(len(blocks)-1):
        goal.append(Predicate('on', [(column[i], block_type), (column[i+1], block_type)]))

    return Problem(name=f"random-column-{n_blocks}",
                   domain=domain,
                   objects={b: block_type for b in blocks},
                   init=init,
                   goal=goal)



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



# Register environments
registered_envs = register_envs()


if __name__ == "__main__":
    print("Registered environments:")
    for env_id in registered_envs:
        print(f"\t{env_id}")
        env = gym.make(env_id)