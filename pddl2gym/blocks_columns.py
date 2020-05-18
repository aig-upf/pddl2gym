from gridenvs.world import GridObject
from gridenvs.utils import Colors
from pddl2gym.env import PDDLGridEnv, PDDLRepresentation
from pddl2gym.simulator import PDDLProblemSimulator
from pddl2gym.utils import to_tuple, to_atoms_dict, get_atom_fixed_param, files_in_dir, parse_problem
import gym
import os


class BlocksColumns(PDDLRepresentation):
    def get_n_actions(self, problem):
        return len(problem.objects_by_type["column"])

    def get_reduced_actions(self, problem, atoms):
        atoms_dict = to_atoms_dict(atoms)

        # one action per column
        actions = []
        for col in problem.objects_by_type["column"]:
            bp = self._get_block_pile(atoms_dict, col)
            if "holding" in atoms_dict:
                hb = atoms_dict["holding"][0][0]
                if len(bp) == 0:
                    action = ("putdown", (hb, col))  # empty column
                else:
                    action = ("stack", (hb, bp[-1])) # column has at least one block, stack on top one
            else:
                assert "hand-free" in atoms_dict
                if len(bp) == 0:
                    action = None  # empty column, we cannot pick any block from there
                elif len(bp) == 1:
                    action = ("pickup", (bp[-1], col))  # only one block
                else:
                    action = ("unstack", (bp[-1], bp[-2]))

            actions.append(action)
        return actions

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

    def get_gridstate(self, problem, atoms):
        columns = problem.objects_by_type["column"]

        # Define block colors
        salients = []
        if "salient" in problem.objects_by_type:
            salients = problem.objects_by_type["salient"]
            assert len(salients)< 3
            salient_colors = [Colors.red, Colors.green, Colors.blue]
        blocks = problem.objects_by_type["block"]
        block_colors = {b: c for b, c in zip(blocks, self.colors)}
        block_colors.update({s: c for s, c in zip(salients, salient_colors)})

        # Get grid size
        all_blocks = salients + blocks
        grid_size = (len(all_blocks), len(columns + 1))

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

        assert len(col_bottom.keys()) == len(columns)

        objects = []
        for i, c in enumerate(columns):
            j = 0
            b = col_bottom[c]
            while b is not None:
                objects.append(GridObject(name=b,
                                          pos=(i, grid_size[1] - j - 1),
                                          rgb=block_colors[b]))
                b = on[b]
                j += 1

        if holding_block is not None:
            objects.append(GridObject(name=holding_block,
                                      pos=(grid_size[0] - 1, 0),
                                      rgb=block_colors[holding_block]))

        return objects


def blocks_columns(domain_file, instance_file, path=None):
    simulator = PDDLProblemSimulator(parse_problem(domain_file, instance_file, path))
    return PDDLGridEnv(simulator=simulator, representation=BlocksColumns(), fixed_init_state=True, max_moves=100)


def register_envs():
    registered_envs = {}
    path = os.path.join(os.path.dirname(__file__), "pddl/blocks_columns")
    domain_path = os.path.join(path, "domain.pddl")
    for problem_file in sorted(files_in_dir(path)):
        if problem_file.endswith(".pddl") and problem_file != "domain.pddl":
            if problem_file.startswith("probBLOCKS"):
                _, i, j = problem_file.split(".")[0].split("-")  # output example: ['probBLOCKS', '4', '0']
                env_id = f'PDDL_BlocksColumns_{i}_{j}-v0'
            elif problem_file.startswith("instance"):
                _, i, t, _, _ = problem_file.split(".")[0].split("_")  # output example: ['instance', '15', 'clear', 'x', '1']
                env_id = f'PDDL_BlocksColumns_{t}_{i}-v0'
            elif problem_file.startswith("target"):
                _, i, j = problem_file.split(".")[0].split("-")  # output example: ['target', '15', '0']
                env_id = f'PDDL_BlocksColumns_target_{i}_{j}-v0'
            else:
                raise NotImplementedError(f"Environment id not specified for problem file {problem_file}")

            try:
                instance_path = os.path.join(path, problem_file)
                gym.register(id=env_id,
                             entry_point='pddl2gym.blocks_columns:blocks_columns',
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