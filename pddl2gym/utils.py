from collections import defaultdict


def to_tuple(str_predicate):
    s = str_predicate[1:-1] if str_predicate[0] == '(' else str_predicate
    res = s.split(" ")
    return res[0], tuple(res[1:])

def to_string(p, params=None):
    if params is None:
        p, params = p
    return f"({p} {' '.join(params)})"

def get_objects_by_type(problem):
    d = defaultdict(list)
    for o, t in problem.objects.items():
        d[t.name].append(o)
    return {k: sorted(v) for k, v in d.items()}


def state_to_atoms_dict(state):
    atoms = defaultdict(list)
    for atom in state:
        name, params = to_tuple(atom)
        atoms[name].append(params)
    return dict(atoms)

def get_atom_fixed_param(atoms, name, param_idx, param_value):
    if name in atoms:
        for params in atoms[name]:
            if param_value == params[param_idx]:
                return params
    return None