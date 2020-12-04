from collections import defaultdict
import os
from pyperplan.pddl.parser import Parser


def parse_domain(domain_file, path=None):
    if path is not None:
        domain_file = os.path.join(path, domain_file)
    parser = Parser(domain_file, probFile=None)
    return parser.parse_domain()


def parse_problem(domain_file, problem_file, path=None):
    if path is not None:
        domain_file = os.path.join(path, domain_file)
        problem_file = os.path.join(path, problem_file)
    parser = Parser(domain_file, problem_file)
    domain = parser.parse_domain()
    problem = parser.parse_problem(domain)  # domain can be found as an attribute of problem
    problem.objects_by_type = get_objects_by_type(problem)
    return problem


def to_tuple(str_predicate):
    s = str_predicate.lstrip('(').rstrip(')').strip()
    res = s.split(" ")
    return res[0], tuple(res[1:])


def to_string(p, params=None):
    if params is None:
        p, params = p
    assert type(params) in (list, tuple)
    return f"({p} {' '.join(params)})"


def get_objects_by_type(problem):
    d = defaultdict(list)
    for o, t in problem.objects.items():
        d[t.name].append(o)
    return {k: sorted(v) for k, v in d.items()}


def to_atoms_dict(atoms):
    atoms_dict = defaultdict(list)
    for a in atoms:
        name, params = to_tuple(a)
        atoms_dict[name].append(params)
    return dict(atoms_dict)


def get_atom_fixed_param(atoms, name, param_idx, param_value):
    if name in atoms:
        for params in atoms[name]:
            if param_value == params[param_idx]:
                return params
    return None


def files_in_dir(path):
    return next(os.walk(path))[2]