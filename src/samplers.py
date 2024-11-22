import math
import numpy as np
import torch
from pythomata import SimpleDFA



class DFA:
    """Represents a DFA"""

    def __init__(
        self,
        num_nodes: int,
        alphabet: Tuple[str],
        transitions: Tuple[dict],
        rng: np.random.Generator,
    ):
        assert len(transitions) == num_nodes
        transitions = {i: v for i, v in enumerate(transitions)}
        dfa = SimpleDFA(
            states=set(list(range(num_nodes))),
            alphabet=set(alphabet),
            initial_state=0,
            accepting_states=set(list(range(num_nodes))),
            transition_function=transitions,
        )
        self.dfa = dfa
        self.rng = rng

    def _sorted_transitions(self):
        nodes = sorted(list(self.dfa._transition_function.keys()))
        transitions = []
        for node in nodes:
            node_transitions = self.dfa._transition_function[node]
            # sort node transitions by outgoing state
            transitions.append(
                tuple(sorted(node_transitions.items(), key=lambda item: item[1]))
            )
        return tuple(transitions)

    def _minimize(self):
        # minimize super
        self.dfa = self.dfa.minimize()
        return self

    def _trim(self):
        # trim super
        self.dfa = self.dfa.trim()
        return self

    def __hash__(self):
        # Here I assume the initial state is always the smallest node
        return hash(self._sorted_transitions())

    def __call__(self, word: str):
        current_node = self.dfa._initial_state
        for symbol in word.split():
            if symbol not in self.dfa._transition_function[current_node]:
                return False
            else:
                current_node = self.dfa._transition_function[current_node][symbol]
        return True

    def forward(self, word: str):
        current_node = self.dfa._initial_state
        for symbol in word.split():
            if symbol not in self.dfa._transition_function[current_node]:
                return None
            else:
                current_node = self.dfa._transition_function[current_node][symbol]
        return current_node

    def trace(self, word: str):
        current_node = self.dfa._initial_state
        path = [current_node]
        for symbol in word.split():
            try:
                self.dfa._transition_function[current_node]
            except:
                breakpoint()
            if symbol not in self.dfa._transition_function[current_node]:
                return path
            else:
                current_node = self.dfa._transition_function[current_node][symbol]
                path.append(current_node)
        return path

    def sample(self, length=1):
        """Samples a random word from the DFA"""
        current_node = self.dfa._initial_state
        word = ""
        for _ in range(length):
            outgoing_symbols = list(self.dfa._transition_function[current_node].keys())
            symbol = self.rng.choice(outgoing_symbols)
            word += symbol + " "
            current_node = self.dfa._transition_function[current_node][symbol]
        word = word.rstrip()
        return word


class RandomDFASampler:
    """Samples random DFAs given configs"""

    num_nodes: int
    alphabet: Tuple[str]
    max_outgoing_edge: int
    rng: np.random.Generator = None

    def __init__(
        self,
        num_nodes: int,
        alphabet: Tuple[str],
        max_outgoing_edge: int,
        seed: int = 42,
    ):
        self.num_nodes = num_nodes
        self.alphabet = alphabet
        self.max_outgoing_edge = max_outgoing_edge
        self.rng = np.random.default_rng(seed)

    def sample(self):
        transitions = [{} for _ in range(self.num_nodes)]
        for node in range(self.num_nodes):
            num_transitions = self.rng.integers(1, self.max_outgoing_edge)
            transition_symbols = self.rng.choice(
                self.alphabet, size=num_transitions, replace=False
            )
            # exclude self loops
            possible_nodes = [n for n in range(self.num_nodes) if n != node]
            transition_nodes = self.rng.choice(
                possible_nodes, size=num_transitions, replace=False
            )
            transitions[node] = dict(zip(transition_symbols, transition_nodes))
        dfa_rng = np.random.default_rng(self.rng.integers(0, 2**32))
        return DFA(self.num_nodes, self.alphabet, tuple(transitions), dfa_rng)

'''
class DataSampler:
    def __init__(self, n_dims):
        self.n_dims = n_dims

    def sample_xs(self):
        raise NotImplementedError


def get_data_sampler(data_name, n_dims, **kwargs):
    names_to_classes = {
        "uniform": UniformSampler, #,
        "gaussian": GaussianSampler,
    }
    if data_name in names_to_classes:
        sampler_cls = names_to_classes[data_name]
        return sampler_cls(n_dims, **kwargs)
    else:
        print("Unknown sampler")
        raise NotImplementedError


def sample_transformation(eigenvalues, normalize=False):
    n_dims = len(eigenvalues)
    U, _, _ = torch.linalg.svd(torch.randn(n_dims, n_dims))
    t = U @ torch.diag(eigenvalues) @ torch.transpose(U, 0, 1)
    if normalize:
        norm_subspace = torch.sum(eigenvalues**2)
        t *= math.sqrt(n_dims / norm_subspace)
    return t


class GaussianSampler(DataSampler):
    def __init__(self, n_dims, bias=None, scale=None, fixed_pts=-1):
        super().__init__(n_dims)
        self.bias = bias
        self.scale = scale

        self.fixed_pts=fixed_pts

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        if self.fixed_pts > 0:
            torch.manual_seed(n_points)
            xs_b = torch.randn(b_size, n_points, self.n_dims)
        if seeds is None:
            xs_b = torch.randn(b_size, n_points, self.n_dims)
        else:
            xs_b = torch.zeros(b_size, n_points, self.n_dims)
            generator = torch.Generator()
            assert len(seeds) == b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                xs_b[i] = torch.randn(n_points, self.n_dims, generator=generator)
        if self.scale is not None:
            xs_b = xs_b @ self.scale
        if self.bias is not None:
            xs_b += self.bias
        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0
        return xs_b #increasing the variance to sample from N(0, 4I), will see if this breaks performance

class UniformSampler(DataSampler):
    def __init__(self, n_dims, start=-1, end=1):
        super().__init__(n_dims)
        self.start = start
        self.end = end

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        xs_b = 2*torch.rand(b_size, n_points, self.n_dims) - 1
        assert torch.min(xs_b) >= -1 and torch.max(xs_b) <= 1
        return xs_b #increasing the variance to sample from N(0, 4I), will see if this breaks performance
'''