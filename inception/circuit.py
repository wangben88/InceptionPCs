from __future__ import annotations
from collections import defaultdict
from typing import Dict, Callable, Hashable, Iterable, Iterator, Any
import jax.numpy as jnp
from jax import vmap
import equinox as eqx
from abc import ABC

from inception.blocks import TensorBlock, InceptionInnerBlock, InceptionInputBlock

def split_iterable(fn: Callable[..., Hashable], l: Iterable[Any]) -> Dict[Hashable, Iterator[Any]]:
    """
    Splits a list into multiple lists based on the function value fn.
    """
    result = defaultdict(list)
    positions = defaultdict(list)
    for pos, item in enumerate(l):
        result[fn(item)].append(item)
        positions[fn(item)].append(pos)
    return result, positions


class InceptionCircuit(eqx.Module, ABC):
    """
    
    Assumes that the blocks in the circuit are connected in a tree structure (rather than a DAG).
    """
    root_block: TensorBlock

    def make_layers(self):
        """
        Returns a list of layers, where:
            - layers[0] contains the input blocks
            - layers[i] represents layer i and consists of tuples (block, ch_layer_id_and_pos), where:
                - block is an inner block
                - ch_layer_id_and_pos is a list of tuples (layer_id, pos) for each child block
        """
        layers = [[]]
        def dfs(block: TensorBlock):
            if isinstance(block, InceptionInputBlock):
                layers[0].append(block)
                return 0, len(layers[0]) - 1
            elif isinstance(block, InceptionInnerBlock):
                max_layer_id = 0
                ch_layer_id_and_pos = []
                for ch in block.chs:
                    layer_id, pos = dfs(ch)
                    max_layer_id = max(max_layer_id, layer_id)
                    ch_layer_id_and_pos.append((layer_id, pos))
                if len(layers) <= max_layer_id + 1:
                    layers.append([(block, ch_layer_id_and_pos)])
                    return max_layer_id + 1, 0
                else:
                    layers[max_layer_id + 1].append((block, ch_layer_id_and_pos))
                    return max_layer_id + 1, len(layers[max_layer_id + 1]) - 1
        dfs(self.root_block)
        return layers
    

    def forward(self, assignment):
        """
        Computes the output of the circuit for a given assignment.
        """
        layers = self.make_layers()

        all_layer_values = []
        # Input layer
        layer_vars = jnp.array([block.var for block in layers[0]], dtype=jnp.int32) 
        layer_assignment = assignment[layer_vars] # (len(input_sublayer),)
        weights = jnp.stack([block.A for block in layers[0]])
        all_layer_values.append(
            vmap(type(layers[0][0]).forward, in_axes=(0, 0))(weights, layer_assignment)
        )

        # Inner Layers
        for layer in layers[1:]:
            new_layer_values = [None] * len(layer)

            # split layer by number of children (to enable vectorization)
            num_chs_to_blocks = defaultdict(list)
            for pos, (block, chs_layer_id_and_pos) in enumerate(layer):
                num_chs_to_blocks[len(block.chs)].append((block, pos, chs_layer_id_and_pos))

            for sublayer in num_chs_to_blocks.values():
                weights, chs_values, positions = [], [], []
                for block, pos, chs_layer_id_and_pos in sublayer:
                    chs_values.append(jnp.stack([all_layer_values[layer_id][pos] for (layer_id, pos) in chs_layer_id_and_pos]))
                    weights.append(block.A)
                    positions.append(pos)
                weights, chs_values = jnp.stack(weights, axis=0), jnp.stack(chs_values, axis=0)
                sublayer_values = vmap(type(sublayer[0][0]).forward, in_axes=(0, 0))(weights, chs_values)
                for pos, value in zip(positions, sublayer_values):
                    new_layer_values[pos] = value

            all_layer_values.append(new_layer_values)
        
        return all_layer_values[-1][0]  

    def norm(self):
        layers = self.make_layers()

        all_layer_values = []
        # Input layer
        weights = jnp.stack([block.A for block in layers[0]])
        all_layer_values.append(
            vmap(type(layers[0][0]).norm, in_axes=0)(weights)
        )

        # Inner Layers
        for layer in layers[1:]:
            new_layer_values = [None] * len(layer)

            # split layer by number of children (to enable vectorization)
            num_chs_to_blocks = defaultdict(list)
            for pos, (block, chs_layer_id_and_pos) in enumerate(layer):
                num_chs_to_blocks[len(block.chs)].append((block, pos, chs_layer_id_and_pos))

            for sublayer in num_chs_to_blocks.values():
                weights, chs_values, positions = [], [], []
                for block, pos, chs_layer_id_and_pos in sublayer:
                    chs_values.append(jnp.stack([all_layer_values[layer_id][pos] for (layer_id, pos) in chs_layer_id_and_pos]))
                    weights.append(block.A)
                    positions.append(pos)
                weights, chs_values = jnp.stack(weights, axis=0), jnp.stack(chs_values, axis=0)
                sublayer_values = vmap(type(sublayer[0][0]).forward, in_axes=(0, 0))(weights, chs_values)
                for pos, value in zip(positions, sublayer_values):
                    new_layer_values[pos] = value

            all_layer_values.append(jnp.stack(new_layer_values, axis=0))
        
        return all_layer_values[-1][0]