from typing import Sequence
from abc import ABC, abstractmethod
from jax.typing import ArrayLike
from jax import random, vmap
import jax.numpy as jnp
import equinox as eqx

class TensorBlock(eqx.Module, ABC):
    pass

class InceptionInputBlock(TensorBlock, ABC):

    @staticmethod
    @abstractmethod
    def forward(A, assignment):
        pass

    @staticmethod
    @abstractmethod
    def norm(A):
        pass


class InceptionPositiveInputBlock(InceptionInputBlock):
    """
        Represents an input block X_{ijkc} of shape (U, W, W, num_cats).
    """
    var: int # id of the variable

    U: int
    W: int
    num_cats: int

    key: ArrayLike
    A: ArrayLike = None

    def __post_init__(self):
        if self.A is None:
            params = jnp.exp(random.uniform(self.key, (self.U, self.W, self.num_cats)) * -2.0)
            params = params / params.sum(axis = -1, keepdims = True)
            self.A = jnp.log(params) # log domain params
            
    @staticmethod
    def forward(A, assignment):
        return A[:, None, :, assignment] + A[:, :, None, assignment]  # (U, W, W)
    
    @staticmethod
    def norm(A):
        log_params = A[:, None, :, :] + A[:, :, None, :]
        linear_params = jnp.exp(log_params)
        aggregated_params = jnp.sum(linear_params, axis=-1)

        return jnp.log(aggregated_params)
    

class InceptionRealInputBlock(InceptionInputBlock):
    """
        Represents an input block X_{ijkc} of shape (U, W, W, num_cats).
    """
    var: int # id of the variable

    U: int
    W: int
    num_cats: int

    key: ArrayLike
    A: ArrayLike = None

    def __post_init__(self):
        if self.A is None:
            params = jnp.exp(random.uniform(self.key, (self.U, self.W, self.num_cats)) * -2.0)
            self.A = params / params.sum(axis = -1, keepdims = True) # linear domain params
            
    @staticmethod
    def forward(A, assignment):
        val = A[:, None, :, assignment] * A[:, :, None, assignment]  # (U, W, W)
        return jnp.stack([jnp.log(jnp.abs(val)), jnp.angle(val)], axis=-1)  # (U, W, W, 2)
    
    @staticmethod
    def norm(A):
        linear_params = A[:, None, :, :] * A[:, :, None, :]
        aggregated_params = jnp.sum(linear_params, axis=-1)

        return jnp.stack([jnp.log(jnp.abs(aggregated_params)), jnp.angle(aggregated_params)], axis=-1)  # (U, W, W, 2)


class InceptionComplexInputBlock(InceptionInputBlock):
    """
        Represents an input block X_{ijkc} of shape (U, W, W, num_cats).
    """
    var: int # id of the variable

    U: int
    W: int
    num_cats: int

    key: ArrayLike
    A: ArrayLike = None

    def __post_init__(self):
        if self.A is None:
            self.A = random.uniform(self.key, (self.U, self.W, self.num_cats, 2))
            
    @staticmethod
    def forward(A, assignment):
        logmod = A[:, None, :, assignment, 0] + A[:, :, None, assignment, 0]
        arg = A[:, None, :, assignment, 1] - A[:, :, None, assignment, 1] # conjugate
        return jnp.stack([logmod, arg], axis=-1)  # (U, U, W, W, 2)
    
    @staticmethod
    def norm(A):
        logmod = A[:, None, :, :, 0] + A[:, :, None, :, 0]
        arg = A[:, None, :, :, 1] - A[:, :, None, :, 1]  # conjugate
        linear_params = jnp.exp(logmod + arg * 1j)
        aggregated_params = jnp.sum(linear_params, axis=-1)

        return jnp.stack([jnp.log(jnp.abs(aggregated_params)), jnp.angle(aggregated_params)], axis=-1)
    

class InceptionInnerBlock(TensorBlock, ABC):
    """
        Given input blocks of X^(r) for r = 1, ..., R, computes:
        Y_{ijk} = \sum_lmn A_{ijlm} conj(A_{ikln}) \prod_r X^(r)_{lmn}

        Dimensions:
        - U_in: l
        - W_in: m, n
        - U_out: i
        - W_out: j, k
    """
    @staticmethod
    @abstractmethod
    def forward(A: ArrayLike, input_blocks: ArrayLike) -> ArrayLike:
        pass



class InceptionPositiveInnerBlock(InceptionInnerBlock):
    U_in: int
    U_out: int
    W_in: int
    W_out: int
    
    chs: Sequence[TensorBlock]

    key: ArrayLike
    A: ArrayLike = None

    def __post_init__(self):
        if self.A is None:
            params = jnp.exp(random.uniform(self.key, (self.U_out, self.W_out, self.U_in, self.W_in)) * -2.0)
            params = params / params.sum(axis=(-2, -1), keepdims=True)
            self.A = jnp.log(params)  # log domain params

    @staticmethod
    def forward(A: ArrayLike, input_blocks: ArrayLike):
        """
        Input: shape (num_blocks, U_in, W_in, W_in)
        """

        prod_block = jnp.sum(input_blocks, axis=0) # (U_in, W_in, W_in)

        prod_block_max = jnp.max(prod_block)
        prod_block_norm = jnp.exp(prod_block - prod_block_max)  # (U_in, W_in, W_in)
        output_block_norm = jnp.einsum('ijlm,ikln,lmn->ijk', jnp.exp(A), jnp.exp(A), prod_block_norm)
        output_block = jnp.log(output_block_norm) + prod_block_max

        return output_block
    

class InceptionRealInnerBlock(InceptionInnerBlock):
    U_in: int
    U_out: int
    W_in: int
    W_out: int
    
    chs: Sequence[TensorBlock]

    key: ArrayLike
    A: ArrayLike = None

    def __post_init__(self):
        if self.A is None:
            params = jnp.exp(random.uniform(self.key, (self.U_out, self.W_out, self.U_in, self.W_in)) * -2.0) 
            self.A = params / params.sum(axis=(-2, -1), keepdims=True) # linear domain real params

    @staticmethod
    def forward(A: ArrayLike, input_blocks: ArrayLike):
        """
        Input: shape (num_blocks, U_in, W_in, W_in, 2) ; last dimension is for log-modulus and argument
        """
        prod_block = jnp.sum(input_blocks, axis=0) # (U_in, W_in, W_in, 2)

        prod_block_logmod, prod_block_arg = prod_block[..., 0], prod_block[..., 1]
        prod_block_logmod_max = jnp.max(prod_block_logmod)
        prod_block_norm = jnp.exp(prod_block_logmod - prod_block_logmod_max + 1j * prod_block_arg)
        output_block_norm = jnp.einsum('ijlm,ikln,lmn->ijk', A, A.conj(), prod_block_norm)
        output_block_logmod, output_block_arg = jnp.log(jnp.abs(output_block_norm)) + prod_block_logmod_max, jnp.angle(output_block_norm)

        return jnp.stack([output_block_logmod, output_block_arg], axis=-1)


class InceptionComplexInnerBlock(InceptionInnerBlock):
    U_in: int
    U_out: int
    W_in: int
    W_out: int
    
    chs: Sequence[TensorBlock]

    key: ArrayLike
    A: ArrayLike = None

    def __post_init__(self):
        if self.A is None:
            params = random.uniform(self.key, (self.U_out, self.W_out, self.U_in, self.W_in, 2))
            self.A = jnp.exp(params[..., 0] + 1j * 2 * jnp.pi * params[..., 1]) # linear domain complex weights

    @staticmethod
    def forward(A: ArrayLike, input_blocks: ArrayLike):
        """
        Input: shape (num_blocks, U_in, W_in, W_in, 2) ; last dimension is for log-modulus and argument
        """
        prod_block = jnp.sum(input_blocks, axis=0) # (U_in, W_in, W_in, 2)

        prod_block_logmod, prod_block_arg = prod_block[..., 0], prod_block[..., 1]
        prod_block_logmod_max = jnp.max(prod_block_logmod)
        prod_block_norm = jnp.exp(prod_block_logmod - prod_block_logmod_max + 1j * prod_block_arg)
        output_block_norm = jnp.einsum('ijlm,ikln,lmn->ijk', A, A.conj(), prod_block_norm)
        output_block_logmod, output_block_arg = jnp.log(jnp.abs(output_block_norm)) + prod_block_logmod_max, jnp.angle(output_block_norm)

        return jnp.stack([output_block_logmod, output_block_arg], axis=-1)


        









