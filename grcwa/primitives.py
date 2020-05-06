from functools import partial
import numpy.linalg as npla
import numpy  as anp
from autograd.extend import defvjp,primitive

# Some formulas are from
# "An extended collection of matrix derivative results
#  for forward and reverse mode algorithmic differentiation"
# by Mike Giles
# https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf

# transpose by swapping last two dimensions
def T(x): return anp.swapaxes(x, -1, -2)

_dot = partial(anp.einsum, '...ij,...jk->...ik')

# batched diag 
_diag = lambda a: anp.eye(a.shape[-1])*a 

# batched diagonal, similar to matrix_diag in tensorflow
def _matrix_diag(a):
    reps = anp.array(a.shape)
    reps[:-1] = 1
    reps[-1] = a.shape[-1]
    newshape = list(a.shape) + [a.shape[-1]]
    return _diag(anp.tile(a, reps).reshape(newshape))

# https://arxiv.org/pdf/1701.00392.pdf Eq(4.77)
# Note the formula from Sec3.1 in https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf is incomplete

inv = primitive(anp.linalg.inv)
def grad_inv(ans, x):
    return lambda g: -_dot(_dot(T(ans), g), T(ans))
defvjp(inv, grad_inv)
    
eig = primitive(anp.linalg.eig)
def grad_eig(ans, x):
    """Gradient of a general square (complex valued) matrix"""
    e, u = ans # eigenvalues as 1d array, eigenvectors in columns
    n = e.shape[-1]
    def vjp(g):
        ge, gu = g
        ge = _matrix_diag(ge)
        f = 1/(e[..., anp.newaxis, :] - e[..., :, anp.newaxis] + 1.e-20)
        f -= _diag(f)
        ut = anp.swapaxes(u, -1, -2)
        r1 = f * _dot(ut, gu)
        r2 = -f * (_dot(_dot(ut, anp.conj(u)), anp.real(_dot(ut,gu)) * anp.eye(n)))
        r = _dot(_dot(inv(ut), ge + r1 + r2), ut)
        if not anp.iscomplexobj(x):
            r = anp.real(r)
            # the derivative is still complex for real input (imaginary delta is allowed), real output
            # but the derivative should be real in real input case when imaginary delta is forbidden
        return r
    return vjp
defvjp(eig, grad_eig)
