from subgrapher.quartet_extracter import LogDomain
# from hypothesis import given, strategies as st
import numpy as np
from hypothesis import given
from copy import copy
from hypothesis.extra import numpy as nps
from hypothesis import strategies as st
from hypothesis.extra.array_api import make_strategies_namespace

@given(a=nps.arrays(dtype=np.float64, shape=st.integers(1, 10), elements=st.floats(-1, 1)))
def test_simple_addition(a):
    a[a <= 0] = 0
    A = a / np.sum(a) if np.sum(a) > 0 else a
    x = LogDomain()
    x.consume(A)
    for i in range(len(A)):
        todivide = LogDomain.from_array(A[:i])
        assert np.allclose(x.divide_by(todivide).item(), np.prod(A[i:]),equal_nan=True), f"{x.divide_by(todivide).item()} != {np.prod(A[i:])}"

@given(a=nps.arrays(dtype=np.float64, shape=st.integers(1, 10), elements=st.floats(-1, 1)))
def test_many_properties(a):
    a[a <= 0] = 0
    A = a / np.sum(a) if np.sum(a) > 0 else a
    for i in range(len(A)):
        for j in range(i, len(A)):
            x = LogDomain()
            x.consume(A[i:j])
            assert np.allclose(x.item(), np.prod(A[i:j]), equal_nan=True), f"{x.item()} != {np.prod(A[i:j])}, {x}, {A[i:j]}"

            y = LogDomain()
            y.consume(A[:j])
            r = y.divide_by(LogDomain.from_array(A[:i])).item()
            assert np.allclose(r, np.prod(A[i:j]), equal_nan=True), f"{r} != {np.prod(A[i:j])}, {y}, {A[:j]}, {A[:i]}"