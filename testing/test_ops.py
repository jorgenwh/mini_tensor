import pytest
import numpy as np
import cupy as cp
import minitensor as mt
from .xp_ops import np_relu, cp_relu


@pytest.fixture
def npf32_array():
    return np.random.randn(500).astype(np.float32)

@pytest.fixture
def cpf32_array():
    return cp.random.randn(500).astype(np.float32)


def test_exp_cpu(npf32_array):
    mtf32_array = mt.from_xp(npf32_array)
    assert np.allclose(np.exp(npf32_array), mt.exp(mtf32_array).as_numpy())

def test_exp_cuda(cpf32_array):
    mtf32_array = mt.from_xp(cpf32_array)
    assert cp.allclose(cp.exp(cpf32_array), mt.exp(mtf32_array).as_cupy())

def test_relu_cpu(npf32_array):
    mtf32_array = mt.from_xp(npf32_array)
    np.testing.assert_array_equal(np_relu(npf32_array), mt.relu(mtf32_array).as_numpy())

def test_relu_cuda(cpf32_array):
    mtf32_array = mt.from_xp(cpf32_array)
    cp.testing.assert_array_equal(cp_relu(cpf32_array), mt.relu(mtf32_array).as_cupy())

def test_sum_cpu(npf32_array):
    mtf32_array = mt.from_xp(npf32_array)
    assert abs(np.sum(npf32_array) - mt.sum(mtf32_array)) < 1e-4

def test_sum_cuda(cpf32_array):
    mtf32_array = mt.from_xp(cpf32_array)
    assert abs(cp.sum(cpf32_array) - mt.sum(mtf32_array)) < 1e-4
