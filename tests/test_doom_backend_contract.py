import numpy as np
import pytest

from test_doom_learning_v6 import brain


def test_cpu_is_default_and_unknown_backend_fails(tmp_path):
    default=brain(tmp_path)
    assert default.backend.name=='cpu'
    with pytest.raises(ValueError,match='Unknown neural backend'):
        brain(tmp_path,backend='cuda')


def test_explicit_cpu_backend_preserves_existing_trace(tmp_path):
    default=brain(tmp_path)
    explicit=brain(tmp_path,backend='cpu')
    actual,_=default.step([],20,stimulation=([0],20),lamina_bias=0)
    expected,_=explicit.step([],20,stimulation=([0],20),lamina_bias=0)
    np.testing.assert_array_equal(actual,expected)
    np.testing.assert_array_equal(default.v,explicit.v)
    np.testing.assert_array_equal(default.g,explicit.g)
