import sys

import numpy as np
import pytest

from test_doom_learning_v6 import brain


pytestmark=pytest.mark.skipif(sys.platform!='darwin',reason='Metal requires macOS')


def test_metal_state_round_trip_preserves_all_neural_arrays(tmp_path):
    model=brain(tmp_path,backend='metal')
    before={name:getattr(model,name).copy() for name in model.fields}
    before['weight']=model.weight.copy()
    model.backend.ensure_initialized()
    model.backend.sync_for_checkpoint()
    for name,expected in before.items():
        np.testing.assert_array_equal(getattr(model,name),expected,err_msg=name)
    metadata=model.backend.metadata()
    assert metadata['name']=='metal'
    assert metadata['device']['name']=='Apple M4 Pro'


def test_metal_restore_rejects_duplicate_delayed_neuron(tmp_path):
    model=brain(tmp_path,backend='metal')
    model.queue_count[0]=2
    model.queue[0,:2]=[1,1]
    with pytest.raises(RuntimeError,match='duplicate delayed neuron'):
        model.backend.ensure_initialized()
