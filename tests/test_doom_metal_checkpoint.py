import json
import sys

import numpy as np
import pytest

from test_doom_metal_parity import paired_brains


pytestmark=pytest.mark.skipif(sys.platform!='darwin',reason='Metal requires macOS')


def advance(model,milliseconds):
    return model.step([],milliseconds,learning=True,stimulation=([0,3],20),lamina_bias=0)[0]


def test_cpu_checkpoint_continues_on_metal_with_float64_state(tmp_path):
    cpu,metal=paired_brains(tmp_path)
    cpu.adaptation_jump=metal.adaptation_jump=0
    advance(cpu,30);path=tmp_path/'cpu.npz';cpu.checkpoint(path)
    eligibility_before=cpu.eligibility.copy()
    expected=advance(cpu,20)
    assert cpu.eligibility[0]>eligibility_before[0]
    metal.restore(path);actual=advance(metal,20)
    np.testing.assert_array_equal(actual,expected)
    np.testing.assert_array_equal(metal.eligibility,cpu.eligibility)
    np.testing.assert_array_equal(metal.eligibility_last,cpu.eligibility_last)
    np.testing.assert_allclose(metal.weight,cpu.weight,rtol=1e-4,atol=0)


def test_metal_checkpoint_continues_on_cpu(tmp_path):
    cpu,metal=paired_brains(tmp_path)
    cpu.adaptation_jump=metal.adaptation_jump=0
    advance(metal,30);path=tmp_path/'metal.npz';metal.checkpoint(path)
    expected=advance(metal,20)
    cpu.restore(path);actual=advance(cpu,20)
    np.testing.assert_array_equal(actual,expected)
    np.testing.assert_array_equal(cpu.eligibility,metal.eligibility)
    np.testing.assert_array_equal(cpu.eligibility_last,metal.eligibility_last)


def test_checkpoint_records_producer_without_requiring_same_backend(tmp_path):
    _,metal=paired_brains(tmp_path);advance(metal,10)
    path=tmp_path/'metal.npz';metal.checkpoint(path)
    with np.load(path,allow_pickle=False) as archive:
        metadata=json.loads(str(archive['metadata']))
    assert metal.backend._last_materialization_reason=='checkpoint'
    assert metadata['producer_backend']=='metal'
    assert metadata['backend']['device']['name']=='Apple M4 Pro'


def test_metal_checkpoint_repeats_bitwise(tmp_path):
    _,first=paired_brains(tmp_path/'first');_,second=paired_brains(tmp_path/'second')
    advance(first,30);path=tmp_path/'metal.npz';first.checkpoint(path)
    expected=advance(first,20);second.restore(path);actual=advance(second,20)
    np.testing.assert_array_equal(actual,expected)
    first.backend.materialize('test');second.backend.materialize('test')
    for name in ['weight',*first.fields]:
        np.testing.assert_array_equal(getattr(second,name),getattr(first,name),err_msg=name)


def test_reset_replaces_stale_resident_state(tmp_path):
    _,reset_model=paired_brains(tmp_path/'reset')
    _,fresh=paired_brains(tmp_path/'fresh')
    reset_model.weights_frozen=fresh.weights_frozen=True
    reset_model.step([],10,stimulation=([0],20),lamina_bias=0)
    reset_model.v.fill(np.nan);reset_model.queue.fill(-1)
    reset_model.reset()
    expected,_=fresh.step([],10,stimulation=([3],20),lamina_bias=0)
    actual,_=reset_model.step([],10,stimulation=([3],20),lamina_bias=0)
    np.testing.assert_array_equal(actual,expected)
    reset_model.backend.materialize('test');fresh.backend.materialize('test')
    for name in ['weight',*fresh.fields]:
        np.testing.assert_array_equal(getattr(reset_model,name),getattr(fresh,name),err_msg=name)


def test_restore_replaces_stale_resident_state(tmp_path):
    _,source=paired_brains(tmp_path/'source')
    _,restored=paired_brains(tmp_path/'restored')
    source.weights_frozen=restored.weights_frozen=True
    source.step([],10,stimulation=([0],20),lamina_bias=0)
    path=tmp_path/'resident.npz';source.checkpoint(path)
    restored.step([],10,stimulation=([3],20),lamina_bias=0)
    restored.v.fill(np.nan);restored.queue.fill(-1)
    restored.restore(path)
    expected,_=source.step([],10,stimulation=([0,3],18),lamina_bias=0)
    actual,_=restored.step([],10,stimulation=([0,3],18),lamina_bias=0)
    np.testing.assert_array_equal(actual,expected)
    source.backend.materialize('test');restored.backend.materialize('test')
    for name in ['weight',*source.fields]:
        np.testing.assert_array_equal(getattr(restored,name),getattr(source,name),err_msg=name)
