import sys

import numpy as np
import pytest

from doom_learning_v6.metal.validate import (
    _state_digest,
    compare_decisions,
    evaluate_gates,
    spike_metrics,
    weight_metrics,
)
from test_doom_metal_parity import paired_brains


def test_parity_metrics_have_fixed_event_semantics():
    cpu={(0,1),(2,3),(4,5)}
    metal={(0,1),(2,3),(4,6)}
    report=spike_metrics(cpu,metal,np.array([1,2,3]),np.array([1,2,3]))
    assert report=={'jaccard':.5,'total_spike_fraction':0.,
        'rate_correlation':1.,'within_one_tick_fraction':1.}


def test_empty_spike_and_zero_rate_inputs_are_finite():
    report=spike_metrics(set(),set(),np.zeros(3),np.zeros(3))
    assert report=={'jaccard':1.,'total_spike_fraction':0.,
        'rate_correlation':1.,'within_one_tick_fraction':1.}
    assert spike_metrics(set(),set(),np.zeros(2),np.array([0.,1.]))['rate_correlation']==0.


@pytest.mark.parametrize('consumer',['validation','benchmark'])
def test_portable_identity_keeps_math_configuration(consumer):
    from doom_learning_v6.metal.validate import _portable_metal_metadata
    from doom_learning_v6.metal.benchmark import portable_backend_metadata
    configuration={'metal_compile_flags':['-std=macos-metal2.4',
        '-fno-fast-math','-ffp-contract=off'],'abi_version':3}
    metadata={'name':'metal','schema':2,'abi_version':3,'sources':{},'binaries':{},
        'compiler':'Apple clang','sdk':'26.0','macos':'26.6.2','architecture':'arm64',
        'metal_language':'macos-metal2.4','build_configuration':configuration,
        'device':{'name':'Apple M4 Pro','registry_id':123},
        'commands':[['compiler','private-machine-path']]}
    function=_portable_metal_metadata if consumer=='validation' else portable_backend_metadata
    portable=function(metadata)
    assert portable['build_configuration']==configuration
    assert 'commands' not in portable and 'registry_id' not in portable['device']


def test_decoder_comparison_requires_exact_decisions():
    decisions=[{'turn':0.,'forward':2.,'attack':False}]
    assert compare_decisions(decisions,[dict(decisions[0])])
    assert not compare_decisions(decisions,[{**decisions[0],'turn':1e-12}])


def test_weight_metrics_report_maximum_relative_error():
    report=weight_metrics(np.array([2.,-4.]),np.array([2.0002,-3.9996]),rtol=1e-4)
    assert np.isclose(report['maximum_relative_error'],1e-4)
    assert report['within_rtol']
    assert not weight_metrics(np.array([0.]),np.array([1.]),rtol=1e-4)['within_rtol']


def test_gate_boundaries_are_inclusive():
    report=evaluate_gates({'jaccard':.995,'total_spike_fraction':.005,
        'rate_correlation':.999,'within_one_tick_fraction':.999},
        decoder_equal=True,scientific_gates_equal=True,
        weights={'maximum_relative_error':1e-4,'within_rtol':True})
    assert all(report.values())
    report=evaluate_gates({'jaccard':.994999,'total_spike_fraction':.005,
        'rate_correlation':.999,'within_one_tick_fraction':.999},
        decoder_equal=True,scientific_gates_equal=True,
        weights={'maximum_relative_error':1e-4,'within_rtol':True})
    assert not report['spike_jaccard']
    assert not report['passed']


@pytest.mark.skipif(sys.platform!='darwin',reason='Metal requires macOS')
def test_state_digest_materializes_resident_metal_state(tmp_path):
    _,metal=paired_brains(tmp_path)
    metal.weights_frozen=True
    metal.step([],10,stimulation=([0],20),lamina_bias=0)
    metal.v.fill(np.nan)
    first=_state_digest(metal)
    assert np.isfinite(metal.v).all()
    assert metal.backend._last_materialization_reason=='validation-digest'
    assert _state_digest(metal)==first
