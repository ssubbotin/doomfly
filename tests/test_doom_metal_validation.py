import numpy as np

from doom_learning_v6.metal.validate import (
    compare_decisions,
    evaluate_gates,
    spike_metrics,
    weight_metrics,
)


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
