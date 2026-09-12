"""Paired native arithmetic regressions, separate from biological validation."""
import sys

import numpy as np
import pytest

from test_doom_metal_decay import _edgeless_pair


pytestmark=pytest.mark.skipif(sys.platform!='darwin',reason='Metal requires macOS')


def _assert_bins_match(cpu,metal,bins):
    for bin_index in range(bins):
        for brain in [cpu,metal]:
            brain.counts.fill(0)
            brain.backend.advance(100)
            brain.backend.materialize('arithmetic-regression')
        context=f'100-tick bin {bin_index+1}'
        assert cpu.cursor==metal.cursor==100*(bin_index+1),context
        np.testing.assert_array_equal(cpu.counts,metal.counts,err_msg=context)
        for name in ['v','g','adaptation']:
            np.testing.assert_array_equal(getattr(cpu,name).view(np.uint32),
                getattr(metal,name).view(np.uint32),err_msg=f'{name}: {context}')
        for name in ['refractory','last']:
            np.testing.assert_array_equal(getattr(cpu,name),getattr(metal,name),
                err_msg=f'{name}: {context}')


@pytest.mark.parametrize('tau',[200.,137.])
@pytest.mark.parametrize('bins',[1,200])
def test_driven_relaxation_matches_cpu_from_first_dark_bin(tmp_path,tau,bins):
    # Separate multiply/add rounding in voltage relaxation breaks bitwise
    # agreement even with default rest and zero conductance/adaptation.
    cpu,metal=_edgeless_pair(tmp_path,tau)
    try:
        for brain in [cpu,metal]:
            brain.drive[:]=[12.,9.87,11.3125,3.75]
        _assert_bins_match(cpu,metal,bins)
    finally:
        cpu.backend.close();metal.backend.close()


@pytest.mark.parametrize('tau',[200.,137.])
@pytest.mark.parametrize('bins',[1,200])
def test_composite_voltage_adaptation_matches_native_cpu(tmp_path,tau,bins):
    # A differently rounded adaptation correction can diverge independently
    # of the simple driven case. Seed all state before the initial upload.
    cpu,metal=_edgeless_pair(tmp_path,tau)
    try:
        for brain in [cpu,metal]:
            brain.drive[:]=[12.,9.87,11.3125,3.75]
            brain.rest[2:]=[-60.,-60.]
            brain.v[:]=[-51.,-52.,-59.,-58.]
            brain.g[:]=[0.,.5,1.,1.25]
            brain.adaptation[:]=[0.,.75,4.,.7]
        _assert_bins_match(cpu,metal,bins)
    finally:
        cpu.backend.close();metal.backend.close()
