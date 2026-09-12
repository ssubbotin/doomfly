"""Paired native arithmetic regressions, separate from biological validation."""
import sys

import numpy as np
import pytest

from test_doom_metal_decay import _edgeless_pair


requires_metal=pytest.mark.skipif(sys.platform!='darwin',reason='Metal requires macOS')


def _assert_composite_conductance(cpu,metal,context):
    # Metal permits a finite nonzero CPU subnormal to flush to zero. Keep
    # every other value bit-exact; this helper only serves the edgeless fixture.
    allowed=np.isfinite(cpu)&(cpu!=0)&(np.abs(cpu)<np.finfo(np.float32).tiny)&(metal==0)
    np.testing.assert_array_equal(cpu[~allowed].view(np.uint32),
        metal[~allowed].view(np.uint32),err_msg=f'g: {context}')


def test_composite_underflow_accepts_only_cpu_subnormal_flushed_to_zero():
    cpu=np.array([4320706,5400880,0,0x00800000,0x3f800000],dtype=np.uint32).view(np.float32)
    metal=np.array([0,0,0,0x00800000,0x3f800000],dtype=np.uint32).view(np.float32)
    _assert_composite_conductance(cpu,metal,'measured bin 44')


@pytest.mark.parametrize('cpu_bits,metal_bits',[
    (0x00800000,0),             # Smallest normal must stay bit-exact.
    (0x3f800000,0x3f800001),    # One-ULP normal mismatch must fail.
    (4320706,1),               # Nonzero Metal subnormal is not a flush.
    (4320706,5400880),          # Distinct CPU/Metal subnormals must fail.
    (0,0x80000000),             # CPU zero must retain its exact bits.
    (0x7f800000,0),             # Infinity is outside the finite contract.
    (0x7fc00000,0),             # NaN is outside the finite contract.
])
def test_composite_underflow_rejects_other_mismatches(cpu_bits,metal_bits):
    cpu=np.array([cpu_bits],dtype=np.uint32).view(np.float32)
    metal=np.array([metal_bits],dtype=np.uint32).view(np.float32)
    with pytest.raises(AssertionError):
        _assert_composite_conductance(cpu,metal,'negative control')


def _assert_bins_match(cpu,metal,bins,*,composite_underflow=False):
    for bin_index in range(bins):
        for brain in [cpu,metal]:
            brain.counts.fill(0)
            brain.backend.advance(100)
            brain.backend.materialize('arithmetic-regression')
        context=f'100-tick bin {bin_index+1}'
        assert cpu.cursor==metal.cursor==100*(bin_index+1),context
        np.testing.assert_array_equal(cpu.counts,metal.counts,err_msg=context)
        for name in ['v','g','adaptation']:
            if name=='g' and composite_underflow and bin_index>=43:
                _assert_composite_conductance(cpu.g,metal.g,context)
                continue
            np.testing.assert_array_equal(getattr(cpu,name).view(np.uint32),
                getattr(metal,name).view(np.uint32),err_msg=f'{name}: {context}')
        for name in ['refractory','last']:
            np.testing.assert_array_equal(getattr(cpu,name),getattr(metal,name),
                err_msg=f'{name}: {context}')


@requires_metal
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


@requires_metal
@pytest.mark.parametrize('tau',[200.,137.])
@pytest.mark.parametrize('bins',[1,43,200])
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
        # One/43-bin controls are entirely strict. The 200-bin composite
        # accepts only specified g underflow from bin 44, retaining every
        # voltage/adaptation bit and all discrete fields throughout.
        _assert_bins_match(cpu,metal,bins,composite_underflow=bins==200)
    finally:
        cpu.backend.close();metal.backend.close()
