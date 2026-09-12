"""Numerical fixtures for resident coefficients, separate from full-graph gates."""
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest


SOURCE=Path(__file__).resolve().parents[1]/'doom_learning_v6/metal'
INTERVALS=[1,2,18,22,100,286,1023,1024]


@pytest.mark.parametrize('tau',[200.,137.])
def test_host_tables_match_every_canonical_float_coefficient(tmp_path,tau):
    compiler=shutil.which('clang++') or shutil.which('c++')
    assert compiler is not None,'A C++17 compiler is required for the native coefficient test'
    probe=tmp_path/'probe.cpp'
    probe.write_text('''#include "decay_tables.h"
#include <cstring>
#include <string>
int main(int argc,char **argv) {
  float dt=0.1f,tau=std::stof(argv[1]);
  auto table=doomfly::metal::make_decay_tables(dt,tau);
  for(int i=0;i<1024;i++) {
    float expected[3]={std::exp(-dt*i/20.f),std::exp(-dt*i/5.f),std::exp(-dt*i/tau)};
    for(int region=0;region<3;region++)
      if(std::memcmp(&table[region*1024+i],&expected[region],sizeof(float)))return 1;
  }
}
''')
    executable=tmp_path/'probe'
    compiled=subprocess.run([compiler,'-O3','-std=c++17','-I',str(SOURCE),str(probe),
        '-o',str(executable)],text=True,capture_output=True)
    assert compiled.returncode==0,compiled.stderr
    subprocess.run([str(executable),str(tau)],check=True)


def test_validation_identity_includes_host_coefficients():
    import hashlib
    from doom_learning_v6.metal.validate import source_identity
    identity=source_identity()
    name='doom_learning_v6/metal/decay_tables.h'
    assert name in identity
    assert identity[name]==hashlib.sha256((SOURCE/'decay_tables.h').read_bytes()).hexdigest()


def _edgeless_pair(tmp_path,tau):
    from doom_learning_v6.brain import MemoryBrain
    n=4;path=tmp_path/'decay-graph.npz'
    np.savez(path,ptr=np.zeros(n+1,dtype=np.int64),post=np.empty(0,dtype=np.int32),
        weight=np.empty(0,dtype=np.float32),ids=np.arange(n,dtype=np.int64),
        retina=np.empty(0,dtype=np.int32),uv=np.empty((0,2),dtype=np.float32),
        lamina=np.empty(0,dtype=np.int32),sugar=np.empty(0,dtype=np.int32),
        superclass=np.array(['test']*n))
    circuit={'edges':np.empty(0,dtype=np.int64),'pre':np.empty(0,dtype=np.int32),
        'kc_mask':np.zeros(n,dtype=np.uint8),'dan_index':np.full(n,-1,dtype=np.int8),
        'gain':np.empty((0,0),dtype=np.float32),'kc':np.empty(0,dtype=np.int32),
        'mb':np.empty(0,dtype=np.int32),'dan':np.empty(0,dtype=np.int32)}
    kwargs={'eta':0.,'circuit':circuit,'modulation_mask':np.zeros(n,dtype=np.uint8),
        'adaptation_tau':tau}
    return [MemoryBrain(path,backend=backend,**kwargs) for backend in ['cpu','metal']]


@pytest.mark.skipif(sys.platform!='darwin',reason='Metal requires macOS')
@pytest.mark.parametrize('tau',[200.,137.])
@pytest.mark.parametrize('interval',INTERVALS)
def test_sleeping_state_decay_and_refractory_boundary(tmp_path,tau,interval):
    cpu,metal=_edgeless_pair(tmp_path,tau)
    try:
        # Artificial sleeping histories isolate voltage, conductance, adaptation
        # and frozen refractory adaptation without a physiological claim. Seed
        # before initial upload; one valid public tick materializes elapsed
        # history at now=0 while preserving the native 100-tick bin limit.
        for brain in [cpu,metal]:
            brain.v[:]=[-51.,-52.,-52.,-52.]
            brain.g[:]=[0.,1.,0.,0.]
            brain.adaptation[:]=[0.,0.,1.,1.]
            brain.refractory[:]=[0,0,0,interval+1]
            brain.active_flag.fill(0);brain.nactive.fill(0);brain.last.fill(-interval)
            brain.backend.advance(1)
            brain.backend.materialize('decay-fixture')
            assert not brain.counts.any()
            assert brain.cursor==1
            np.testing.assert_array_equal(brain.last,np.zeros(4,dtype=np.int64))
            np.testing.assert_array_equal(brain.refractory,[0,0,0,1])
            assert brain.v[3]==-52. and brain.g[3]==0.
            for name in ['v','g','adaptation']:
                assert np.isfinite(getattr(brain,name)).all(),name
        # Composite voltage arithmetic retains the established micrograph tolerance.
        np.testing.assert_allclose(metal.v,cpu.v,rtol=1e-5,atol=.002)
        if interval<1024:
            # Multiplication by one isolates the actual canonical native coefficient.
            np.testing.assert_array_equal(metal.g,cpu.g)
            np.testing.assert_array_equal(metal.adaptation,cpu.adaptation)
        else:
            # The preserved GPU exp fallback has no bitwise host-equality guarantee.
            # Compare its complete formula against the actual canonical CPU fallback.
            np.testing.assert_allclose(metal.g,cpu.g,rtol=1e-5,atol=0.)
            np.testing.assert_allclose(metal.adaptation,cpu.adaptation,rtol=1e-5,atol=0.)
    finally:
        cpu.backend.close();metal.backend.close()
