import subprocess
import sys

import pytest


def test_build_rejects_non_macos(tmp_path):
    from doom_learning_v6.metal.build import build
    if sys.platform=='darwin':pytest.skip('Non-macOS guard is exercised on Linux')
    with pytest.raises(RuntimeError,match='arm64 macOS'):
        build(tmp_path)


def test_generated_metal_artifacts_are_ignored():
    paths=['outputs/doom-learning/metal/libmemory-metal.dylib',
        'outputs/doom-learning/metal/kernels.air',
        'outputs/doom-learning/metal/kernels.metallib',
        'outputs/doom-learning/physiology-v6/libmemory.so.json']
    for path in paths:
        result=subprocess.run(['git','check-ignore','-q',path],check=False)
        assert result.returncode==0,path


def test_probe_confirms_native_abi_version():
    if sys.platform!='darwin':pytest.skip('Metal probe requires macOS')
    from doom_learning_v6.metal.build import ABI_VERSION,probe
    assert probe()['native_abi_version']==ABI_VERSION
