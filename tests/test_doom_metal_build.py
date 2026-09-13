import json
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.fixture
def metal_toolchain(monkeypatch):
    """Replace the unavailable Apple tools while retaining real cache files."""
    import doom_learning_v6.metal.build as module
    monkeypatch.setattr(module.sys,'platform','darwin')
    monkeypatch.setattr(module.platform,'machine',lambda:'arm64')
    compiled=[]
    def run(command,**kwargs):
        if '--version' in command:output='Apple clang test version\n'
        elif '--show-sdk-version' in command:output='26.0\n'
        elif command[0]=='sw_vers':output='26.6.2\n'
        else:
            compiled.append(command.copy())
            target=Path(command[command.index('-o')+1])
            target.write_bytes(json.dumps(command).encode())
            output=''
        return subprocess.CompletedProcess(command,0,stdout=output)
    monkeypatch.setattr(module.subprocess,'run',run)
    return module,compiled


def test_build_emits_strict_metal_arithmetic(metal_toolchain,tmp_path):
    module,compiled=metal_toolchain
    record=module.build(tmp_path)
    command=compiled[0]
    assert '-fno-fast-math' in command
    assert '-ffp-contract=off' in command
    assert '-ffast-math' not in command
    assert record['build_configuration']['metal_compile_flags']==[
        '-std=macos-metal2.4','-fno-fast-math','-ffp-contract=off']


def test_matching_build_reuses_verified_artifacts(metal_toolchain,tmp_path):
    module,compiled=metal_toolchain
    first=module.build(tmp_path)
    second=module.build(tmp_path)
    assert second==first
    assert len(compiled)==3


def test_build_identity_includes_host_coefficients(metal_toolchain,tmp_path):
    module,_=metal_toolchain
    record=module.build(tmp_path)
    assert 'decay_tables.h' in record['sources']
    assert record['sources']['decay_tables.h']==hashlib.sha256(
        (module.SOURCE/'decay_tables.h').read_bytes()).hexdigest()


def test_changed_host_coefficients_cannot_reuse_cache(metal_toolchain,tmp_path,monkeypatch):
    module,compiled=metal_toolchain
    source=tmp_path/'source';source.mkdir()
    for name in ['api.h','backend.mm','kernels.metal','decay_tables.h','build.py']:
        shutil.copyfile(module.SOURCE/name,source/name)
    monkeypatch.setattr(module,'SOURCE',source)
    monkeypatch.setattr(module,'__file__',str(source/'build.py'))
    output=tmp_path/'output';module.build(output)
    header=source/'decay_tables.h'
    header.write_bytes(header.read_bytes()+b'\n// Cache invalidation fixture.\n')
    changed=module.build(output)
    assert len(compiled)==6
    assert changed['sources']['decay_tables.h']==hashlib.sha256(header.read_bytes()).hexdigest()


@pytest.mark.parametrize('legacy_configuration',[None,{
    'metal_compile_flags':['-std=macos-metal2.4','-ffast-math']}])
def test_old_math_configuration_cannot_reuse_cache(metal_toolchain,tmp_path,legacy_configuration):
    module,compiled=metal_toolchain
    module.build(tmp_path)
    path=tmp_path/'build.json';record=json.loads(path.read_text())
    if legacy_configuration is None:record.pop('build_configuration',None)
    else:record['build_configuration']=legacy_configuration
    path.write_text(json.dumps(record))
    rebuilt=module.build(tmp_path)
    assert len(compiled)==6
    assert rebuilt['build_configuration']['metal_compile_flags']==[
        '-std=macos-metal2.4','-fno-fast-math','-ffp-contract=off']


@pytest.mark.parametrize('field,value',[
    ('abi_version',3),('abi_version',5),('abi_version',99),('builder_sha256','0'*64),('library_compile_flags',['-O0'])])
def test_changed_build_identity_cannot_reuse_cache(metal_toolchain,tmp_path,field,value):
    module,compiled=metal_toolchain
    original=module.build(tmp_path)
    path=tmp_path/'build.json';record=json.loads(path.read_text())
    record['build_configuration'][field]=value
    path.write_text(json.dumps(record))
    rebuilt=module.build(tmp_path)
    assert len(compiled)==6
    assert rebuilt==original


@pytest.mark.parametrize('changed_artifact',['kernels.air','kernels.metallib','libmemory-metal.dylib'])
def test_changed_binary_cannot_reuse_cache(metal_toolchain,tmp_path,changed_artifact):
    module,compiled=metal_toolchain
    module.build(tmp_path)
    (tmp_path/changed_artifact).write_bytes(b'changed binary')
    module.build(tmp_path)
    assert len(compiled)==6


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
