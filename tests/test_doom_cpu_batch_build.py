import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor


def test_build_records_source_and_binary_identity(tmp_path):
    from doom_learning_v6.cpu_batch.build import ABI_VERSION,build,library_path
    first=build(tmp_path);second=build(tmp_path)
    assert first==second
    assert first['abi_version']==ABI_VERSION==1
    assert set(first['sources'])=={'api.h','executor.cpp'}
    assert len(first['binary_sha256'])==64
    assert library_path(tmp_path).exists()
    assert json.loads((tmp_path/'build.json').read_text())==first


def test_probe_confirms_native_abi(tmp_path):
    from doom_learning_v6.cpu_batch.build import ABI_VERSION,probe
    assert probe(tmp_path)['native_abi_version']==ABI_VERSION


def test_generated_cpu_batch_artifacts_are_ignored():
    paths=['outputs/doom-learning/cpu-batch/build.json',
        'outputs/doom-learning/cpu-batch/libcpu-batch.so',
        'outputs/doom-learning/cpu-batch/libcpu-batch.dylib']
    for path in paths:
        assert subprocess.run(['git','check-ignore','-q',path],check=False).returncode==0,path


def test_build_module_probe_cli_prints_native_identity_without_import_warning(tmp_path):
    result=subprocess.run([sys.executable,'-m','doom_learning_v6.cpu_batch.build',
        '--probe','--output',str(tmp_path)],check=True,text=True,
        stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    assert json.loads(result.stdout)['native_abi_version']==1
    assert 'RuntimeWarning' not in result.stderr


def test_concurrent_builds_share_one_complete_artifact(tmp_path):
    from doom_learning_v6.cpu_batch.build import build
    with ThreadPoolExecutor(max_workers=8) as pool:
        calls=[pool.submit(build,tmp_path) for _ in range(8)]
        records=[call.result() for call in calls]
    assert all(record==records[0] for record in records[1:])
    assert not list(tmp_path.glob('*.partial'))
