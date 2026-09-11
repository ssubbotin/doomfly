"""Reproducible native build for the portable CPU batch executor."""
import argparse
import ctypes as C
import fcntl
import hashlib
import json
import os
import platform
from pathlib import Path
import shutil
import subprocess
import sys

from doom_learning.common import ROOT


ABI_VERSION=1
SOURCE=Path(__file__).parent
DEFAULT_OUTPUT=ROOT/'outputs/doom-learning/cpu-batch'


def _digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def library_path(output_dir=DEFAULT_OUTPUT):
    return Path(output_dir)/('libcpu-batch.dylib' if sys.platform=='darwin' else 'libcpu-batch.so')


def _compiler():
    requested=os.environ.get('CXX')
    candidates=[requested] if requested else []
    candidates+=['clang++','c++'] if sys.platform=='darwin' else ['clang++','g++','c++']
    for candidate in candidates:
        if candidate and shutil.which(candidate):return candidate
    raise RuntimeError('A C++17 compiler is required')


def build(output_dir=DEFAULT_OUTPUT):
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    sources={name:_digest(SOURCE/name) for name in ['api.h','executor.cpp']}
    library=library_path(output);metadata=output/'build.json'
    with (output/'.build.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if metadata.exists() and library.exists():
            record=json.loads(metadata.read_text())
            if record.get('sources')==sources and \
                    record.get('binary_sha256')==_digest(library):
                return record
        compiler=_compiler()
        flags=['-O3','-std=c++17','-pthread']
        flags+=['-dynamiclib'] if sys.platform=='darwin' else ['-shared','-fPIC']
        temporary=library.with_suffix(library.suffix+'.partial')
        command=[compiler,*flags,'-I',str(SOURCE),str(SOURCE/'executor.cpp'),
            '-o',str(temporary)]
        subprocess.run(command,check=True)
        temporary.replace(library)
        version=subprocess.run([compiler,'--version'],check=True,text=True,
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT).stdout.splitlines()[0]
        record={'schema':1,'abi_version':ABI_VERSION,'sources':sources,
            'binary_sha256':_digest(library),'compiler':version,'flags':flags,
            'platform':sys.platform,'architecture':platform.machine()}
        metadata_temporary=metadata.with_suffix('.json.partial')
        metadata_temporary.write_text(json.dumps(record,indent=2)+'\n')
        metadata_temporary.replace(metadata)
        return record


def probe(output_dir=DEFAULT_OUTPUT):
    record=build(output_dir);library=C.CDLL(str(library_path(output_dir)))
    function=library.df_cpu_batch_abi_version
    function.argtypes=[];function.restype=C.c_uint32
    native=int(function())
    if native!=ABI_VERSION:raise RuntimeError('Native CPU batch ABI version mismatch')
    return {**record,'native_abi_version':native}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--probe',action='store_true')
    args=parser.parse_args()
    result=probe(args.output) if args.probe else build(args.output)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
