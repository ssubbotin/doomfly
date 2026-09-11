"""Build and inspect the target-local Objective-C++ and Metal artifacts."""
import argparse
import ctypes as C
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys


ROOT=Path(__file__).resolve().parents[2]
SOURCE=Path(__file__).resolve().parent
DEFAULT_OUTPUT=ROOT/'outputs/doom-learning/metal'
ABI_VERSION=1


class DeviceInfo(C.Structure):
    _fields_=[('abi_version',C.c_uint32),('supported',C.c_uint32),
        ('has_unified_memory',C.c_uint32),('supports_apple7',C.c_uint32),
        ('recommended_working_set',C.c_uint64),('registry_id',C.c_uint64),
        ('device_name',C.c_char*256)]


def _digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _capture(command):
    return subprocess.run(command,check=True,text=True,stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT).stdout.strip()


def library_path(output_dir=DEFAULT_OUTPUT):return Path(output_dir)/'libmemory-metal.dylib'


def build(output_dir=DEFAULT_OUTPUT):
    if sys.platform!='darwin' or platform.machine()!='arm64':
        raise RuntimeError('Metal backend requires arm64 macOS')
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    sources={name:_digest(SOURCE/name) for name in ['api.h','backend.mm','kernels.metal']}
    air=output/'kernels.air';metallib=output/'kernels.metallib';library=library_path(output)
    metadata=output/'build.json'
    if metadata.exists() and air.exists() and metallib.exists() and library.exists():
        record=json.loads(metadata.read_text())
        binaries={'air':_digest(air),'metallib':_digest(metallib),'library':_digest(library)}
        if record.get('sources')==sources and record.get('binaries')==binaries:return record
    partial_air=air.with_suffix('.air.partial')
    partial_metallib=metallib.with_suffix('.metallib.partial')
    partial_library=library.with_suffix('.dylib.partial')
    metal_command=['xcrun','-sdk','macosx','metal','-std=macos-metal2.4','-c',
        str(SOURCE/'kernels.metal'),'-o',str(partial_air)]
    metallib_command=['xcrun','-sdk','macosx','metallib',str(partial_air),'-o',str(partial_metallib)]
    library_command=['clang++','-O3','-std=c++17','-dynamiclib','-fobjc-arc','-arch','arm64',
        '-mmacosx-version-min=13.0','-I',str(SOURCE),str(SOURCE/'backend.mm'),
        '-framework','Foundation','-framework','Metal','-o',str(partial_library)]
    for command in [metal_command,metallib_command,library_command]:subprocess.run(command,check=True)
    partial_air.replace(air);partial_metallib.replace(metallib);partial_library.replace(library)
    binaries={'air':_digest(air),'metallib':_digest(metallib),'library':_digest(library)}
    record={'schema':1,'abi_version':ABI_VERSION,'sources':sources,'binaries':binaries,
        'commands':[metal_command,metallib_command,library_command],
        'compiler':_capture(['clang++','--version']).splitlines()[0],
        'sdk':_capture(['xcrun','-sdk','macosx','--show-sdk-version']),
        'macos':_capture(['sw_vers','-productVersion']),'architecture':platform.machine(),
        'metal_language':'macos-metal2.4'}
    temporary=metadata.with_suffix('.json.partial')
    temporary.write_text(json.dumps(record,indent=2)+'\n');temporary.replace(metadata)
    return record


def probe(output_dir=DEFAULT_OUTPUT):
    record=build(output_dir);library=C.CDLL(str(library_path(output_dir)))
    function=library.df_metal_probe;function.argtypes=[C.POINTER(DeviceInfo)];function.restype=C.c_int
    error=library.df_metal_last_error;error.restype=C.c_char_p
    info=DeviceInfo();status=function(C.byref(info))
    if status:raise RuntimeError(error().decode())
    return {**record,'device':{'name':info.device_name.decode(),'registry_id':info.registry_id,
        'recommended_working_set':info.recommended_working_set,
        'has_unified_memory':bool(info.has_unified_memory),
        'supports_apple7':bool(info.supports_apple7),'supported':bool(info.supported)}}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--probe',action='store_true');args=parser.parse_args()
    result=probe(args.output) if args.probe else build(args.output)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
