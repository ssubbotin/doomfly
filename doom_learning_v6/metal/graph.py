"""Build a complete target-indexed view of the retained outgoing graph."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class IncomingGraph:
    ptr:np.ndarray
    pre:np.ndarray
    edge:np.ndarray
    metadata:dict


def _digest(array):return hashlib.sha256(array.tobytes()).hexdigest()


def _validate_outgoing(ptr,post):
    if ptr.dtype!=np.int64 or ptr.ndim!=1 or not ptr.flags.c_contiguous or len(ptr)<2:
        raise ValueError('Invalid outgoing pointer array')
    if post.dtype!=np.int32 or post.ndim!=1 or not post.flags.c_contiguous:
        raise ValueError('Invalid outgoing post array')
    n=len(ptr)-1
    if ptr[0]!=0 or ptr[-1]!=len(post) or np.any(np.diff(ptr)<0):
        raise ValueError('Invalid outgoing CSR')
    if np.any(post<0) or np.any(post>=n):raise ValueError('Outgoing post out of bounds')
    if len(post)>np.iinfo(np.int32).max:raise ValueError('Metal edge index exceeds int32')


def _metadata(ptr,post,incoming_ptr,pre,edge):
    return {'schema':1,'neurons':len(ptr)-1,'edges':len(post),
        'out_ptr_sha256':_digest(ptr),'out_post_sha256':_digest(post),
        'incoming_ptr_sha256':_digest(incoming_ptr),'incoming_pre_sha256':_digest(pre),
        'incoming_edge_sha256':_digest(edge),
        'builder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def _validate_incoming(out_ptr,out_post,incoming):
    n=len(out_ptr)-1;e=len(out_post);ptr,pre,edge=incoming.ptr,incoming.pre,incoming.edge
    if ptr.dtype!=np.int64 or ptr.shape!=(n+1,) or not ptr.flags.c_contiguous:
        raise ValueError('Invalid incoming pointer array')
    if pre.dtype!=np.int32 or pre.shape!=(e,) or not pre.flags.c_contiguous:
        raise ValueError('Invalid incoming source array')
    if edge.dtype!=np.int32 or edge.shape!=(e,) or not edge.flags.c_contiguous:
        raise ValueError('Invalid incoming edge array')
    if ptr[0]!=0 or ptr[-1]!=e or np.any(np.diff(ptr)<0):
        raise ValueError('Invalid incoming CSR')
    if not np.array_equal(np.sort(edge),np.arange(e,dtype=np.int32)):
        raise ValueError('Incoming edge IDs are not an exact permutation')
    source=np.repeat(np.arange(n,dtype=np.int32),np.diff(out_ptr))
    if not np.array_equal(pre,source[edge]):raise ValueError('Incoming source mismatch')
    for target in range(n):
        ids=edge[ptr[target]:ptr[target+1]]
        if np.any(out_post[ids]!=target):raise ValueError('Incoming target mismatch')
    expected=_metadata(out_ptr,out_post,ptr,pre,edge)
    if incoming.metadata!=expected:raise ValueError('Incoming graph metadata mismatch')


def load_or_build_incoming(out_ptr,out_post,cache_dir):
    """Load or deterministically derive a stable incoming edge permutation."""
    out_ptr=np.asarray(out_ptr);out_post=np.asarray(out_post)
    _validate_outgoing(out_ptr,out_post)
    path=Path(cache_dir)/'incoming.npz'
    if path.exists():
        with np.load(path,allow_pickle=False) as archive:
            incoming=IncomingGraph(archive['ptr'].copy(),archive['pre'].copy(),archive['edge'].copy(),
                json.loads(str(archive['metadata'])))
        _validate_incoming(out_ptr,out_post,incoming)
        return incoming
    n=len(out_ptr)-1
    source=np.repeat(np.arange(n,dtype=np.int32),np.diff(out_ptr))
    order=np.argsort(out_post,kind='stable').astype(np.int32,copy=False)
    counts=np.bincount(out_post,minlength=n)
    incoming_ptr=np.r_[0,np.cumsum(counts,dtype=np.int64)]
    pre=np.ascontiguousarray(source[order],dtype=np.int32)
    edge=np.ascontiguousarray(order,dtype=np.int32)
    incoming=IncomingGraph(incoming_ptr,pre,edge,{})
    incoming=IncomingGraph(incoming.ptr,incoming.pre,incoming.edge,
        _metadata(out_ptr,out_post,incoming.ptr,incoming.pre,incoming.edge))
    _validate_incoming(out_ptr,out_post,incoming)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.npz.partial')
    with temporary.open('wb') as stream:
        np.savez(stream,ptr=incoming.ptr,pre=incoming.pre,edge=incoming.edge,
            metadata=json.dumps(incoming.metadata,sort_keys=True))
    temporary.replace(path)
    return incoming
