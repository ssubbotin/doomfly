import json

import numpy as np
import pytest

from doom_learning_v6.metal.graph import load_or_build_incoming


PTR=np.array([0,3,4,6],dtype=np.int64)
POST=np.array([1,1,0,1,2,1],dtype=np.int32)


def test_incoming_csr_is_stable_complete_permutation(tmp_path):
    incoming=load_or_build_incoming(PTR,POST,tmp_path)
    np.testing.assert_array_equal(incoming.ptr,[0,1,5,6])
    np.testing.assert_array_equal(incoming.edge,[2,0,1,3,5,4])
    np.testing.assert_array_equal(incoming.pre,[0,0,0,1,2,2])
    np.testing.assert_array_equal(np.sort(incoming.edge),np.arange(6))


def test_valid_cached_index_reloads_identically(tmp_path):
    expected=load_or_build_incoming(PTR,POST,tmp_path)
    actual=load_or_build_incoming(PTR.copy(),POST.copy(),tmp_path)
    assert actual.metadata==expected.metadata
    for name in ['ptr','pre','edge']:
        np.testing.assert_array_equal(getattr(actual,name),getattr(expected,name))


def test_corrupt_cached_index_is_rejected(tmp_path):
    incoming=load_or_build_incoming(PTR,POST,tmp_path)
    edge=incoming.edge.copy();edge[0]=edge[1]
    np.savez(tmp_path/'incoming.npz',ptr=incoming.ptr,pre=incoming.pre,edge=edge,
        metadata=json.dumps(incoming.metadata))
    with pytest.raises(ValueError,match='permutation'):
        load_or_build_incoming(PTR,POST,tmp_path)


def test_out_of_range_post_is_rejected(tmp_path):
    with pytest.raises(ValueError,match='out of bounds'):
        load_or_build_incoming(PTR,np.array([1,1,3,1,2,1],dtype=np.int32),tmp_path)
