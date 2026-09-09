"""New ordinary-prefix profile; DAT-003's original rejection remains unchanged."""

import os
import numpy as np
import pytest

from simesh_rewrite.amrvac_dat_reader import make_amrvac_v5_block_reader,make_amrvac_v5_ordinary_block_reader
from simesh_rewrite.blockio import read_blocks_into
from test_dat_003 import synthetic_source,i3,_new_destination,_expected_transfer


@pytest.mark.parametrize("byte_order",["<",">"])
def test_staggered_prefix_saved_ghosts_and_exact_selectors(tmp_path,byte_order):
    block=(3,2,4)
    ghosts=(((1,0,2),(0,2,1)),((0,0,0),(0,0,0)),((2,1,0),(1,0,2)))
    tails={i:np.full(3*int(np.prod(np.array(block)+lo+np.array(hi)+1)),
                    0x0123456789ABCDEF,dtype=f"{byte_order}u8").tobytes()
           for i,(lo,hi) in enumerate(ghosts)}
    with synthetic_source(tmp_path,byte_order=byte_order,block_shape=block,
                          ghosts=ghosts,padding_after=tails) as source:
        index=source.index._replace(staggered=True)
        with pytest.raises(ValueError,match="staggered"):
            make_amrvac_v5_block_reader(source.file_descriptor,index,source.binding)
        reader=make_amrvac_v5_ordinary_block_reader(source.file_descriptor,index,source.binding)
        ids=np.array([2,0,2,1],dtype=np.int64)
        fields=np.array([3,0,3,1],dtype=np.int64)
        output=_new_destination((4,4,5,4,6))
        expected=_expected_transfer(source,ids,fields,i3(1,0,1),i3(3,2,4),output,i3(1,1,2))
        os.lseek(source.file_descriptor,17,os.SEEK_SET)
        read_blocks_into(reader,i3(1,0,1),i3(3,2,4),ids,fields,output,i3(1,1,2))
        np.testing.assert_array_equal(output.view(np.uint64),expected)
        assert os.lseek(source.file_descriptor,0,os.SEEK_CUR)==17


def test_incomplete_tail_preflight_preserves_all_output(tmp_path):
    block=(4,4,4)
    ghosts=(((0,0,0),(0,0,0)),)*3
    size=3*5**3*8
    with synthetic_source(tmp_path,block_shape=block,ghosts=ghosts,
                          padding_after={0:bytes(size),1:bytes(size),2:bytes(size-8)}) as source:
        reader=make_amrvac_v5_ordinary_block_reader(source.file_descriptor,
            source.index._replace(staggered=True),source.binding)
        output=_new_destination((2,2,4,4,4))
        expected=output.view(np.uint64).copy()
        with pytest.raises(ValueError,match="record end"):
            read_blocks_into(reader,i3(0,0,0),i3(4,4,4),np.array([0,2],dtype=np.int64),
                             np.array([0,1],dtype=np.int64),output,i3(0,0,0))
        np.testing.assert_array_equal(output.view(np.uint64),expected)
