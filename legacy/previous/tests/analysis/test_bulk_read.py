"""Bulk record decoding preserves format semantics and exact payload bits."""

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import os
import tempfile
import unittest
import numpy as np

from simesh.amrvac.analysis_read import read_full_interiors
from simesh_rewrite.amrvac_dat_reader import _AMRVACV5BlockReaderState, _current_identity


@contextmanager
def records(path, byte_order, ghosts, staggered=False, truncate=0):
    block = (4,6,8)
    special = np.array([0,0x8000000000000000,0x3ff0000000000000,
                        0x7ff8000000000123,0xfff0000000000000,1],dtype=np.uint64)
    offsets, expected = [], []
    with path.open('wb') as stream:
        stream.write(bytes(64))
        for row, ghost in enumerate(ghosts):
            offsets.append(stream.tell())
            lower, upper = ghost[:3], ghost[3:]
            shape = tuple(n+a+b for n,a,b in zip(block,lower,upper))
            cells = int(np.prod(shape))
            bits = (np.arange(3*cells,dtype=np.float64)+row*1.e6).reshape((3,*shape)).view(np.uint64)
            for field in range(3):
                selected = bits[field].reshape(-1)[::17]
                selected[:] = np.resize(np.roll(special,row+field),len(selected))
            slices = tuple(slice(a,a+n) for a,n in zip(lower,block))
            expected.append(bits[(slice(None),*slices)].copy())
            stream.write(np.asarray(ghost,dtype=byte_order+'i4').tobytes())
            stream.write(bits.transpose(0,3,2,1).astype(byte_order+'u8').tobytes())
            if staggered:
                stream.write(bytes(24*int(np.prod(np.array(shape)+1))))
        if truncate:
            stream.truncate(stream.tell()-truncate)
    fd = os.open(path,os.O_RDONLY)
    try:
        state = _AMRVACV5BlockReaderState(fd,byte_order,_current_identity(fd),
                                        (len(ghosts),3,*block),np.array(offsets,dtype=np.int64),staggered)
        yield SimpleNamespace(state=state), np.array(expected)
    finally:
        os.close(fd)


class BulkReadTests(unittest.TestCase):
    def test_uniform_and_variable_records_endian_fields_and_bits(self):
        zero = (0,0,0,0,0,0)
        saved = (1,0,2,0,2,1)
        shifted = (0,2,1,1,0,2)
        with tempfile.TemporaryDirectory() as directory:
            for byte_order in ('<','>'):
                for ghosts in ((zero,)*3,(saved,)*3,(saved,zero,shifted),(saved,shifted,saved)):
                    for staggered in (False,True):
                        with records(Path(directory)/'records.dat',byte_order,ghosts,staggered) as (reader,expected):
                            for fields in ([0,1,2],[2,0],[1,2],[2]):
                                for batch in (100,1024**2):
                                    result,stats = read_full_interiors(reader,np.array(fields),chunk_bytes=batch)
                                    np.testing.assert_array_equal(result.view(np.uint64),expected[:,fields])
                                    self.assertEqual(stats['selected_value_bytes'],result.nbytes)

    def test_truncated_tail_and_changed_file_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'records.dat'
            with records(path,'<',[(0,0,0,0,0,0)]*2,staggered=True,truncate=8) as (reader,_):
                with self.assertRaises(ValueError):
                    read_full_interiors(reader,[0,1])
            with records(path,'<',[(0,0,0,0,0,0)]*2) as (reader,_):
                with path.open('ab') as stream:
                    stream.write(b'x')
                with self.assertRaises(OSError):
                    read_full_interiors(reader,[0])
            with records(path,'<',[(0,0,0,0,0,0)]*2) as (reader,_):
                with path.open('r+b') as stream:
                    stream.seek(64)
                    stream.write(np.array([-1],dtype='<i4').tobytes())
                reader.state = replace(reader.state,file_identity=_current_identity(reader.state.file_descriptor))
                with self.assertRaises(ValueError):
                    read_full_interiors(reader,[0])


if __name__ == '__main__':
    unittest.main()
