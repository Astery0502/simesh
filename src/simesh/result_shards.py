"""Incremental, verified delivery of identified line and profile result shards."""

from dataclasses import dataclass
import hashlib
import json
import os
import re
from pathlib import Path
import tempfile

import numpy as np

from ._validation import frozen_array
from .geometry import LineSet
from .line_profiles import LineProfiles
from .results_io import save_result, load_result, ResultFileError, _json_object, _unique_object, _keys, _invalid_constant


def _digest(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path, data):
    fd, name = tempfile.mkstemp(prefix='.index-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(data, stream, allow_nan=False)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _manifest(path, data):
    _write_json(path/'manifest.json', data)


def _seed_ids(result):
    if isinstance(result, LineProfiles):
        return result.lines.seeds.ids
    if isinstance(result, LineSet):
        return result.seeds.ids
    raise TypeError('shards must contain LineSet or LineProfiles results')


def save_result_shards(path, batches, *, seed_ids, metadata=None, source=None):
    """Consume ordered seed batches, publishing each independently and atomically.

    Parameters
    ----------
    path : str or Path
        New shard directory, which must not exist; its parent must exist.
    batches : iterable of LineSet or LineProfiles
        Nonempty batches of one result type, in exact requested ID order.
    seed_ids : ndarray
        Unique int64 requested IDs in delivery order.
    metadata : dict, optional
        Caller-supplied physical/numerical choices using finite JSON values.
    source : dict, optional
        Caller-provided unverified source description.

    Returns
    -------
    ResultShards
        Completed version-2 delivery. ResultShards.complete concerns seed delivery, not
        scientific termination.

    Notes
    -----
    An interrupted stream leaves committed shards inspectable; it does not provide integration checkpoint/resume.
    """
    ids = np.asarray(seed_ids)
    if ids.ndim != 1 or ids.dtype != np.int64 or len(np.unique(ids)) != len(ids):
        raise ValueError('seed_ids must be unique int64 IDs in delivery order')
    ids = frozen_array(ids,np.int64)
    metadata = _json_object({} if metadata is None else metadata, 'metadata')
    source = None if source is None else _json_object(source, 'source')
    path = Path(path)
    path.mkdir()
    np.save(path/'seed_ids.npy',ids,allow_pickle=False)
    manifest = dict(format='simesh-result-shards', schema_version=2, complete=False,
                    seed_ids=dict(file='seed_ids.npy',count=len(ids),sha256=_digest(path/'seed_ids.npy')),
                    shards=[], metadata=metadata, source=source)
    _manifest(path, manifest)
    first = 0
    kind = None
    for batch in batches:
        batch_ids = _seed_ids(batch)
        stop = first+len(batch_ids)
        if not len(batch_ids) or not np.array_equal(batch_ids, ids[first:stop]):
            raise ValueError('shard seed IDs must exactly match the next requested range')
        if kind is not None and type(batch).__name__ != kind:
            raise ValueError('all shards must use the same result type')
        kind = type(batch).__name__
        name = f'shard-{len(manifest["shards"]):06d}.npz'
        save_result(path/name, batch, metadata={'seed_range':[first,stop]}, source=source)
        entry = dict(file=name, start=first, stop=stop,sha256=_digest(path/name),kind=kind)
        _write_json(path/f'index-{len(manifest["shards"]):06d}.json',entry)
        manifest['shards'].append(entry)
        first = stop
        del batch
    if first != len(ids):
        raise ValueError(f'incomplete shard stream: delivered {first} of {len(ids)} seeds')
    manifest['complete'] = True
    _manifest(path, manifest)
    return ResultShards(path,manifest,ids)


@dataclass(frozen=True)
class ResultShards:
    """Incremental delivery index; load one verified shard at a time.

    Attributes
    ----------
    path : Path
        Shard directory.
    manifest : dict
        Version-specific metadata; prefer stable seed_ids/complete accessors.
    """
    path: Path
    manifest: dict
    _ids: np.ndarray | None = None

    @property
    def seed_ids(self):
        """Requested delivery IDs, independent of the manifest schema version."""
        if self._ids is not None:
            return self._ids
        return frozen_array(self.manifest['seed_ids'],np.int64)

    @property
    def complete(self):
        """Whether every requested seed was delivered, independent of scientific termination.
        """
        return self.manifest['complete']

    def __len__(self):
        """Number of committed shards, not number of requested seeds."""
        return len(self.manifest['shards'])

    def load(self, index):
        """Verify and load one shard; never load or join the other payloads."""
        entry = self.manifest['shards'][index]
        path = self.path/entry['file']
        if _digest(path) != entry['sha256']:
            raise ResultFileError('shard checksum does not match the manifest')
        result = load_result(path)
        expected = self.seed_ids[entry['start']:entry['stop']]
        if (type(result.result).__name__ != entry['kind'] or
                not np.array_equal(_seed_ids(result.result), expected) or
                result.metadata.get('seed_range') != [entry['start'],entry['stop']]):
            raise ResultFileError('shard identity or range disagrees with the manifest')
        return result


def _read_json(path):
    return json.loads(path.read_text(),object_pairs_hook=_unique_object,
                      parse_constant=_invalid_constant)


def _valid_digest(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}',value) is not None


def open_result_shards(path):
    """Read v1/v2 deliveries, recovering committed indices of incomplete v2 runs.

    Parameters
    ----------
    path : str or Path
        Existing shard directory.

    Returns
    -------
    ResultShards
        Version-1/2 delivery metadata and on-demand verified loading; orphaned or
        temporary payloads are excluded from committed shards.
    """
    path = Path(path)
    try:
        data = _read_json(path/'manifest.json')
        _keys(data,('format','schema_version','complete','seed_ids','shards','metadata','source'),'shard manifest')
        _json_object(data['metadata'],'metadata')
        if data['source'] is not None:
            _json_object(data['source'],'source')
        if (data['format'] != 'simesh-result-shards' or type(data['schema_version']) is not int or
                data['schema_version'] not in (1,2) or type(data['complete']) is not bool or
                type(data['shards']) is not list):
            raise ValueError('invalid shard manifest')
        if data['schema_version'] == 1:
            values = data['seed_ids']
            if (type(values) is not list or
                    any(type(i) is not int or not -(2**63)<=i<2**63 for i in values)):
                raise ValueError('invalid seed IDs')
            ids = frozen_array(values,np.int64)
        else:
            descriptor = data['seed_ids']
            _keys(descriptor,('file','count','sha256'),'seed IDs descriptor')
            if (descriptor['file'] != 'seed_ids.npy' or type(descriptor['count']) is not int or
                    descriptor['count'] < 0 or not _valid_digest(descriptor['sha256'])):
                raise ValueError('invalid seed IDs descriptor')
            seed_path = path/descriptor['file']
            if _digest(seed_path) != descriptor['sha256']:
                raise ValueError('seed IDs checksum does not match the manifest')
            ids = np.load(seed_path,allow_pickle=False,mmap_mode='r')
            if not isinstance(ids,np.ndarray) or ids.shape != (descriptor['count'],) or ids.dtype != np.int64:
                raise ValueError('invalid seed IDs array')
            if not data['complete']:
                if data['shards']:
                    raise ValueError('incomplete v2 manifest must use committed indices')
                entries = []
                index_paths = list(path.glob('index-*.json'))
                if any(re.fullmatch(r'index-[0-9]{6,}\.json',p.name) is None for p in index_paths):
                    raise ValueError('invalid shard index name')
                index_paths.sort(key=lambda p: int(p.stem[6:]))
                for index, index_path in enumerate(index_paths):
                    if index_path.name != f'index-{index:06d}.json':
                        raise ValueError('missing or invalid shard index')
                    entries.append(_read_json(index_path))
                data['shards'] = entries
        if len(np.unique(ids)) != len(ids):
            raise ValueError('invalid seed IDs')
        first = 0
        kind = None
        for index, entry in enumerate(data['shards']):
            _keys(entry,('file','start','stop','sha256','kind'),'shard entry')
            if (entry['file'] != f'shard-{index:06d}.npz' or type(entry['start']) is not int or entry['start'] != first or
                    type(entry['stop']) is not int or not first<entry['stop']<=len(ids) or
                    entry['kind'] not in ('LineSet','LineProfiles') or
                    (kind is not None and kind != entry['kind']) or not _valid_digest(entry['sha256'])):
                raise ValueError('invalid shard range, type or file')
            first = entry['stop']
            kind = entry['kind']
        if data['complete'] and first != len(ids):
            raise ValueError('complete manifest has missing ranges')
        return ResultShards(path, data, ids)
    except (OSError, EOFError, KeyError, TypeError, ValueError) as error:
        raise ResultFileError(f'invalid shard manifest: {error}') from error
