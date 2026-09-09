"""Sequential paired N3 runs; preserve all samples and exact output checks."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import sysconfig


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--donor',type=Path,required=True)
    p.add_argument('--file',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--profile',choices=('magnetic','thermal','bounded','file-curl'),required=True)
    p.add_argument('--image-size',type=int,default=500)
    p.add_argument('--repeats',type=int,default=4)
    p.add_argument('--workers',type=int,default=4)
    a=p.parse_args()
    a.output_dir.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1]
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
    records=[]
    for repeat in range(a.repeats):
        for flavor in (('donor','new') if repeat%2==0 else ('new','donor')):
            output=a.output_dir/f'{flavor}-{repeat}.json'
            command=[sys.executable,'-I','-S',str(root/'scripts/compare_n3.py'),
                '--source-root',str(a.donor/'src' if flavor=='donor' else root/'src'),
                '--dependencies',sysconfig.get_path('purelib'),'--flavor',flavor,'--file',str(a.file.resolve()),
                '--output',str(output),'--profile',a.profile,'--image-size',str(a.image_size),'--workers',str(a.workers)]
            subprocess.run(command,env=env,check=True)
            record=json.loads(output.read_text());record['repeat']=repeat;records.append(record)
            if record['hashes']!=records[0]['hashes']:
                different=[key for key,value in record['hashes'].items() if value!=records[0]['hashes'].get(key)]
                raise AssertionError((flavor,repeat,different))
    medians={}
    for flavor in ('donor','new'):
        group=[r for r in records if r['flavor']==flavor and (a.repeats==1 or r['repeat']>0)]
        medians[flavor]={key:statistics.median(r[key] for r in group) for key in
                         ('import_seconds','file_to_result_seconds','startup_to_result_seconds')}
    (a.output_dir/'summary.json').write_text(json.dumps({'runs':records,'medians':medians},indent=2)+'\n')
    print(json.dumps(medians,indent=2),flush=True)


if __name__=='__main__':main()
