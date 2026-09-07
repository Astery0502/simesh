"""Sequential runtime campaigns; never benchmark while building or testing."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path('benchmark-results/analysis-core')
CAMPAIGNS={
    'f':[
        ('f-base','--workload f --pool 256'),
        ('f-spatial','--workload f --pool 256 --order spatial'),
        ('f-batch','--workload f --pool 256 --batch 256'),
        ('f-support','--workload f --pool 128 --support 256 --batch 128'),
        ('f-cache','--workload f --pool 256 --value-cache 1024'),
        ('f-retain','--workload f --pool 600'),
    ],
    'l':[
        ('l-base','--workload l --pool 512 --support 256'),
        ('l-tile4','--workload l --pool 512 --support 256 --tile 4'),
        ('l-cache','--workload l --pool 512 --support 256 --value-cache 2048'),
        ('l-retain','--workload l --pool 1216 --support 256'),
        ('l-support','--workload l --pool 512 --support 768'),
    ],
    'd':[
        ('d-base','--workload d --pool 256 --support 256'),
        ('d-cache','--workload d --pool 256 --support 256 --value-cache 2048'),
        ('d-support','--workload d --pool 256 --support 832'),
    ],
    'backend':[(f'f-{backend}-{workers}-{schedule}',
        f'--workload f --pool 600 --batch 256 --workers {workers} --backend {backend} --schedule {schedule}')
        for backend,schedule in [('threadpool','static'),('openmp','static'),('openmp','dynamic')]
        for workers in (1,2,4)],
}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('campaign',choices=CAMPAIGNS)
    p.add_argument('--rounds',type=int,default=2)
    p.add_argument('--repeats',type=int,default=3)
    args=p.parse_args()
    reference=None
    for trial in range(args.rounds):
        configs=CAMPAIGNS[args.campaign]
        if trial%2:
            configs=list(reversed(configs))
        for name,options in configs:
            output=ROOT/f'runtime-{name}-r{trial}.json'
            command=[sys.executable,'scripts/analysis_core/probe_runtime.py',*options.split(),
                     '--repeats',str(args.repeats),'--output',str(output)]
            with output.with_suffix('.log').open('w') as log:
                subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
            result=json.loads(output.read_text())
            signatures={r['signature'] for r in result['runs']}
            if len(signatures)!=1 or (reference is not None and next(iter(signatures))!=reference):
                raise AssertionError('candidate changed complete output')
            reference=next(iter(signatures))
            print(name,trial,'wall',[round(r['wall_seconds'],5) for r in result['runs']],
                  'bytes',result.get('pool_controlled_bytes'),flush=True)


if __name__=='__main__':
    main()
