"""Serialize builds and measurements across this host's exploration worktrees."""

import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lock', type=Path, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('provide a command after --')
    args.lock.parent.mkdir(parents=True, exist_ok=True)
    with args.lock.open('a+') as lock:
        announced = 0.
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic()-announced >= 30:
                    print('Waiting for the shared compute slot.', flush=True)
                    announced = time.monotonic()
                time.sleep(1)
        lock.seek(0)
        lock.truncate()
        lock.write(f'pid={os.getpid()} command={command!r}\n')
        lock.flush()
        print('Acquired the shared compute slot.', flush=True)
        # The inherited descriptor keeps the lease until the child exits, even
        # if the wrapper is interrupted. Never remove the shared lock file.
        result = subprocess.run(command, pass_fds=(lock.fileno(),))
        raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
