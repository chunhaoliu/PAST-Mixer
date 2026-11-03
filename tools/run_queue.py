"""Run a JSON list of run.py jobs across GPUs with a fixed number of workers per GPU.

Each job: {"name": str, "args": [cli tokens...]}. Results/checkpoints paths and
--model_id are taken from the job; a job whose metrics.json already exists is skipped.

Usage:
    python tools/run_queue.py --jobs jobs.json --out OUTDIR --gpus 0,1 --per_gpu 3
"""
import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def _done(job, out):
    results = out / 'results'
    return any(results.glob(f"*_{job['name']}_*/metrics.json"))


def _worker(gpu, jobs, out, python, status, lock, checkpoints=None):
    while True:
        try:
            job = jobs.get_nowait()
        except queue.Empty:
            return
        log_path = out / 'logs' / f"{job['name']}.log"
        if _done(job, out):
            with lock:
                status[job['name']] = 'skipped (metrics exist)'
            continue
        # Each job sees only its own card (avoids stray contexts on cuda:0 from pinned-memory threads).
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        cmd = [python, '-u', str(PROJECT / 'run.py'), *job['args'],
               '--model_id', job['name'], '--gpu', '0',
               '--checkpoints', str(checkpoints or out / 'checkpoints'), '--results', str(out / 'results')]
        start = time.time()
        with lock:
            status[job['name']] = f'running on cuda:{gpu}'
        with open(log_path, 'w', encoding='utf-8') as log:
            log.write(' '.join(cmd) + '\n')
            log.flush()
            code = subprocess.call(cmd, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT, env=env)
        minutes = (time.time() - start) / 60
        with lock:
            status[job['name']] = f'exit={code} gpu={gpu} minutes={minutes:.1f}'
            (out / 'queue_status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobs', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--checkpoints', help='Checkpoint root (default: <out>/checkpoints)')
    parser.add_argument('--gpus', default='0')
    parser.add_argument('--per_gpu', type=int, default=2)
    parser.add_argument('--python', default=sys.executable)
    args = parser.parse_args()

    out = Path(args.out)
    (out / 'logs').mkdir(parents=True, exist_ok=True)
    job_list = json.loads(Path(args.jobs).read_text(encoding='utf-8'))
    names = [job['name'] for job in job_list]
    if len(names) != len(set(names)):
        raise ValueError('job names must be unique')
    jobs = queue.Queue()
    for job in job_list:
        jobs.put(job)
    status = {name: 'queued' for name in names}
    lock = threading.Lock()
    threads = []
    for gpu in [int(g) for g in args.gpus.split(',')]:
        for _ in range(args.per_gpu):
            thread = threading.Thread(target=_worker, args=(gpu, jobs, out, args.python, status, lock, args.checkpoints))
            thread.start()
            threads.append(thread)
            time.sleep(2)
    for thread in threads:
        thread.join()
    (out / 'queue_status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
    failed = [name for name, state in status.items() if state.startswith('exit=') and not state.startswith('exit=0')]
    print(json.dumps(status, indent=2))
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
