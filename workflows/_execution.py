"""Output guards and a filesystem rendezvous before PhysicsNeMo initializes DDP."""
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRECTORIES = ('models', 'workflows', 'legacy', 'configs', 'data', 'scripts',
                      'baselines', 'evaluation', 'figures', 'patches', 'tests', '.git')


def guard_output(path):
    path = Path(path).expanduser().resolve()
    if path == ROOT:
        raise ValueError('Use a dedicated output directory, not the repository root')
    for component in SOURCE_DIRECTORIES:
        source = ROOT / component
        if path == source or source in path.parents:
            raise ValueError('Outputs cannot be written into packaged source directories: ' + str(path))
    return path


def atomic_text(path, text):
    path = Path(path)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        temporary.write_text(text)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json_if_ready(path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None


def initialize_run(run, cfg, action, resume=False, environ=None, timeout=180):
    """Write one atomic config, coordinating torchrun ranks without a process group.

    A fresh challenge and per-rank nonce prevent a prior run's readiness file
    from releasing a resumed job. All ranks must share the run filesystem.
    The small adjacent .<run>.launch-* directory is an execution artifact.
    """
    from omegaconf import OmegaConf
    run = guard_output(run)
    environment = os.environ if environ is None else environ
    rank, world = int(environment.get('RANK', '0')), int(environment.get('WORLD_SIZE', '1'))
    if world < 1 or not 0 <= rank < world:
        raise ValueError('Invalid RANK/WORLD_SIZE')
    text = OmegaConf.to_yaml(OmegaConf.create(cfg), resolve=True)
    path = run / 'resolved.yaml'

    def prepare():
        if run.exists() and (not run.is_dir() or any(run.iterdir())) and not (resume and action == 'train'):
            raise FileExistsError('Run directory is nonempty; choose another directory or use --resume for training')
        if action == 'generate':
            output = guard_output(cfg['generation']['io']['output_filename'])
            if output.exists():
                raise FileExistsError(output)
        run.mkdir(parents=True, exist_ok=True)
        atomic_text(path, text)

    if world == 1:
        prepare()
        return path
    signature = hashlib.sha256(json.dumps({
        'run': str(run), 'configuration': text, 'world_size': world, 'action': action,
        'resume': resume, 'rendezvous': [environment.get(key, '') for key in
            ('TORCHELASTIC_RUN_ID', 'MASTER_ADDR', 'MASTER_PORT')],
    }, sort_keys=True).encode()).hexdigest()[:24]
    rendezvous = run.parent / ('.' + run.name + '.launch-' + signature)
    rendezvous.mkdir(parents=True, exist_ok=True)
    nonce, deadline = uuid.uuid4().hex, time.monotonic() + timeout
    challenge_path, ready_path = rendezvous / 'challenge.json', rendezvous / 'ready.json'
    if rank == 0:
        token = uuid.uuid4().hex
        atomic_text(challenge_path, json.dumps({'token': token, 'world_size': world}))
    acknowledged = None
    while time.monotonic() < deadline:
        challenge = read_json_if_ready(challenge_path)
        if challenge and challenge['token'] != acknowledged:
            acknowledged = challenge['token']
            atomic_text(rendezvous / f'rank-{rank}.json', json.dumps({'token': acknowledged, 'nonce': nonce}))
        if rank == 0 and acknowledged == token:
            acknowledgments = {str(i): read_json_if_ready(rendezvous / f'rank-{i}.json') for i in range(world)}
            if all(value and value['token'] == token for value in acknowledgments.values()):
                release = {'token': token, 'nonces': {key: value['nonce'] for key, value in acknowledgments.items()}}
                try:
                    prepare()
                except Exception as error:
                    release.update(status='error', message=str(error))
                    atomic_text(ready_path, json.dumps(release))
                    raise
                release['status'] = 'ready'
                atomic_text(ready_path, json.dumps(release))
                return path
        release = read_json_if_ready(ready_path)
        if (release and release.get('token') == acknowledged
                and release.get('nonces', {}).get(str(rank)) == nonce):
            if release['status'] != 'ready':
                raise RuntimeError('Rank 0 could not initialize the run: ' + release['message'])
            return path
        time.sleep(.05)
    raise TimeoutError(f'Launch rendezvous timed out at rank {rank}; check other ranks, matching arguments, and shared filesystem: {rendezvous}')
