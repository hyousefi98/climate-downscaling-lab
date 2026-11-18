"""CPU-only launcher contracts: no model imports, GPUs, or external artifacts."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_two_ranks_initialize_once_and_resume_without_stale_release(tmp_path):
    pytest.importorskip('omegaconf')
    run = tmp_path / 'training'
    program = r'''
import json, os, sys
from pathlib import Path
from types import SimpleNamespace
from omegaconf import OmegaConf
from workflows import _launch
run = Path(sys.argv[1]); resume = sys.argv[2] == 'yes'
def mock_runtime():
    assert 'torch.distributed' not in sys.modules
    config = OmegaConf.to_container(OmegaConf.load(run / 'resolved.yaml'), resolve=True)
    assert config['training']['io']['checkpoint_dir'] == str(run)
    (run / ('worker-' + os.environ['RANK'] + '.json')).write_text(json.dumps(config, sort_keys=True))
_launch.check_inputs = lambda *args: None
_launch.importlib = SimpleNamespace(import_module=lambda name: SimpleNamespace(main=mock_runtime))
sys.argv = ['train', '--run-dir', str(run)] + (['--resume'] if resume else [])
_launch.main('train')
'''
    for resume in ('no', 'yes'):
        processes = []
        for rank in (1, 0):
            env = dict(os.environ, RANK=str(rank), WORLD_SIZE='2', LOCAL_RANK=str(rank),
                       TORCHELASTIC_RUN_ID='reused-test-id', MASTER_ADDR='localhost', MASTER_PORT='29500')
            processes.append(subprocess.Popen([sys.executable, '-c', program, str(run), resume], cwd=ROOT,
                                             env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
        for process in processes:
            stdout, stderr = process.communicate(timeout=30)
            assert process.returncode == 0, stdout + stderr
        assert (run / 'worker-0.json').read_text() == (run / 'worker-1.json').read_text()
    # Reuse the same launch signature once more: old ready files must not bypass
    # participation from the two new processes.
    processes = [subprocess.Popen([sys.executable, '-c', program, str(run), 'yes'], cwd=ROOT,
                 env=dict(os.environ, RANK=str(rank), WORLD_SIZE='2', LOCAL_RANK=str(rank),
                          TORCHELASTIC_RUN_ID='reused-test-id', MASTER_ADDR='localhost', MASTER_PORT='29500'),
                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for rank in (1, 0)]
    for process in processes:
        stdout, stderr = process.communicate(timeout=30)
        assert process.returncode == 0, stdout + stderr


def test_all_packaged_directories_are_protected():
    from workflows._execution import guard_output, SOURCE_DIRECTORIES
    for directory in SOURCE_DIRECTORIES:
        with pytest.raises(ValueError, match='source directories'):
            guard_output(ROOT / directory / 'accidental-output.nc')


def test_matched_timestamps_reject_duplicate_or_partial_selection(tmp_path):
    from workflows.reproduce_matched import read_times
    path = tmp_path / 'times.json'
    path.write_text(json.dumps(['2012-01-01T00:00:00'] * 200))
    with pytest.raises(ValueError, match='unique'):
        read_times(path)
    path.write_text(json.dumps(['2012-01-01T00:00:00']))
    with pytest.raises(ValueError, match='exact 200'):
        read_times(path)


def test_corrected_training_and_generation_share_coastal_buffer():
    pytest.importorskip('omegaconf')
    from workflows._launch import configuration, parser
    for experiment, expected in [('aorc', 5), ('wrf', 0), ('era5_conus404', 0)]:
        for action, stage in [('train', 'regression'), ('train', 'diffusion'), ('generate', None)]:
            argv = ['--experiment', experiment]
            if stage:
                argv += ['--stage', stage]
            args = parser(action).parse_args(argv)
            _, cfg = configuration(args, action)
            assert cfg['dataset']['coastal_buffer_pixels'] == expected
    _, legacy = configuration(parser('generate').parse_args(['--profile', 'legacy']), 'generate')
    assert legacy['dataset']['coastal_buffer_pixels'] == 10
    _, override = configuration(parser('generate').parse_args(['--set', 'dataset.coastal_buffer_pixels=7']), 'generate')
    assert override['dataset']['coastal_buffer_pixels'] == 7
