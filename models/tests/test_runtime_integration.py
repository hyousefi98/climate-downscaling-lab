"""Small CPU integration checks; research dependencies are optional in core CI."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.integration


def prepared_fixture(tmp_path):
    pytest.importorskip('torch')
    pytest.importorskip('physicsnemo')
    xr = pytest.importorskip('xarray')
    from data.source_profiles import AORC_CHANNELS
    from data.preprocessing import write_zarr_new, training_statistics
    from models.datasets.physical import PhysicalPairDataset
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location('aorc_fixture_prepare', root / 'scripts/preprocess/aorc.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    times = np.arange('2012-01-01T01', '2012-01-01T19', dtype='datetime64[h]')
    packed = [10, 3000, 2000, 9000, 100, 2900, 30, 40]
    fields = {n: (('time', 'latitude', 'longitude'), np.broadcast_to(v + np.arange(18, dtype=np.int16)[:, None, None], (18, 96, 96)).copy()) for n, v in zip(AORC_CHANNELS, packed)}
    source = xr.Dataset(fields, coords={'time': times, 'latitude': 30 + np.arange(96) / 120, 'longitude': -115 + np.arange(96) / 120})
    inputs, targets = module.prepare(source)
    stats = training_statistics(inputs, targets, np.ones((32, 32), bool), input_array='aorc', target_array='aorc', start='2012-01-01', end='2013-01-01')
    for name, ds in [('input', inputs), ('target', targets)]: write_zarr_new(ds, tmp_path / (name + '.zarr'))
    path = tmp_path / 'stats.json'; path.write_text(json.dumps(stats))
    arguments = dict(input_data_path=tmp_path / 'input.zarr', target_data_path=tmp_path / 'target.zarr', stats_path=path,
                     input_variables=AORC_CHANNELS, output_variables=AORC_CHANNELS, train=True, train_years=(2012, 2012), crop_shape=(32, 32))
    return PhysicalPairDataset(**arguments), arguments, targets


def test_preparation_runtime_unet_backward_diffusion_forward_and_metrics(tmp_path):
    torch = pytest.importorskip('torch'); pytest.importorskip('physicsnemo')
    from physicsnemo.models.diffusion import UNet, EDMPrecondSuperResolution
    from evaluation.metrics import basic_metrics
    torch.set_num_threads(1)
    dataset, _, targets = prepared_fixture(tmp_path)
    target, inputs = dataset[0]
    assert target.shape == inputs.shape == (8, 32, 32)
    assert torch.isfinite(target).all() and torch.isfinite(inputs).all()
    decoded = dataset.denormalize_output(target.numpy())
    np.testing.assert_allclose(decoded[:, dataset.loss_mask], targets.aorc.values[0, :, dataset.loss_mask].T, rtol=2e-6, atol=2e-5)
    options = dict(img_resolution=(32, 32), img_out_channels=8, model_channels=16, channel_mult=(1, 1),
                   num_blocks=1, attn_resolutions=(), N_grid_channels=4, embedding_type='zero', use_apex_gn=False)
    regression = UNet(img_in_channels=12, **options)
    optimizer = torch.optim.Adam(regression.parameters(), lr=1e-4)
    prediction = regression(torch.zeros_like(target)[None], inputs[None])
    loss = (prediction[0, :, dataset.loss_mask] - target[:, dataset.loss_mask]).square().mean()
    loss.backward()
    assert torch.isfinite(loss) and any(p.grad is not None and torch.count_nonzero(p.grad) for p in regression.parameters())
    optimizer.step()
    diffusion = EDMPrecondSuperResolution(img_in_channels=20, sigma_data=1., **options)
    with torch.inference_mode():
        denoised = diffusion(torch.zeros_like(target)[None], torch.cat([inputs[None], prediction.detach()], dim=1), torch.tensor([.5]))
    assert denoised.shape == (1, 8, 32, 32) and torch.isfinite(denoised).all()
    metrics = basic_metrics(decoded[0], decoded[0], dataset.loss_mask)
    assert metrics['RMSE'] == 0


def test_wrong_units_and_invalid_land_values_are_rejected(tmp_path):
    pytest.importorskip('torch'); pytest.importorskip('physicsnemo')
    import zarr
    from models.datasets.physical import PhysicalPairDataset
    dataset, arguments, _ = prepared_fixture(tmp_path)
    source = zarr.open_group(str(arguments['input_data_path']), mode='a')
    y, x = np.argwhere(dataset.loss_mask)[0]
    source['aorc'][0, 5, y, x] = -100.
    with pytest.raises(ValueError, match='diagnostic bounds'): dataset[0]
    source['aorc'][0, 5, y, x] = np.nan
    with pytest.raises(ValueError, match='Nonfinite'): dataset[0]
    source.attrs['numeric_profile'] = 'legacy_model'
    with pytest.raises(ValueError, match='numeric_profile'): PhysicalPairDataset(**arguments)
    source.attrs['numeric_profile'] = 'physical'
    import netCDF4
    mask_file = tmp_path / 'shifted_mask.nc'
    with netCDF4.Dataset(mask_file, 'w') as handle:
        handle.createDimension('y', 32); handle.createDimension('x', 32)
        handle.createVariable('LANDMASK', 'i1', ('y', 'x'))[:] = 1
        handle.createVariable('lat', 'f8', ('y', 'x'))[:] = source['lat'][:] + .01
        handle.createVariable('lon', 'f8', ('y', 'x'))[:] = source['lon'][:]
    with pytest.raises(ValueError, match='Land-mask coordinates'):
        PhysicalPairDataset(**arguments, landmask_path=mask_file)


def test_frozen_artifact_binding_and_humidity_coordinate(tmp_path):
    pytest.importorskip('torch'); pytest.importorskip('physicsnemo')
    from models.datasets.frozen_aorc import FrozenAORCPairDataset, file_sha256
    dataset, arguments, _ = prepared_fixture(tmp_path)
    # Use simple declared checkpoint-coordinate statistics to isolate the unit map.
    names = dataset.input_variables
    stats = {'input': {n: {'mean': 0., 'std': 1.} for n in names}, 'output': {n: {'mean': 0., 'std': 1.} for n in names}}
    arguments['stats_path'].write_text(json.dumps(stats))
    interface = tmp_path / 'interface.json'; interface.write_text(json.dumps({'model_coordinate': 'aorc_legacy_10q_hpa', 'sha256': {'normalization': file_sha256(arguments['stats_path'])}}))
    frozen = FrozenAORCPairDataset(checkpoint_interface_path=interface, input_source='synthetic', **arguments)
    target, _ = frozen[0]
    physical = frozen.denormalize_output(target.numpy())
    np.testing.assert_allclose(target.numpy()[4, frozen.loss_mask], 10 * physical[4, frozen.loss_mask], rtol=1e-6)
    arguments['stats_path'].write_text(json.dumps(dict(stats, changed=True)))
    with pytest.raises(ValueError, match='hash mismatch'):
        FrozenAORCPairDataset(checkpoint_interface_path=interface, input_source='synthetic', **arguments)
    arguments['stats_path'].write_text(json.dumps(dict(stats, metadata={'numeric_profile': 'physical'})))
    interface.write_text(json.dumps({'model_coordinate': 'aorc_legacy_10q_hpa', 'sha256': {'normalization': file_sha256(arguments['stats_path'])}}))
    with pytest.raises(ValueError, match='cannot replace frozen'):
        FrozenAORCPairDataset(checkpoint_interface_path=interface, input_source='synthetic', **arguments)


@pytest.mark.parametrize('field', ['precip_threshold', 'mask_sha256', 'included_timestamps'])
def test_physical_statistics_require_threshold_and_support_binding(tmp_path, field):
    pytest.importorskip('torch'); pytest.importorskip('physicsnemo')
    from models.datasets.physical import PhysicalPairDataset
    _, arguments, _ = prepared_fixture(tmp_path)
    stats = json.loads(arguments['stats_path'].read_text())
    stats['metadata'].pop(field)
    arguments['stats_path'].write_text(json.dumps(stats))
    with pytest.raises(ValueError, match=field):
        PhysicalPairDataset(**arguments)


def test_physical_statistics_training_membership_and_holdout_separation(tmp_path):
    pytest.importorskip('torch'); pytest.importorskip('physicsnemo')
    from models.datasets.physical import PhysicalPairDataset
    _, arguments, _ = prepared_fixture(tmp_path)
    original = json.loads(arguments['stats_path'].read_text())
    times = original['metadata']['included_timestamps']
    subset = PhysicalPairDataset(**arguments, requested_times=times[:1])
    assert len(subset) == 1
    with pytest.raises(ValueError, match='Evaluation timestamps overlap'):
        PhysicalPairDataset(**dict(arguments, train=False, valid_years=(2012, 2012)))
    # Simulate a statistics file from another selected training period, retaining
    # the other generated metadata. The full training split must reject it.
    original['metadata']['included_timestamps'] = times[:1]
    arguments['stats_path'].write_text(json.dumps(original))
    with pytest.raises(ValueError, match='Selected training timestamps differ'):
        PhysicalPairDataset(**arguments)
    with pytest.raises(ValueError, match='not a subset'):
        PhysicalPairDataset(**arguments, requested_times=times[1:2])
    # A declared held-out subset disjoint from the statistics period remains valid.
    heldout = PhysicalPairDataset(**dict(arguments, train=False, valid_years=(2012, 2012)), requested_times=times[1:])
    assert len(heldout) == len(times) - 1


@pytest.mark.parametrize('field,wrong', [('temporal_profile', 'historical_forward'),
                                       ('precipitation_interval', '(t,t+6h]'),
                                       ('pressure_units', 'Pa'),
                                       ('humidity_representation', 'mixing_ratio')])
def test_physical_statistics_reject_wrong_time_and_unit_metadata(tmp_path, field, wrong):
    pytest.importorskip('torch'); pytest.importorskip('physicsnemo')
    from models.datasets.physical import PhysicalPairDataset
    _, arguments, _ = prepared_fixture(tmp_path)
    stats = json.loads(arguments['stats_path'].read_text())
    stats['metadata'][field] = wrong
    arguments['stats_path'].write_text(json.dumps(stats))
    with pytest.raises(ValueError, match='Statistics and source metadata differ: ' + field):
        PhysicalPairDataset(**arguments)


def test_legacy_quality_policy_runs_before_zero_imputation():
    pytest.importorskip('torch'); pytest.importorskip('physicsnemo')
    from types import SimpleNamespace
    from models.datasets.quality import check_legacy_arrays
    dataset = SimpleNamespace(input_variables=['TMP_2maboveground'], output_variables=['TMP_2maboveground'],
                              loss_mask=np.ones((2, 2), bool), quality_policy='strict')
    good = np.full((1, 2, 2), 2900., np.float32)
    check_legacy_arrays(dataset, good, good)
    bad = good.copy(); bad[0, 0, 0] = -32767.
    with pytest.raises(ValueError, match='diagnostic bounds'): check_legacy_arrays(dataset, bad, good)
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match='Nonfinite'): check_legacy_arrays(dataset, bad, good)
    dataset.quality_policy = 'archive_replay'
    check_legacy_arrays(dataset, bad, good)


def test_physical_time_calendar_and_crop_guards(tmp_path):
    pytest.importorskip('torch'); pytest.importorskip('physicsnemo')
    import zarr
    from models.datasets.physical import PhysicalStore, PhysicalPairDataset
    _, arguments, _ = prepared_fixture(tmp_path)
    with pytest.raises(ValueError, match='exceeds source'):
        PhysicalPairDataset(**dict(arguments, crop_shape=(64, 32)))
    source = zarr.open_group(str(arguments['input_data_path']), mode='a')
    source['time'].attrs['calendar'] = '360_day'
    with pytest.raises(ValueError, match='Gregorian'):
        PhysicalStore(arguments['input_data_path'])
    source['time'].attrs['calendar'] = 'standard'
    original = source['time'][:]
    source['time'][:] = original[::-1]
    with pytest.raises(ValueError, match='strictly increasing'):
        PhysicalStore(arguments['input_data_path'])
    attributes = dict(source['time'].attrs)
    del source['time']
    source.create_dataset('time', data=original.astype(float) + .5)
    source['time'].attrs.update(attributes)
    with pytest.raises(ValueError, match='finite integers'):
        PhysicalStore(arguments['input_data_path'])
