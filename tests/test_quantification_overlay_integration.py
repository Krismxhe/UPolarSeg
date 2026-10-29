"""Small real-module checks for independent, source-consistent overlay exports."""

from __future__ import annotations

import csv
import importlib
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip('torch')
omegaconf = pytest.importorskip('omegaconf')
Image = pytest.importorskip('PIL.Image')


def _mask(extra_width=0):
    mask = torch.zeros(16, 32, dtype=torch.long)
    mask[:, :7] = 1
    mask[:, 7:10 + extra_width] = 2
    return mask


def _logits(masks):
    return torch.nn.functional.one_hot(masks, 3).permute(0, 3, 1, 2).float() * 16.0


class _Network(torch.nn.Module):
    def __init__(self, masks):
        super().__init__()
        self.register_buffer('logits', _logits(masks))
        self.register_buffer('feature', torch.ones(masks.shape[0], 4, 16, 32))
        self.calls = []

    def forward(self, images, return_features=False):
        self.calls.append(bool(return_features))
        return {'seg_logits': self.logits, 'features': {'decoder_final': self.feature}}


class _Adapter(torch.nn.Module):
    def __init__(self, masks):
        super().__init__()
        self.register_buffer('logits', _logits(masks))
        self.calls = 0

    def forward(self, features, logits):
        self.calls += 1
        return self.logits


class _Loss(torch.nn.Module):
    def forward(self, logits, masks):
        loss = torch.nn.functional.cross_entropy(logits, masks)
        return loss, loss, loss


@pytest.fixture
def module_factory(monkeypatch, tmp_path):
    implementation = pytest.importorskip('src.models.seg_module')

    def make(*, overlay='none', gt_overlay='none', layout='combined', source='seg_logits', readouts='both', masks='none',
             adapter=False, polar=True, polar_mode='linear', count=1, cfg=None):
        if cfg is None:
            cfg = omegaconf.OmegaConf.create({
                'dataset': {
                    'name': 'synthetic_polar', 'num_classes': 3, 'mask_mode': 'index',
                    'class_names': ['background', 'OD', 'PPA'], 'foreground_classes': [1, 2],
                },
                'model': {'arch': 'tiny', 'decoder_channels': [4]},
                'train': {'img_size': 32},
                'polar': {
                    'enabled': polar,
                    'geometry': {'mode': polar_mode, 'center': None, 'max_radius': 32.0,
                                 'angle_bins': 16, 'radius_bins': 32, 'pt_type': 'corner'},
                    'eval': {'metric_space': 'cart'},
                },
                'quantification': {
                    'enabled': adapter, 'adapter_enabled': adapter, 'weight': 0.1,
                    'dv_mode': 'max_center_chord', 'denominator_floor_px': 1.0,
                },
                'eval': {
                    'save_csv': False, 'save_predictions': masks,
                    'save_quantification_overlays': overlay,
                    'save_quantification_gt_overlays': gt_overlay,
                    'quantification_overlay_layout': layout,
                    'quantification_overlay_source': source,
                    'quantification_overlay_readouts': readouts,
                    'quantification_overlay_legend': False,
                    'prediction_cartesian_size': 'original',
                    'output_dir': str(tmp_path / 'export'),
                    'odppa': {'enabled': False, 'bootstrap_samples': 8, 'permutation_samples': 8},
                },
                'logging': {'save_dir': str(tmp_path), 'name': 'run'},
                'hardware': {'accelerator': 'cpu', 'devices': 1, 'num_nodes': 1, 'strategy': 'auto'},
                'split': 'test', 'checkpoint': 'unused.ckpt',
            })
        network = _Network(_mask().unsqueeze(0).repeat(count, 1, 1))
        monkeypatch.setattr(implementation, 'build_model', lambda *_: network)
        monkeypatch.setattr(implementation, 'build_loss', lambda *_: _Loss())
        model = implementation.SegModule(cfg).eval()
        monkeypatch.setattr(model, 'log', lambda *args, **kwargs: None)
        if adapter:
            model.quant_head = _Adapter(_mask(extra_width=2).unsqueeze(0).repeat(count, 1, 1))
        return model

    return make


def _batch(count=1, *, image_path='', orig_size=True):
    from torch.utils.data import default_collate

    samples = []
    for index in range(count):
        sample = {
            'image': torch.zeros(3, 16, 32), 'mask': _mask(),
            'case_id': f'{index + 1:03d}', 'image_path': image_path, 'mask_path': '',
        }
        if orig_size:
            sample['orig_size'] = [(48, 80), (64, 40)][index % 2]
        samples.append(sample)
    return default_collate(samples)


def _readout_calls(monkeypatch, readout):
    calls = {'hard': 0, 'soft': 0}
    for key, method in [('hard', 'measure_mask'), ('soft', 'read_probabilities')]:
        original = getattr(readout, method)

        def counted(*args, _original=original, _key=key, **kwargs):
            calls[_key] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(readout, method, counted)
    return calls


def _capture_saves(monkeypatch, writer):
    saved = []
    original = writer.save

    def save(mask, case_id, **kwargs):
        saved.append((None if mask is None else mask.copy(), case_id, kwargs))
        return original(mask, case_id, **kwargs)

    monkeypatch.setattr(writer, 'save', save)
    return saved


def test_default_off_has_no_readout_or_export_side_effects(module_factory, monkeypatch, tmp_path):
    implementation = importlib.import_module('src.models.seg_module')

    def forbidden(*args, **kwargs):
        raise AssertionError('Disabled overlay must not construct measurement geometry or a writer')

    monkeypatch.setattr(implementation, 'build_polar_readout', forbidden)
    overlay_module = importlib.import_module('src.evaluation.quantification_overlays')
    monkeypatch.setattr(overlay_module, 'QuantificationOverlayWriter', forbidden)
    model = module_factory(polar=False)
    model.on_test_epoch_start()
    model.test_step(_batch(), 0)
    assert model._quantification_readout() is None
    assert model._quantification_overlay_writer is None
    assert model._prediction_writer is None
    assert model.model.calls == [False]
    assert not any(key.startswith('quant_') for key in model.test_records[0])
    assert model.test_records[0]['overlay_polar_path'] == ''
    assert not (tmp_path / 'export').exists()


@pytest.mark.parametrize('polar,polar_mode,reason', [
    (False, 'linear', 'requires_polar_coordinates'),
    (True, 'log', 'requires_linear_radial_grid'),
])
def test_enabled_overlay_rejects_unsupported_input_geometry(module_factory, polar, polar_mode, reason):
    with pytest.raises(ValueError, match=reason):
        module_factory(overlay='both', polar=polar, polar_mode=polar_mode)


def test_overlay_only_reuses_diagnostics_and_restores_each_original_size(module_factory, monkeypatch, tmp_path):
    model = module_factory(overlay='both', count=2)
    assert model.quantification is None
    assert model._odppa_eval_enabled is False
    calls = _readout_calls(monkeypatch, model._quantification_readout())
    model.on_test_epoch_start()
    model.test_step(_batch(2), 0)
    # Existing diagnostics reads prediction/GT once each; rendering adds no reads.
    assert calls == {'hard': 2, 'soft': 2}
    assert model.model.calls == [False]
    assert model._prediction_writer is None
    for row, expected_hw in zip(model.test_records, [(48, 80), (64, 40)]):
        assert row['pred_path'] == row['pred_polar_path'] == row['pred_cartesian_path'] == ''
        assert Path(row['overlay_metadata_path']).is_file()
        metadata = json.loads(Path(row['overlay_metadata_path']).read_text())
        assert 'dh' in metadata['readouts']['hard']
        assert 'dh' in metadata['readouts']['soft']
        with Image.open(row['overlay_polar_path']) as image:
            assert image.mode == 'RGB'
            assert (image.height, image.width) == (16, 32)
        with Image.open(row['overlay_cartesian_path']) as image:
            assert image.mode == 'RGB'
            assert (image.height, image.width) == expected_hw
        assert all(not isinstance(value, (np.ndarray, torch.Tensor)) for value in row.values())
    assert not (tmp_path / 'export' / 'masks_polar').exists()
    assert not (tmp_path / 'export' / 'masks_cartesian').exists()


@pytest.mark.parametrize('adapter,source,actual_source,expected_width,expected_reads', [
    (False, 'quantification', 'seg_logits', 3.0, 2),
    (True, 'seg_logits', 'seg_logits', 3.0, 3),
    (True, 'quantification', 'quant_logits', 5.0, 2),
])
def test_overlay_source_matches_its_mask_and_reuses_each_forward(
    module_factory, monkeypatch, adapter, source, actual_source, expected_width, expected_reads,
):
    model = module_factory(overlay='polar', masks='polar', adapter=adapter, source=source)
    calls = _readout_calls(monkeypatch, model._quantification_readout())
    model.on_test_epoch_start()
    saves = _capture_saves(monkeypatch, model._quantification_overlay_writer)
    model.test_step(_batch(), 0)
    assert model.model.calls == [adapter]
    if adapter:
        assert model.quant_head.calls == 1
    assert calls == {'hard': expected_reads, 'soft': expected_reads}
    mask, case_id, values = saves[0]
    assert case_id == '001'
    assert values['prediction_source'] == actual_source
    assert float(values['hard']['pmw']) == expected_width
    np.testing.assert_array_equal(mask, _mask(extra_width=2 if actual_source == 'quant_logits' else 0))
    # Measurements may use the adapter, but saved labels always use segmentation logits.
    with Image.open(model.test_records[0]['pred_polar_path']) as pure_mask:
        assert pure_mask.mode != 'RGB'
        np.testing.assert_array_equal(np.asarray(pure_mask), _mask())
        assert set(np.unique(pure_mask)) == {0, 1, 2}
    assert model.test_records[0]['per_class'][2]['dice'] == 1.0


@pytest.mark.parametrize('readouts,expected', [
    ('hard', {'hard': 1, 'soft': 0}),
    ('soft', {'hard': 0, 'soft': 1}),
    ('both', {'hard': 1, 'soft': 1}),
])
def test_cli_batch_helper_needs_no_gt_and_only_reads_requested_methods(module_factory, monkeypatch, readouts, expected):
    model = module_factory(overlay='polar', readouts=readouts)
    calls = _readout_calls(monkeypatch, model._quantification_readout())
    outputs, quant_logits, source = model._evaluation_outputs(torch.zeros(1, 3, 16, 32))
    payload = model._quantification_overlay_batch(outputs['seg_logits'], quant_logits, source)
    assert calls == expected
    assert model.model.calls == [False]
    assert (payload['hard'] is not None) == (readouts in {'hard', 'both'})
    assert (payload['soft'] is not None) == (readouts in {'soft', 'both'})


def test_overlay_original_size_falls_back_to_image_path(module_factory, tmp_path):
    image_path = tmp_path / 'original.png'
    Image.new('RGB', (67, 43)).save(image_path)
    model = module_factory(overlay='cartesian')
    model.on_test_epoch_start()
    model.test_step(_batch(image_path=str(image_path), orig_size=False), 0)
    with Image.open(model.test_records[0]['overlay_cartesian_path']) as image:
        assert image.size == (67, 43)


@pytest.mark.parametrize('odppa', [False, True])
def test_overlay_paths_are_appended_and_preserved_in_all_case_csvs(module_factory, monkeypatch, tmp_path, odppa):
    from src.evaluation.evaluator import OVERLAY_PATH_FIELDS, SegEvaluator
    from src.evaluation.error_maps import ERROR_MAP_PATH_FIELDS
    from src.evaluation.quantification_stats import read_reference_csv

    model = module_factory(overlay='polar')
    model.on_test_epoch_start()
    model.test_step(_batch(), 0)
    model.cfg.eval.odppa.enabled = odppa
    if odppa:
        statistics = importlib.import_module('src.evaluation.odppa_stats')
        monkeypatch.setattr(statistics, 'summarize_records', lambda *args, **kwargs: [])
        model.test_records[0].update({
            'reference_complete': True, 'od_reference_valid': True,
            'od_pred_empty': False, 'od_hd95_valid': True,
            'ppa_pred_empty': False, 'ppa_hd95_valid': True,
        })
    evaluator = SegEvaluator(model.cfg, split='test', run_name='tiny')
    outputs = evaluator.write(model.test_records, tmp_path / 'csv')
    for name in ['per_case', 'quantification_per_case']:
        with Path(outputs[name]).open(newline='') as stream:
            reader = csv.DictReader(stream)
            offset = reader.fieldnames.index(OVERLAY_PATH_FIELDS[0])
            assert reader.fieldnames[offset:offset + len(OVERLAY_PATH_FIELDS)] == OVERLAY_PATH_FIELDS
            assert reader.fieldnames[-len(ERROR_MAP_PATH_FIELDS):] == list(ERROR_MAP_PATH_FIELDS)
            row = next(reader)
        for key in OVERLAY_PATH_FIELDS:
            assert row[key] == model.test_records[0][key]
        parsed = read_reference_csv(outputs[name])[0]
        assert all(row[key] == parsed[key] == '' for key in ERROR_MAP_PATH_FIELDS)
        assert parsed['case_id'] == '001'
        assert parsed['overlay_polar_path'] == row['overlay_polar_path']


@pytest.mark.parametrize('which', ['prediction', 'groundtruth'])
def test_evaluate_overlay_only_writes_resolved_config(module_factory, monkeypatch, tmp_path, capsys, which):
    entry = pytest.importorskip('evaluate')
    model = module_factory(
        overlay='polar' if which == 'prediction' else 'none',
        gt_overlay='polar' if which == 'groundtruth' else 'none',
    )
    monkeypatch.setattr(entry.SegModule, 'load_from_checkpoint', lambda *args, **kwargs: model)
    monkeypatch.setattr(entry, 'SegDataModule', lambda *args, **kwargs: object())
    monkeypatch.setattr(entry, 'TensorBoardLogger', lambda *args, **kwargs: object())
    monkeypatch.setattr(entry, 'CSVLogger', lambda *args, **kwargs: object())

    class Trainer:
        is_global_zero = True

        def __init__(self, **kwargs):
            pass

        def test(self, model, datamodule):
            model.on_test_epoch_start()
            model.test_step(_batch(), 0)
            return [{'test/dice': 1.0}]

    monkeypatch.setattr(entry.pl, 'Trainer', Trainer)
    entry.main.__wrapped__(model.cfg)
    out = tmp_path / 'export'
    assert (out / 'config.yaml').is_file()
    assert not (out / 'per_case.csv').exists()
    captured = capsys.readouterr().out
    if which == 'prediction':
        assert 'Quantification overlays written' in captured
        assert str(out / 'overlays_polar') in captured
    else:
        assert 'Ground-truth hard quantification overlays written' in captured
        assert str(out / 'overlays_gt_polar') in captured
    assert 'Prediction masks written' not in captured


def test_predict_cli_preserves_overlay_overrides_and_supports_overlay_only(module_factory, monkeypatch, tmp_path):
    import sys

    entry = pytest.importorskip('predict_masks')
    model = module_factory(overlay='polar', readouts='hard', source='quantification')
    cfg = model.cfg
    monkeypatch.setattr(entry, 'build_cfg', lambda *args, **kwargs: cfg)
    monkeypatch.setattr(entry.SegModule, 'load_from_checkpoint', lambda *args, **kwargs: model)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)

    class DataModule:
        def __init__(self, cfg, eval_split):
            pass

        def setup(self, stage):
            assert stage == 'test'

        def test_dataloader(self):
            return [_batch()]

    monkeypatch.setattr(entry, 'SegDataModule', DataModule)
    monkeypatch.setattr(sys, 'argv', [
        'predict_masks.py', '--ckpt', 'unused.ckpt', '--model_config', 'tiny',
        '--dataset', 'synthetic_polar', '--dataset_short', 'sample',
        '--model_name', 'tiny', '--output_root', str(tmp_path / 'cli'), '--save_masks', 'none',
    ])
    calls = _readout_calls(monkeypatch, model._quantification_readout())
    entry.main()
    assert calls == {'hard': 1, 'soft': 0}
    assert model.model.calls == [False]
    assert cfg.eval.quantification_overlay_source == 'quantification'
    assert (tmp_path / 'cli' / 'tiny' / 'overlays_polar' / 'sample' / '001.png').is_file()
    assert not (tmp_path / 'cli' / 'tiny' / 'masks_polar').exists()
    assert not (tmp_path / 'cli' / 'tiny' / 'masks_cart').exists()


def test_gt_only_reads_target_once_without_prediction_readouts(module_factory, monkeypatch):
    model = module_factory(gt_overlay='both', count=2)
    calls = _readout_calls(monkeypatch, model._quantification_readout())
    model.on_test_epoch_start()
    saved = _capture_saves(monkeypatch, model._quantification_overlay_writer)
    model.test_step(_batch(2), 0)
    assert calls == {'hard': 1, 'soft': 0}
    assert not model.quant_test_metrics
    assert model.model.calls == [False]
    for row, values, original_hw in zip(model.test_records, saved, [(48, 80), (64, 40)]):
        mask, case_id, arguments = values
        assert mask is arguments['hard'] is arguments['soft'] is None
        np.testing.assert_array_equal(arguments['groundtruth_mask'], _mask())
        assert float(arguments['groundtruth_hard']['pmw']) == 3.0
        assert row['overlay_polar_path'] == row['overlay_cartesian_path'] == ''
        assert not any(key.startswith('quant_') for key in row)
        metadata = json.loads(Path(row['overlay_metadata_path']).read_text())
        assert metadata['readouts'] == {}
        assert metadata['groundtruth']['annotation_complete'] is True
        assert metadata['groundtruth']['hard']['raw']['pmw'] == 3.0
        with Image.open(row['overlay_gt_polar_path']) as image:
            assert image.size == (32, 16)
        with Image.open(row['overlay_gt_cartesian_path']) as image:
            assert (image.height, image.width) == original_hw


def test_separate_prediction_and_gt_use_their_own_masks_and_reuse_target_diagnostics(module_factory, monkeypatch):
    model = module_factory(overlay='both', gt_overlay='cartesian', layout='separate',
                           adapter=True, source='quantification', masks='polar')
    calls = _readout_calls(monkeypatch, model._quantification_readout())
    model.on_test_epoch_start()
    saved = _capture_saves(monkeypatch, model._quantification_overlay_writer)
    batch = _batch()
    batch['mask'] = _mask(extra_width=1).unsqueeze(0)
    model.test_step(batch, 0)
    assert model.model.calls == [True]
    assert model.quant_head.calls == 1
    assert calls == {'hard': 2, 'soft': 2}
    mask, _, values = saved[0]
    np.testing.assert_array_equal(mask, _mask(extra_width=2))
    np.testing.assert_array_equal(values['groundtruth_mask'], _mask(extra_width=1))
    assert float(values['hard']['pmw']) == 5.0
    assert float(values['groundtruth_hard']['pmw']) == 4.0
    row = model.test_records[0]
    assert row['overlay_polar_path'] == row['overlay_cartesian_path'] == row['overlay_gt_polar_path'] == ''
    for space in ('polar', 'cartesian'):
        for method in ('hard', 'soft'):
            destination = Path(row[f'overlay_{space}_{method}_path'])
            assert destination.parent.name == method
            assert destination.is_file()
    assert Path(row['overlay_gt_cartesian_path']).is_file()
    with Image.open(row['pred_polar_path']) as image:
        np.testing.assert_array_equal(image, _mask())
    metadata = json.loads(Path(row['overlay_metadata_path']).read_text())
    assert metadata['prediction_source'] == 'quant_logits'
    assert metadata['groundtruth']['hard']['raw']['pmw'] == 4.0


def test_gt_only_rejects_missing_target_without_reading_predictions(module_factory, monkeypatch):
    model = module_factory(gt_overlay='polar')
    calls = _readout_calls(monkeypatch, model._quantification_readout())
    with pytest.raises(ValueError, match='require target masks'):
        model._quantification_overlay_batch(None, None, 'seg_logits')
    assert calls == {'hard': 0, 'soft': 0}


@pytest.mark.parametrize('metadata_source', ['batch', 'csv'])
def test_partial_gt_annotation_is_forwarded_without_disabling_od_diameters(
    module_factory, monkeypatch, tmp_path, metadata_source,
):
    model = module_factory(gt_overlay='polar')
    if metadata_source == 'csv':
        metadata_path = tmp_path / 'case_metadata.csv'
        metadata_path.write_text('case_id,reference_complete\n001,false\n')
        model.cfg.eval.odppa.metadata_csv = str(metadata_path)
        model = module_factory(cfg=model.cfg)
        assert model._odppa_eval_enabled is False
    model.on_test_epoch_start()
    saved = _capture_saves(monkeypatch, model._quantification_overlay_writer)
    batch = _batch()
    batch['reference_complete'] = ['false' if metadata_source == 'batch' else 'true']
    model.test_step(batch, 0)
    assert saved[0][2]['reference_complete'] == 'false'
    metadata = json.loads(Path(model.test_records[0]['overlay_metadata_path']).read_text())
    groundtruth = metadata['groundtruth']
    assert groundtruth['annotation_complete'] is False
    assert groundtruth['hard']['pmw']['annotation_valid'] is False
    assert groundtruth['hard']['dv'] is not None
    assert groundtruth['hard']['dh'] is not None


def test_predict_cli_gt_only_flags_load_gt_without_prediction_readouts(module_factory, monkeypatch, tmp_path):
    import sys

    entry = pytest.importorskip('predict_masks')
    cfg = module_factory().cfg
    metadata_path = tmp_path / 'cli_metadata.csv'
    metadata_path.write_text('case_id,reference_complete\n001,false\n')
    cfg.eval.odppa.metadata_csv = str(metadata_path)
    monkeypatch.setattr(entry, 'build_cfg', lambda *args, **kwargs: cfg)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    created = []

    def load(*args, cfg, **kwargs):
        model = module_factory(cfg=cfg)
        created.append((model, _readout_calls(monkeypatch, model._quantification_readout())))
        return model

    monkeypatch.setattr(entry.SegModule, 'load_from_checkpoint', load)

    class DataModule:
        def __init__(self, cfg, eval_split):
            pass

        def setup(self, stage):
            assert stage == 'test'

        def test_dataloader(self):
            return [_batch()]

    monkeypatch.setattr(entry, 'SegDataModule', DataModule)
    monkeypatch.setattr(sys, 'argv', [
        'predict_masks.py', '--ckpt', 'unused.ckpt', '--model_config', 'tiny',
        '--dataset', 'synthetic_polar', '--dataset_short', 'sample',
        '--model_name', 'tiny', '--output_root', str(tmp_path / 'cli_gt'),
        '--save_masks', 'none', '--save_gt_overlays', 'cartesian', '--overlay_layout', 'separate',
    ])
    entry.main()
    model, calls = created[0]
    assert calls == {'hard': 1, 'soft': 0}
    assert model.model.calls == [False]
    assert cfg.eval.save_quantification_gt_overlays == 'cartesian'
    assert cfg.eval.quantification_overlay_layout == 'separate'
    out = tmp_path / 'cli_gt' / 'tiny'
    assert (out / 'overlays_gt_cartesian' / 'sample' / '001.png').is_file()
    assert not (out / 'overlays_polar').exists()
    assert not (out / 'overlays_cartesian').exists()
    assert not (out / 'masks_polar').exists()
    metadata = json.loads((out / 'overlay_metadata' / 'sample' / '001.json').read_text())
    assert metadata['groundtruth']['annotation_complete'] is False
