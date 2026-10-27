"""Configuration shared by dataset preparation, preflight, and training."""
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path=None):
    path = Path(path) if path else PROJECT_ROOT / 'configs/restoration_v2.yaml'
    with path.open(encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    for key in ('clean_splits', 'assets', 'output'):
        value = Path(config['dataset'][key])
        config['dataset'][key] = str(value if value.is_absolute() else PROJECT_ROOT / value)
    value = Path(config['training']['run_dir'])
    config['training']['run_dir'] = str(value if value.is_absolute() else PROJECT_ROOT / value)
    dataset, training = config['dataset'], config['training']
    if dataset['image_size'] < 32 or dataset['image_size'] % 32:
        raise ValueError('image_size must be a positive multiple of 32')
    if dataset['variants'] < 2:
        raise ValueError('At least two variants are required for single and mixed damage coverage')
    if not dataset['artifact_types'] or len(set(dataset['artifact_types'])) != len(dataset['artifact_types']):
        raise ValueError('artifact_types must be nonempty and unique')
    if not set(dataset['severities']) <= {'light', 'medium', 'heavy'} or not dataset['severities']:
        raise ValueError('Unsupported or empty severity selection')
    if training['reconstruction_epochs'] < 1 or training['refinement_epochs'] < 0:
        raise ValueError('Invalid stage epoch budgets')
    if training['reconstruction_epochs'] + training['refinement_epochs'] > training['max_total_epochs']:
        raise ValueError('Configured stages exceed max_total_epochs')
    if not 0 <= training['warmup_epochs'] < training['reconstruction_epochs']:
        raise ValueError('Warmup must leave at least one epoch for encoder fine-tuning')
    if training['batch_size'] < 1 or training['workers'] < 0 or dataset['workers'] < 1:
        raise ValueError('Invalid batch size or worker count')
    if not 0 <= training['identity_probability'] <= 1:
        raise ValueError('identity_probability must be between 0 and 1')
    if training['patience'] < 1 or training['perceptual_ramp_epochs'] < 1:
        raise ValueError('Patience and perceptual ramp must be positive')
    if training['preview_every'] < 1 or training['preview_samples'] < 0:
        raise ValueError('Invalid preview settings')
    for key in ('decoder_lr', 'encoder_lr', 'refinement_lr'):
        if training[key] <= 0:
            raise ValueError(f'{key} must be positive')
    for key in ('ssim_weight', 'perceptual_weight', 'weight_decay', 'min_delta'):
        if training[key] < 0:
            raise ValueError(f'{key} must be nonnegative')
    return config
