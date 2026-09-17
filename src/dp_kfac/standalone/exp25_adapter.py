"""Configuration and output adapter for the shared Exp25 training loop."""
from copy import deepcopy
from types import SimpleNamespace
import torch
from opacus.accountants.utils import get_noise_multiplier
from exp25 import config as original
from exp25.run_one import run
from .run_logging import METRICS_FIELDS, MetricsCSVWriter, rng_seeds, write_summary, write_yaml


def protocol(c):
    t, d, p = c['training'], c['data'], c['privacy']
    steps = original.TRAIN_SAMPLES // d['batch_size']
    if steps == 0:
        raise ValueError('batch_size exceeds training dataset size')
    rate = d['batch_size'] / original.TRAIN_SAMPLES
    return SimpleNamespace(
        condition=original.condition, EPOCHS=t['epochs'], BATCH_SIZE=d['batch_size'],
        TRAIN_SAMPLES=original.TRAIN_SAMPLES, STEPS_PER_EPOCH=steps,
        TOTAL_STEPS=steps*t['epochs'], SAMPLE_RATE=rate,
        EPSILON=p['epsilon'], DELTA=p['delta'], CLIP=p['max_grad_norm'],
        LR=t['learning_rate'], DAMPING=c['kfac']['damping'], A_POWER=c['kfac']['a_power'],
        noise_multiplier=lambda: get_noise_multiplier(target_epsilon=p['epsilon'],
            target_delta=p['delta'], sample_rate=rate, steps=steps*t['epochs'],
            accountant=p['accountant'], epsilon_tolerance=1e-4))


def train_exp25(c, directory, smoke=False):
    settings = protocol(c)
    writer = MetricsCSVWriter(directory / 'metrics.csv')
    rows = []

    def resolved(sigma, device, train_size, test_size):
        value = deepcopy(c)
        value.update(train_size=train_size, test_size=test_size,
            steps_per_epoch=settings.STEPS_PER_EPOCH, total_steps=settings.TOTAL_STEPS,
            sample_rate=settings.SAMPLE_RATE, noise_multiplier=sigma, smoke=smoke,
            executed_epochs=1 if smoke else settings.EPOCHS,
            actual_device=str(device), gpu_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
            physical_gpu_index=c['runtime']['gpu'] if device.type == 'cuda' else None,
            run_directory=str(directory), rng_seeds=rng_seeds(c),
            gradient_engine='bk_gd', geometry=settings.condition(c['algorithm'])[0],
            sampling='fixed_size_shuffle_drop_last; nominal_Opacus_RDP')
        write_yaml(directory / 'resolved_config.yaml', value)

    def epoch(row):
        metrics = dict.fromkeys(METRICS_FIELDS)
        for name in ('epoch', 'train_loss', 'test_loss', 'test_accuracy', 'epsilon_spent',
                     'noise_multiplier', 'sample_rate', 'clip_fraction', 'mean_clip_factor'):
            metrics[name] = row[name]
        metrics.update(global_step=row['accountant_steps'],
            precond_build_seconds=row['geometry_build_time'],
            epoch_train_seconds=row['private_training_time'],
            epoch_total_seconds=row['epoch_total_seconds'],
            peak_allocated_mb_epoch=row['cuda_peak_memory']/2**20)
        writer.append(metrics)
        rows.append(row)

    run(c['algorithm'], c['seed'], smoke, c['runtime']['device'],
        protocol=settings, on_resolved=resolved, on_epoch=epoch,
        threads=c['runtime']['threads'], data_root=c['data']['root'])
    last = rows[-1]
    summary = dict(status='completed', algorithm=c['algorithm'], optimizer='adam',
        seed=c['seed'], smoke=smoke, completed_epochs=len(rows),
        global_step=last['accountant_steps'], final_test_loss=last['test_loss'],
        final_test_accuracy=last['test_accuracy'], best_test_accuracy=last['best_accuracy'],
        final_epsilon=last['epsilon_spent'], noise_multiplier=last['noise_multiplier'],
        epoch_diagnostics=rows)
    write_summary(directory, summary)
    return summary
