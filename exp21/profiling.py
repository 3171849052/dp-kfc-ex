"""Exp20 asynchronous CUDA events; synchronization only at phase boundaries."""
from exp20.profiling import EventProfiler, timed, timestamp, phase_start, phase_end

PHASES = ('first_pass_seconds', 'norm_seconds', 'second_pass_seconds',
          'bk_reconstruction_seconds', 'aggregate_transform_seconds', 'noise_step_seconds')
