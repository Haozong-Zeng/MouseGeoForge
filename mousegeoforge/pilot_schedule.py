"""Explicit update-indexed LR schedules for controlled spatial pilots."""
import math


def learning_rates(config):
    steps = config['steps']
    base = config['learning_rate']
    spec = config.get('lr_schedule', {'type': 'constant'})
    if not isinstance(steps, int) or not 1 <= steps <= 15000:
        raise ValueError('Pilot budget must be an integer in 1..15000')
    if not math.isfinite(base) or base <= 0:
        raise ValueError('Learning rate must be finite and positive')
    if spec == {'type': 'constant'}:
        return [base] * steps
    if set(spec) != {'type', 'hold_steps', 'min_learning_rate'} or spec['type'] != 'hold_cosine':
        raise ValueError('Unknown LR schedule or fields')
    hold, minimum = spec['hold_steps'], spec['min_learning_rate']
    if not isinstance(hold, int) or not 0 <= hold < steps:
        raise ValueError('hold_steps must be an integer smaller than total steps')
    if not math.isfinite(minimum) or not 0 < minimum <= base:
        raise ValueError('min_learning_rate must be in (0, learning_rate]')
    return [base if step <= hold else minimum + (base-minimum)*.5*
            (1+math.cos(math.pi*(step-hold)/(steps-hold)))
            for step in range(1, steps+1)]
