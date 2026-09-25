"""Shared text progress and wall-clock estimates for training entry points."""

from __future__ import annotations

import sys
import time

from tqdm import tqdm


def batch_progress(iterable, *, desc: str, enabled: bool = True):
    # Explicit enablement also works in Colab's non-TTY !python subprocesses.
    return tqdm(
        iterable, desc=desc, unit="batch", disable=not enabled, file=sys.stdout,
        mininterval=1.0, miniters=1, dynamic_ncols=True, ascii=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}, ETA {remaining}, {rate_fmt}{postfix}]",
    )


class EpochTimer:
    """Estimate the remaining configured epochs, including validation/save time."""

    def __init__(self, total_epochs: int, start_epoch: int = 1):
        self.total_epochs = total_epochs
        self.start_epoch = start_epoch
        self.started = self.epoch_started = time.perf_counter()

    def finish_epoch(self, epoch: int) -> dict[str, float]:
        now = time.perf_counter()
        elapsed = now - self.started
        completed = epoch - self.start_epoch + 1
        result = {
            "epoch_seconds": now - self.epoch_started,
            "run_elapsed_seconds": elapsed,
            "estimated_remaining_seconds": elapsed / completed * max(0, self.total_epochs - epoch),
        }
        self.epoch_started = now
        return result


def timing_summary(timing: dict[str, float]) -> str:
    def duration(key):
        return tqdm.format_interval(max(0, timing[key]))

    return (
        f"epoch_time={duration('epoch_seconds')} "
        f"elapsed={duration('run_elapsed_seconds')} "
        f"ETA(max_epochs)={duration('estimated_remaining_seconds')}"
    )
