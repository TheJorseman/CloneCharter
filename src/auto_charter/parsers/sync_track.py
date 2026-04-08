"""BPM map and beat grid construction from [SyncTrack] events.

Handles:
- Multiple BPM changes (up to 200+ in a single song like Caos La Planta)
- Time signature changes (3/4, 6/8, 7/4, etc.)
- Conversion between ticks and wall-clock seconds
- Beat onset tick list aligned to quarter-note boundaries
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple


class BPMEvent(NamedTuple):
    tick: int
    bpm: float  # BPM as float (e.g., 130.0)


class TimeSigEvent(NamedTuple):
    tick: int
    numerator: int
    denominator: int  # always a power of 2; default 4 if omitted in .chart


@dataclass
class BPMMap:
    """Timing map built from a song's [SyncTrack] section.

    Provides tick↔seconds conversion and beat grid generation.
    All beat boundaries are at quarter-note (resolution ticks) positions,
    regardless of time signature (time sig affects measure structure only).
    """

    resolution: int  # ticks per quarter note (192 for .chart, normalised from MIDI)
    bpm_events: list[BPMEvent] = field(default_factory=list)
    time_sig_events: list[TimeSigEvent] = field(default_factory=list)

    # Cached: list of (start_tick, start_time_s, bpm) segments
    _segments: list[tuple[int, float, float]] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self._build_segments()

    def _build_segments(self) -> None:
        """Build piecewise-linear tick→time segments from BPM events."""
        if not self.bpm_events:
            return
        events = sorted(self.bpm_events, key=lambda e: e.tick)
        self._segments = []
        current_time = 0.0
        for i, ev in enumerate(events):
            self._segments.append((ev.tick, current_time, ev.bpm))
            if i + 1 < len(events):
                next_tick = events[i + 1].tick
                delta_ticks = next_tick - ev.tick
                delta_time = delta_ticks / self.resolution * (60.0 / ev.bpm)
                current_time += delta_time

    def tick_to_seconds(self, tick: int) -> float:
        """Convert a tick position to wall-clock seconds."""
        if not self._segments:
            return 0.0
        # Find the last segment whose start_tick <= tick
        seg = self._segments[0]
        for s in self._segments:
            if s[0] <= tick:
                seg = s
            else:
                break
        seg_tick, seg_time, bpm = seg
        delta_ticks = tick - seg_tick
        return seg_time + delta_ticks / self.resolution * (60.0 / bpm)

    def seconds_to_tick(self, seconds: float) -> int:
        """Convert wall-clock seconds to the nearest tick."""
        if not self._segments:
            return 0
        seg = self._segments[0]
        for s in self._segments:
            if s[1] <= seconds:
                seg = s
            else:
                break
        seg_tick, seg_time, bpm = seg
        delta_time = seconds - seg_time
        return seg_tick + round(delta_time * bpm / 60.0 * self.resolution)

    def bpm_at_tick(self, tick: int) -> float:
        """Return the BPM active at a given tick."""
        if not self._segments:
            return 120.0
        bpm = self._segments[0][2]
        for seg_tick, _, seg_bpm in self._segments:
            if seg_tick <= tick:
                bpm = seg_bpm
            else:
                break
        return bpm

    def time_sig_at_tick(self, tick: int) -> tuple[int, int]:
        """Return (numerator, denominator) of the time signature active at tick."""
        num, den = 4, 4
        for ev in sorted(self.time_sig_events, key=lambda e: e.tick):
            if ev.tick <= tick:
                num, den = ev.numerator, ev.denominator
            else:
                break
        return num, den

    def build_beat_grid(self, end_tick: int) -> list[int]:
        """Return a list of tick positions for every quarter-note beat up to end_tick.

        Beat boundaries are placed at multiples of `resolution` ticks within each
        BPM segment. Quarter-note = resolution ticks, always, regardless of time sig.
        """
        if not self._segments:
            return []

        beats: list[int] = []
        events = sorted(self.bpm_events, key=lambda e: e.tick)

        for i, ev in enumerate(events):
            seg_start = ev.tick
            seg_end = events[i + 1].tick if i + 1 < len(events) else end_tick + self.resolution

            # Walk quarter-note boundaries within this segment
            # Start from the first beat boundary >= seg_start
            if seg_start == 0:
                tick = 0
            else:
                # Resume from last beat emitted
                tick = beats[-1] + self.resolution if beats else 0
                # Snap forward to seg_start if needed
                while tick < seg_start:
                    tick += self.resolution

            while tick < seg_end and tick <= end_tick:
                beats.append(tick)
                tick += self.resolution

        return sorted(set(beats))

    def beat_times(self, end_tick: int) -> tuple[list[int], list[float], list[float], list[float], list[tuple[int, int]]]:
        """Return parallel lists for beat grid.

        Returns:
            beat_ticks, beat_times_s, beat_durations_s, bpm_at_beat, time_sig_at_beat
        """
        ticks = self.build_beat_grid(end_tick)
        times_s = [self.tick_to_seconds(t) for t in ticks]
        durations_s = []
        for i, t in enumerate(times_s):
            if i + 1 < len(times_s):
                durations_s.append(times_s[i + 1] - t)
            else:
                # Last beat: use current BPM to estimate duration
                durations_s.append(60.0 / self.bpm_at_tick(ticks[i]))
        bpms = [self.bpm_at_tick(t) for t in ticks]
        sigs = [self.time_sig_at_tick(t) for t in ticks]
        return ticks, times_s, durations_s, bpms, sigs
