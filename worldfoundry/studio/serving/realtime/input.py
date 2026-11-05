"""Timestamped control input primitives for realtime world-model serving."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace

SUPPORTED_CONTROL_KEYS = frozenset({"w", "a", "s", "d", "i", "j", "k", "l"})
KEY_ALIASES = {
    "arrowup": "i",
    "arrowdown": "k",
    "arrowleft": "j",
    "arrowright": "l",
}


def normalize_control_key(key: str) -> str:
    normalized = str(key or "").strip().lower()
    return KEY_ALIASES.get(normalized, normalized)


@dataclass(slots=True)
class RealtimeControlState:
    """Pressed-key state with last-pressed-wins conflict resolution."""

    pressed: set[str] = field(default_factory=set)
    _order: dict[str, int] = field(default_factory=dict)
    _sequence: int = 0

    def apply(self, event: str, key: str) -> bool:
        normalized = normalize_control_key(key)
        if normalized not in SUPPORTED_CONTROL_KEYS:
            return False
        event = str(event or "").strip().lower()
        if event == "keydown":
            self.pressed.add(normalized)
            self._sequence += 1
            self._order[normalized] = self._sequence
            return True
        if event == "keyup":
            self.pressed.discard(normalized)
            self._order.pop(normalized, None)
            return True
        return False

    def _latest(self, keys: tuple[str, ...]) -> str | None:
        active = [key for key in keys if key in self.pressed]
        return max(active, key=lambda key: self._order.get(key, -1), default=None)

    def effective(self) -> frozenset[str]:
        return frozenset(
            key
            for key in (
                self._latest(("w", "s")),
                self._latest(("a", "d")),
                self._latest(("i", "k")),
                self._latest(("j", "l")),
            )
            if key is not None
        )


ControlSegment = tuple[float, float, frozenset[str]]


class RealtimeControlResampler:
    """Resample timestamped input edges into a model chunk timeline."""

    def __init__(self, *, fps: float, start_time: float = 0.0) -> None:
        if fps <= 0:
            raise ValueError("fps must be > 0")
        self.fps = float(fps)
        self.dt = 1.0 / self.fps
        self.next_chunk_start = float(start_time)
        self._events: deque[tuple[float, str, str]] = deque()
        self._state = RealtimeControlState()
        self._live_state = RealtimeControlState()
        self._event_delay = 0.0
        self._minimum_state_end = float(start_time)

    def on_edge(self, *, arrival_time: float, event: str, key: str) -> bool:
        normalized = normalize_control_key(key)
        if normalized not in SUPPORTED_CONTROL_KEYS or event not in {"keydown", "keyup"}:
            return False
        self._live_state.apply(event, normalized)
        self._events.append((float(arrival_time) + self._event_delay, event, normalized))
        return True

    def _delay_events(self, delay: float) -> None:
        self._events = deque((stamp + delay, event, key) for stamp, event, key in self._events)
        self._event_delay += delay

    def sample_chunk(self, num_frames: int, *, wall_time: float) -> list[ControlSegment]:
        """Consume controls, retaining each new state for a native sampling interval.

        Late releases of consumed holds apply at the chunk boundary. Unsampled
        taps move onto the next available timeline, together with subsequent
        edges, so their order and longer durations survive inference latency.
        """
        if num_frames < 1:
            raise ValueError("num_frames must be >= 1")
        duration = num_frames * self.dt
        interval = self.dt
        if self.next_chunk_start <= 0.0 or wall_time - self.next_chunk_start > duration:
            self.next_chunk_start = float(wall_time)
        start = self.next_chunk_start
        end = start + duration
        segments: list[ControlSegment] = []
        cursor = start
        effective = self._state.effective()
        while self._events:
            stamp = self._events[0][0]
            candidate = replace(self._state, pressed=set(self._state.pressed), _order=dict(self._state._order))
            count = 0
            for event_time, event, key in self._events:
                if event_time != stamp:
                    break
                candidate.apply(event, key)
                count += 1
            next_effective = candidate.effective()
            event_time = max(stamp, start)
            if next_effective != effective:
                event_time = max(event_time, self._minimum_state_end)
                # A new active state needs a complete interval: both frame-end
                # and frame-midpoint consumers must see it before its release.
                if next_effective and event_time + interval > end + 1e-8:
                    event_time = max(event_time, end)
                if event_time > stamp and (next_effective or event_time > start):
                    self._delay_events(event_time - stamp)
            if event_time >= end:
                break
            for _ in range(count):
                self._events.popleft()
            if next_effective != effective:
                if event_time > cursor:
                    segments.append((cursor, event_time, effective))
                cursor = event_time
                self._minimum_state_end = event_time + interval
            self._state = candidate
            effective = next_effective
        if cursor < end:
            segments.append((cursor, end, effective))
        self.next_chunk_start = end
        if not self._events:
            self._event_delay = 0.0
        return segments

    def reset(self, *, start_time: float) -> None:
        self._events.clear()
        self._state = RealtimeControlState()
        self._live_state = RealtimeControlState()
        self._event_delay = 0.0
        self._minimum_state_end = float(start_time)
        self.next_chunk_start = float(start_time)

    @property
    def effective_keys(self) -> frozenset[str]:
        """Current received key state; observing it never consumes model input."""
        return self._live_state.effective()

    @property
    def has_pending_input(self) -> bool:
        """Whether held controls or unsampled edges still need a chunk boundary."""
        return bool(self._events or self.effective_keys)


def interactions_from_keys(keys: frozenset[str]) -> list[str]:
    tokens: list[str] = []
    for key, token in (
        ("w", "forward"),
        ("s", "backward"),
        ("a", "left"),
        ("d", "right"),
        ("i", "camera_up"),
        ("k", "camera_down"),
        ("j", "camera_l"),
        ("l", "camera_r"),
    ):
        if key in keys:
            tokens.append(token)
    return tokens


def interactions_from_segments(segments: list[ControlSegment]) -> list[str]:
    """Return the most recent non-idle control state in a sampled chunk."""

    for _, _, keys in reversed(segments):
        interactions = interactions_from_keys(keys)
        if interactions:
            return interactions
    return []


__all__ = [
    "ControlSegment",
    "KEY_ALIASES",
    "RealtimeControlResampler",
    "RealtimeControlState",
    "SUPPORTED_CONTROL_KEYS",
    "interactions_from_keys",
    "interactions_from_segments",
    "normalize_control_key",
]
