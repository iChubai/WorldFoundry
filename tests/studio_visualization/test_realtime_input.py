from __future__ import annotations

import asyncio
import json
from typing import Any, cast

import numpy as np
import pytest

from worldfoundry.studio.serving.realtime.input import ControlSegment, RealtimeControlResampler
from worldfoundry.studio.visualization.backends.world_realtime import (
    LatestFrameBuffer,
    RealtimePeerManager,
    _ActivePeer,
    _ActiveSocket,
)


def _edge(resampler: RealtimeControlResampler, stamp: float, event: str, key: str) -> None:
    assert resampler.on_edge(arrival_time=stamp, event=event, key=key)


def _states(segments: list[ControlSegment]) -> list[frozenset[str]]:
    return [keys for start, end, keys in segments if end > start]


def test_realtime_input_retains_tap_during_inference_after_state_queries() -> None:
    resampler = RealtimeControlResampler(fps=20, start_time=100)
    _edge(resampler, 100, "keydown", "w")
    assert _states(resampler.sample_chunk(4, wall_time=100)) == [frozenset({"w"})]
    _edge(resampler, 100.02, "keyup", "w")
    _edge(resampler, 100.05, "keydown", "d")
    _edge(resampler, 100.10, "keyup", "d")
    for _ in range(3):
        assert resampler.effective_keys == frozenset()
        assert resampler.has_pending_input
    segments = resampler.sample_chunk(4, wall_time=100.25)
    assert _states(segments) == [frozenset(), frozenset({"d"}), frozenset()]
    assert segments[1][1] - segments[1][0] == pytest.approx(0.05)
    assert not resampler.has_pending_input
    assert _states(resampler.sample_chunk(4, wall_time=100.5)) == [frozenset()]


def test_realtime_input_releases_consumed_hold_without_replaying_it() -> None:
    resampler = RealtimeControlResampler(fps=20, start_time=100)
    _edge(resampler, 100, "keydown", "w")
    resampler.sample_chunk(4, wall_time=100)
    _edge(resampler, 100.01, "keydown", "w")  # Browser key repeat is not a new tap.
    _edge(resampler, 100.02, "keyup", "w")
    assert _states(resampler.sample_chunk(4, wall_time=101)) == [frozenset()]
    assert not resampler.has_pending_input


@pytest.mark.parametrize("chunk_frames", [1, 4])
def test_realtime_input_preserves_repeated_and_mixed_taps_across_chunks(chunk_frames: int) -> None:
    resampler = RealtimeControlResampler(fps=20, start_time=100)
    for stamp, event, key in (
        (100.01, "keydown", "d"),
        (100.02, "keyup", "d"),
        (100.03, "keydown", "d"),
        (100.04, "keyup", "d"),
        (100.05, "keydown", "a"),
        (100.06, "keydown", "w"),
        (100.07, "keyup", "a"),
        (100.08, "keyup", "w"),
    ):
        _edge(resampler, stamp, event, key)
    states: list[frozenset[str]] = []
    for _ in range(16):
        for keys in _states(resampler.sample_chunk(chunk_frames, wall_time=101)):
            if not states or states[-1] != keys:
                states.append(keys)
        if not resampler.has_pending_input:
            break
    assert states == [
        frozenset({"d"}),
        frozenset(),
        frozenset({"d"}),
        frozenset(),
        frozenset({"a"}),
        frozenset({"a", "w"}),
        frozenset({"w"}),
        frozenset(),
    ]
    assert not resampler.has_pending_input


def test_realtime_input_retains_chunk_tail_tap_until_a_complete_sample_fits() -> None:
    resampler = RealtimeControlResampler(fps=20, start_time=100)
    _edge(resampler, 100.19, "keydown", "d")
    _edge(resampler, 100.191, "keyup", "d")
    assert _states(resampler.sample_chunk(4, wall_time=100)) == [frozenset()]
    assert resampler.has_pending_input
    segments = resampler.sample_chunk(4, wall_time=100.25)
    assert _states(segments) == [frozenset({"d"}), frozenset()]
    assert segments[0][1] - segments[0][0] == pytest.approx(0.05)
    assert not resampler.has_pending_input


def test_realtime_input_new_edges_follow_retained_backlog() -> None:
    resampler = RealtimeControlResampler(fps=20, start_time=100)
    _edge(resampler, 100.01, "keydown", "d")
    _edge(resampler, 100.011, "keyup", "d")
    assert _states(resampler.sample_chunk(1, wall_time=101)) == [frozenset({"d"})]
    _edge(resampler, 100.02, "keydown", "a")
    _edge(resampler, 100.021, "keyup", "a")
    segments = resampler.sample_chunk(4, wall_time=101.06)
    assert _states(segments) == [frozenset(), frozenset({"a"}), frozenset()]
    assert not resampler.has_pending_input


def test_realtime_input_preserves_long_duration_and_last_pressed_conflicts() -> None:
    resampler = RealtimeControlResampler(fps=20, start_time=100)
    for stamp, event, key in (
        (100.0, "keydown", "w"),
        (100.1, "keydown", "s"),
        (100.3, "keyup", "s"),
        (100.4, "keyup", "w"),
    ):
        _edge(resampler, stamp, event, key)
    segments = resampler.sample_chunk(16, wall_time=101)
    assert _states(segments) == [frozenset({"w"}), frozenset({"s"}), frozenset({"w"}), frozenset()]
    assert [end - start for start, end, _ in segments[:3]] == pytest.approx([0.1, 0.2, 0.1])


def test_realtime_input_focus_loss_releases_chord_and_reset_discards_backlog() -> None:
    resampler = RealtimeControlResampler(fps=20, start_time=100)
    for key in ("w", "ArrowRight"):
        _edge(resampler, 100, "keydown", key)
    assert _states(resampler.sample_chunk(4, wall_time=100)) == [frozenset({"w", "l"})]
    for key in ("w", "ArrowRight"):
        _edge(resampler, 100.01, "keyup", key)
    assert _states(resampler.sample_chunk(4, wall_time=100.25)) == [frozenset()]
    _edge(resampler, 100.3, "keydown", "d")
    _edge(resampler, 100.31, "keyup", "d")
    resampler.reset(start_time=200)
    assert not resampler.has_pending_input
    assert resampler.effective_keys == frozenset()
    assert _states(resampler.sample_chunk(4, wall_time=200)) == [frozenset()]
    _edge(resampler, 200.21, "keydown", "a")
    assert frozenset({"a"}) in _states(resampler.sample_chunk(4, wall_time=200.25))


@pytest.mark.parametrize("transport", ["socket", "webrtc"])
@pytest.mark.parametrize("boundary", [None, "step", "prompt", "prompt_scheduled"])
def test_realtime_input_workers_keep_inflight_tap_and_isolate_step(
    transport: str,
    boundary: str | None,
) -> None:
    class Sink:
        readyState = "open"
        closed = False

        def __init__(self) -> None:
            self.messages: list[dict[str, Any]] = []

        def send(self, raw: str) -> None:
            self.messages.append(json.loads(raw))

        async def send_str(self, raw: str) -> None:
            self.send(raw)

    async def exercise() -> None:
        started = asyncio.Event()
        release = asyncio.Event()
        calls: list[tuple[list[str], Any, str | None]] = []

        class Runtime:
            last_generation_metrics: dict[str, float] = {}

            def next_chunk_frames(self, default: int) -> int:
                return default

            async def generate(
                self,
                interactions: list[str],
                *,
                control_segments: Any = None,
                prompt: str | None = None,
                **kwargs: Any,
            ) -> tuple[list[np.ndarray], float]:
                calls.append((list(interactions), control_segments, prompt))
                if len(calls) == 1:
                    started.set()
                    await release.wait()
                return [np.zeros((4, 6, 3), dtype=np.uint8)], 1.0

        sink = Sink()
        now = asyncio.get_running_loop().time()
        resampler = RealtimeControlResampler(fps=20, start_time=now)
        if transport == "socket":
            active = _ActiveSocket(socket=sink, resampler=resampler, frame_packets=asyncio.Queue(maxsize=8))
        else:
            active = _ActivePeer(peer=None, channel=sink, resampler=resampler, frames=LatestFrameBuffer(maxsize=8))
        peers = RealtimePeerManager(runtime=cast(Any, Runtime()), fps=20, chunk_frames=4)
        if boundary == "prompt_scheduled":
            active.prompt_scheduled = True
            active.initial_segment_pending = True
        _edge(resampler, now, "keydown", "w")
        active.first_action.set()
        if isinstance(active, _ActiveSocket):
            worker = asyncio.create_task(peers._socket_generation_worker(active))
        else:
            worker = asyncio.create_task(peers._generation_worker(active))
        try:
            await asyncio.wait_for(started.wait(), timeout=2)
            _edge(resampler, now + 0.02, "keyup", "w")
            _edge(resampler, now + 0.05, "keydown", "d")
            _edge(resampler, now + 0.10, "keyup", "d")
            if boundary in {"step", "prompt_scheduled"}:
                active.pending_steps = 1
            elif boundary == "prompt":
                active.pending_prompt = "Turn right"
                active.pending_prompt_dirty = True
            assert resampler.effective_keys == frozenset()
            release.set()

            async def finished() -> None:
                expected = 3 if boundary == "step" else 2
                while len(calls) < expected or active.first_action.is_set():
                    if worker.done():
                        await worker
                        pytest.fail(f"worker stopped: {sink.messages}")
                    await asyncio.sleep(0.001)

            await asyncio.wait_for(finished(), timeout=2)
            if boundary == "prompt_scheduled":
                assert calls == [([], None, None), ([], None, None)]
                assert resampler.has_pending_input
                return
            assert calls[0][0] == ["forward"]
            if boundary == "step":
                assert calls[1] == ([], None, None)
            assert calls[-1][0] == ["right"]
            assert frozenset({"d"}) in _states(calls[-1][1])
            assert calls[-1][2] == ("Turn right" if boundary == "prompt" else None)
            assert not resampler.has_pending_input
            assert not any(message["type"] == "error" for message in sink.messages)
        finally:
            active.closed = True
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)

    asyncio.run(exercise())
