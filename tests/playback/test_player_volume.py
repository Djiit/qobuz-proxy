"""A fast slider drag must not echo stale volumes back to the Qobuz app.

The app sends one SET_VOLUME per slider step and moves its slider to every
volume we report. Each command runs in its own task and waits on a slow SOAP
SetVolume, so reports for older steps used to arrive after newer commands.
"""

import asyncio
from unittest.mock import MagicMock

from qobuz_proxy.backends import BackendTrackMetadata, PlaybackState
from qobuz_proxy.backends.base import AudioBackend
from qobuz_proxy.playback.player import QobuzPlayer
from qobuz_proxy.playback.volume_handler import VolumeCommandHandler
from qobuz_proxy.proto import qconnect_payload_pb2 as pb

# qp5.log 16:38:38.739-16:38:40.887: SET_VOLUME arrivals ("recv") and SetVolume
# round trips finishing ("done"), in the order the proxy logged them.
RECORDED_DRAG: list[tuple[str, int]] = [
    ("recv", 20),
    ("recv", 51),
    ("done", 0),
    ("recv", 22),
    ("recv", 51),
    ("done", 0),
    ("recv", 22),
    ("done", 0),
    ("recv", 49),
    ("done", 0),
    ("recv", 14),
    ("done", 0),
    ("done", 0),
    ("done", 0),
    ("recv", 45),
    ("done", 0),
]


class _SlowSoapBackend(AudioBackend):
    """Backend whose SetVolume calls stay in flight until the test finishes them."""

    def __init__(self) -> None:
        super().__init__(name="test")
        self.in_flight: list[tuple[int, asyncio.Event]] = []
        self.applied: list[int] = []

    async def set_volume(self, level: int) -> None:
        done = asyncio.Event()
        self.in_flight.append((level, done))
        await done.wait()
        self.applied.append(level)

    async def get_volume(self) -> int:
        return self.applied[-1] if self.applied else 0

    def finish(self, newest_first: bool = False) -> None:
        _, done = self.in_flight.pop(-1 if newest_first else 0)
        done.set()

    async def play(self, url: str, metadata: BackendTrackMetadata) -> None: ...
    async def pause(self) -> None: ...
    async def resume(self) -> bool:
        return True

    async def stop(self, *, next_track_id: str | None = None) -> None: ...
    async def seek(self, position_ms: int) -> None: ...
    async def get_position(self) -> int:
        return 0

    async def get_state(self) -> PlaybackState:
        return self._state

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None: ...


def _make_rig():
    backend = _SlowSoapBackend()
    player = QobuzPlayer(queue=MagicMock(), metadata_service=MagicMock(), backend=backend)
    handler = VolumeCommandHandler(player)
    received: list[int] = []
    # Each report with the latest volume the app had sent at that moment.
    reports: list[tuple[int, int]] = []

    async def report(volume: int) -> None:
        reports.append((volume, received[-1] if received else volume))

    player.set_volume_report_callback(report)  # type: ignore[arg-type]
    tasks: list[asyncio.Task[None]] = []

    def deliver(volume: int) -> None:
        # Same dispatch as Speaker: one task per SET_VOLUME message.
        message = pb.QConnectMessage(messageType=42)
        message.srvrRndrSetVolume.volume = volume
        received.append(volume)
        tasks.append(asyncio.create_task(handler.handle_message(42, message)))

    return player, backend, deliver, reports, tasks


async def _settle() -> None:
    for _ in range(20):
        await asyncio.sleep(0)


async def _drain(backend: _SlowSoapBackend, tasks: list, newest_first: bool = False) -> None:
    await _settle()
    while backend.in_flight:
        backend.finish(newest_first)
        await _settle()
    await asyncio.gather(*tasks)


async def test_recorded_drag_never_reports_an_older_volume():
    _, backend, deliver, reports, tasks = _make_rig()

    for event, volume in RECORDED_DRAG:
        if event == "recv":
            deliver(volume)
        elif backend.in_flight:
            backend.finish()
        await _settle()
    await _drain(backend, tasks)

    assert [r for r in reports if r[0] != r[1]] == []
    assert reports[-1] == (45, 45)
    assert backend.applied[-1] == 45


async def test_final_volume_is_applied_last_when_older_call_finishes_late():
    _, backend, deliver, reports, tasks = _make_rig()

    deliver(22)
    await _settle()
    deliver(49)
    await _drain(backend, tasks, newest_first=True)

    assert backend.applied[-1] == 49
    assert reports == [(49, 49)]


async def test_rebroadcast_during_drag_does_not_report_an_older_volume():
    player, backend, deliver, reports, tasks = _make_rig()

    deliver(20)
    await _settle()
    # SET_ACTIVE re-broadcast lands mid-drag, before 20 reached the speaker.
    tasks.append(asyncio.create_task(player.broadcast_current_volume()))
    await _settle()
    deliver(51)
    await _drain(backend, tasks)

    assert [r for r in reports if r[0] != r[1]] == []
    assert reports[-1] == (51, 51)


async def test_single_volume_change_is_applied_and_reported():
    _, backend, deliver, reports, tasks = _make_rig()

    deliver(30)
    await _drain(backend, tasks)

    assert backend.applied == [30]
    assert reports == [(30, 30)]
