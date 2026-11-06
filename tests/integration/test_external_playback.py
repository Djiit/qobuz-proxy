"""A Qobuz reconnect must leave another controller's audio and queue untouched."""

import asyncio
import json
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest
import websockets
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from qobuz_proxy.backends.dlna.backend import DLNABackend, _source_label
from qobuz_proxy.backends.dlna.client import DLNAClient
from qobuz_proxy.backends.types import BackendTrackMetadata, PlaybackState
from qobuz_proxy.config import Config, SpeakerConfig
from qobuz_proxy.connect.discovery import DiscoveryService
from qobuz_proxy.connect.protocol import DecodedMessage, MessageType
from qobuz_proxy.connect.ws_manager import WsManager
from qobuz_proxy.playback.command_handler import PlaybackCommandHandler
from qobuz_proxy.playback.player import QobuzPlayer
from qobuz_proxy.playback.queue import QueueTrack
from qobuz_proxy.playback.volume_handler import VolumeCommandHandler
from qobuz_proxy.proto import qconnect_payload_pb2 as pb
from qobuz_proxy.speaker import Speaker
from tests.connect.test_ws_manager import _last_join, _send_report
from tests.connect.test_ws_manager import valid_tokens as valid_tokens


QOBUZ_URI = "http://proxy:7120/audio/123.flac"
NEXT_URI = "http://proxy:7120/audio/456_2.flac"
SPOTIFY_URI = "x-sonos-spotify:spotify%3atrack%3aexample"


@pytest.fixture
def speaker_rig(valid_tokens):
    backend = DLNABackend("192.0.2.1", name="Kitchen")
    backend._is_sonos = True
    backend._current_proxy_url = QOBUZ_URI
    backend._next_track_proxy_url = NEXT_URI
    backend._next_track_queue_nr = 2
    backend._state = PlaybackState.PAUSED
    client = AsyncMock(spec=DLNAClient)
    client.get_track_uri.return_value = SPOTIFY_URI
    client.get_transport_info.return_value = "PLAYING"
    client.get_position_info.return_value = 60000
    backend._client = client

    metadata = MagicMock()
    metadata.get_streaming_url = AsyncMock(return_value=QOBUZ_URI)
    metadata.get_metadata = AsyncMock(return_value=None)
    metadata.get_track_format.return_value = (6, 44100, 16)
    queue = MagicMock()
    queue.set_current_by_item_id = AsyncMock()
    player = QobuzPlayer(queue, metadata, backend)
    player._current_track = QueueTrack(
        track_id="123", queue_item_id=1, streaming_url=QOBUZ_URI, duration_ms=116000
    )
    player._current_duration_ms = 116000
    player._state = PlaybackState.PAUSED
    player._set_position(4000)
    manager = WsManager(Config())
    manager.set_tokens(valid_tokens, activate=True)
    manager._session_uuid = b"s" * 16
    manager._ws = AsyncMock()
    manager._is_connected = True
    manager._renderer_active = True
    manager._active_confirmed = True
    handler = PlaybackCommandHandler(player)
    handler._note_active(True)
    manager.on_disconnected(handler.note_disconnected)
    tasks = []

    def dispatch(kind, message):
        tasks.append(handler.dispatch_message(kind, message))

    for kind in handler.get_message_types():
        manager.register_handler(kind, dispatch)

    volume_handler = VolumeCommandHandler(player)

    def dispatch_volume(kind, message):
        tasks.append(asyncio.create_task(volume_handler.handle_message(kind, message)))

    for kind in volume_handler.get_message_types():
        manager.register_handler(kind, dispatch_volume)

    speaker = Speaker(SpeakerConfig(name="Kitchen"), MagicMock(), "test")
    speaker._ws_manager = manager
    speaker._player = player
    speaker._backend = backend
    speaker._queue = queue
    backend.on_external_playback(speaker._on_external_playback)
    player.set_playback_permission_check(backend.can_apply_remote_state)
    player.set_next_track_callbacks(handler.get_next_track_info, handler.clear_next_track_info)
    return speaker, backend, client, player, manager, tasks


@pytest.fixture
def rig(speaker_rig):
    return speaker_rig[1:]


async def deliver_snapshot(manager, tasks, playing_state=3):
    batch = pb.QConnectBatch()
    batch.messages.add(messageType=43).srvrRndrSetActive.active = True
    state = batch.messages.add(messageType=41).srvrRndrSetState
    state.currentQueueItem.trackId = 123
    state.currentQueueItem.queueItemId = 1
    state.currentPosition = 4000
    state.playingState = playing_state
    await manager._handle_payload(
        DecodedMessage(msg_type=MessageType.PAYLOAD, payload=batch.SerializeToString())
    )
    await asyncio.gather(*tasks)
    tasks.clear()


def assert_no_transport_writes(client):
    for method in (
        "seek",
        "play",
        "pause",
        "stop",
        "set_av_transport_uri",
        "clear_queue",
        "add_uri_to_queue",
        "remove_track_from_queue",
        "set_next_av_transport_uri",
        "set_volume",
    ):
        getattr(client, method).assert_not_awaited()


@pytest.mark.parametrize("playing_state", [1, 2, 3])
async def test_reconnect_snapshot_releases_spotify_without_transport_writes(rig, playing_state):
    backend, client, player, manager, tasks = rig
    await deliver_snapshot(manager, tasks, playing_state)
    assert_no_transport_writes(client)
    assert player.state == PlaybackState.STOPPED
    assert backend._next_track_queue_nr is None
    assert not manager.is_renderer_active
    assert await _send_report(manager) is False

    # Even a server that replays active=True cannot reacquire an external source.
    await manager._send_join_session()
    assert _last_join(manager).isActive is False
    assert _last_join(manager).reason == 2
    manager._is_connected = True  # Simulate a cached activation on another connection.
    await deliver_snapshot(manager, tasks, 2)
    assert_no_transport_writes(client)
    assert not manager.is_renderer_active


@pytest.mark.parametrize("already_detected", [True, False])
async def test_explicit_reselection_can_start_same_qobuz_track(rig, valid_tokens, already_detected):
    backend, client, player, manager, tasks = rig
    if already_detected:
        await deliver_snapshot(manager, tasks)
    speaker = Speaker(config=MagicMock(), api_client=MagicMock(), app_id="test")
    speaker._backend = backend
    speaker._player = player
    speaker._queue = player.queue
    speaker._ws_manager = manager
    manager._connection_loop = AsyncMock()
    await speaker._setup_websocket(valid_tokens)
    await manager._send_join_session()
    manager._is_connected = True
    assert _last_join(manager).isActive is True
    assert _last_join(manager).reason == 1

    async def start_queue(index):
        client.get_track_uri.return_value = QOBUZ_URI
        return True

    client.play_from_queue.side_effect = start_queue
    await deliver_snapshot(manager, tasks, 2)
    client.play_from_queue.assert_awaited_once_with(0)
    client.seek.assert_awaited_once_with(4000)
    assert player.state == PlaybackState.PLAYING
    assert manager.is_renderer_active


async def test_takeover_leaves_cloud_and_http_reselection_recovers_playback_and_volume(
    speaker_rig, valid_tokens
):
    speaker, backend, client, player, manager, tasks = speaker_rig
    connections = asyncio.Queue()
    connected = asyncio.Event()
    original_connected = manager._on_connected

    def note_connected():
        if original_connected:
            original_connected()
        connected.set()

    manager.on_connected(note_connected)
    manager.set_token_refresher(AsyncMock())

    async def accept(ws):
        # Exercise the actual WebSocket lifetime, including context-manager
        # cleanup on cancellation, instead of only changing connection flags.
        for _ in range(3):  # AUTHENTICATE, SUBSCRIBE, JOIN_SESSION
            frame = await ws.recv()
        decoded = manager._codec.decode_frame(frame)
        batch = manager._codec.decode_qconnect_batch(decoded.payload)
        await connections.put((ws, batch.messages[0].rndrSrvrJoinSession))
        await ws.wait_closed()

    http_app = web.Application()
    discovery = DiscoveryService(
        Config(), "test", on_connect=speaker._on_app_connected, web_app=http_app
    )
    speaker._discovery = discovery
    await discovery._start_http_server()
    async with websockets.serve(accept, "127.0.0.1", 0) as server:
        valid_tokens.ws_token.endpoint = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"
        async with TestClient(TestServer(http_app)) as http:
            try:
                manager.set_tokens(valid_tokens, activate=True)
                discovery.set_session(valid_tokens)
                await manager.start()
                original_task = manager._receive_task
                await manager.start()
                assert manager._receive_task is original_task
                old_ws, original_join = await asyncio.wait_for(connections.get(), 1)
                await asyncio.wait_for(connected.wait(), 1)
                old_task = manager._receive_task

                assert await backend.check_external_playback()
                response = await http.get("/streamcore/get-connect-info")
                assert (await response.json())["current_session_id"] == ""
                assert discovery.get_received_tokens() is None
                assert not manager.is_connected
                assert not manager.is_renderer_active
                await asyncio.wait_for(old_ws.wait_closed(), 1)
                assert_no_transport_writes(client)

                # Refreshing credentials must never resume the released session.
                manager.set_tokens(valid_tokens)
                await manager.start()
                assert manager._receive_task is old_task
                manager._token_refresher.assert_not_awaited()

                # The app can retry with the same device identity and tokens.
                connected.clear()
                response = await http.post(
                    "/streamcore/connect-to-qconnect",
                    json={
                        "session_id": valid_tokens.session_id,
                        "jwt_qconnect": {
                            "jwt": valid_tokens.ws_token.jwt,
                            "exp": valid_tokens.ws_token.exp,
                            "endpoint": valid_tokens.ws_token.endpoint,
                        },
                    },
                )
                assert response.status == 200
                _, new_join = await asyncio.wait_for(connections.get(), 1)
                await asyncio.wait_for(connected.wait(), 1)
                assert new_join.deviceInfo.deviceUuid == original_join.deviceInfo.deviceUuid
                assert new_join.isActive
                assert new_join.deviceInfo.capabilities.volumeRemoteControl == 2
                assert old_task.done()
                response = await http.get("/streamcore/get-connect-info")
                assert (await response.json())["current_session_id"] == valid_tokens.session_id

                async def start_queue(index):
                    client.get_track_uri.return_value = QOBUZ_URI
                    return True

                client.play_from_queue.side_effect = start_queue
                await deliver_snapshot(manager, tasks, 2)
                assert player.state == PlaybackState.PLAYING
                client.play_from_queue.assert_awaited_once_with(0)
                volume = pb.QConnectBatch()
                volume.messages.add(messageType=42).srvrRndrSetVolume.volume = 28
                await manager._handle_payload(
                    DecodedMessage(msg_type=MessageType.PAYLOAD, payload=volume.SerializeToString())
                )
                await asyncio.gather(*tasks)
                client.set_volume.assert_awaited_once_with(28)
            finally:
                await manager.stop()


async def test_selection_that_detects_takeover_keeps_new_discovery_session(
    speaker_rig, valid_tokens
):
    speaker, _, _, _, manager, _ = speaker_rig
    discovery = DiscoveryService(Config(), "test")
    speaker._discovery = discovery
    discovery.set_session(valid_tokens)
    manager._connection_loop = AsyncMock()

    await speaker._setup_websocket(valid_tokens)

    response = await discovery._handle_connect_info(MagicMock())
    assert json.loads(response.text)["current_session_id"] == valid_tokens.session_id
    assert discovery.get_received_tokens() is valid_tokens
    assert not manager._external_playback
    await manager.stop()


@pytest.mark.parametrize("uri", [QOBUZ_URI, NEXT_URI])
async def test_current_and_gapless_qobuz_tracks_remain_controllable(rig, uri):
    backend, client, _, manager, _ = rig
    client.get_track_uri.return_value = uri
    await backend.seek(4000)
    assert await backend.resume()
    await backend.pause()
    client.seek.assert_awaited_once_with(4000)
    client.play.assert_awaited_once()
    client.pause.assert_awaited_once()
    assert manager.is_renderer_active


@pytest.mark.parametrize("uri", [SPOTIFY_URI, None, ""])
async def test_transport_and_queue_commands_fail_closed_for_foreign_or_unknown_uri(rig, uri):
    backend, client, _, _, _ = rig
    client.get_track_uri.return_value = uri
    with pytest.raises(RuntimeError, match="Cannot seek"):
        await backend.seek(4000)
    await backend.pause()
    assert not await backend.resume()
    metadata = BackendTrackMetadata(track_id="456", title="Next", artist="Artist")
    assert not await backend.set_next_track(NEXT_URI, metadata)
    await backend.clear_next_track()
    await backend.stop()
    await backend.disconnect()
    assert_no_transport_writes(client)
    client.disconnect.assert_awaited_once()
    assert backend._external_playback is (uri == SPOTIFY_URI)


async def test_source_change_during_poll_does_not_advance_qobuz_queue(rig, monkeypatch):
    backend, client, player, manager, _ = rig
    player._state = PlaybackState.PLAYING
    backend._state = PlaybackState.PLAYING
    client.get_transport_info.return_value = "STOPPED"
    ended = MagicMock()
    backend.on_track_ended(ended)
    backend._is_connected = True
    ticks = 0

    async def tick(_):
        nonlocal ticks
        ticks += 1
        if ticks == 2:
            backend._is_connected = False

    monkeypatch.setattr("qobuz_proxy.backends.dlna.backend.asyncio.sleep", tick)
    await backend._poll_state_loop()
    ended.assert_not_called()
    assert_no_transport_writes(client)
    assert player.state == PlaybackState.STOPPED
    assert not manager.is_renderer_active


async def test_standard_dlna_uses_media_uri(rig):
    backend, client, _, _, _ = rig
    backend._is_sonos = False
    client.get_media_info.return_value = SPOTIFY_URI
    assert await backend.check_external_playback()
    client.get_media_info.assert_awaited_once()
    client.get_track_uri.assert_not_awaited()


@pytest.mark.parametrize("uri", [QOBUZ_URI, NEXT_URI])
async def test_standard_dlna_without_media_uri_falls_back_to_track_uri(rig, uri):
    """Bose SoundTouch reports an empty CurrentURI while playing our track; only
    GetPositionInfo.TrackURI identifies it. Without the fallback every command
    failed closed and the speaker could not be paused, stopped or skipped."""
    backend, client, _, manager, _ = rig
    backend._is_sonos = False
    client.get_media_info.return_value = ""
    client.get_track_uri.return_value = uri
    await backend.seek(4000)
    assert await backend.resume()
    await backend.pause()
    await backend.stop()
    client.seek.assert_awaited_once_with(4000)
    client.play.assert_awaited_once()
    client.pause.assert_awaited_once()
    client.stop.assert_awaited_once()
    assert manager.is_renderer_active


@pytest.mark.parametrize("track_uri", [None, ""])
async def test_standard_dlna_without_any_uri_still_fails_closed(rig, track_uri):
    backend, client, _, manager, _ = rig
    backend._is_sonos = False
    client.get_media_info.return_value = ""
    client.get_track_uri.return_value = track_uri
    await backend.pause()
    await backend.stop()
    assert_no_transport_writes(client)
    assert not backend._external_playback
    assert manager.is_renderer_active


async def test_standard_dlna_foreign_track_uri_is_an_external_takeover(rig):
    backend, client, _, manager, _ = rig
    backend._is_sonos = False
    client.get_media_info.return_value = ""
    client.get_track_uri.return_value = SPOTIFY_URI
    assert await backend.check_external_playback()
    assert_no_transport_writes(client)
    assert not manager.is_renderer_active


async def test_deactivation_stops_renderer_without_media_uri(rig):
    """Switching the Qobuz session to another speaker must stop this one."""
    backend, client, player, manager, tasks = rig
    backend._is_sonos = False
    client.get_media_info.return_value = ""
    client.get_track_uri.return_value = QOBUZ_URI
    client.stop.return_value = True
    batch = pb.QConnectBatch()
    batch.messages.add(messageType=43).srvrRndrSetActive.active = False
    await manager._handle_payload(
        DecodedMessage(msg_type=MessageType.PAYLOAD, payload=batch.SerializeToString())
    )
    await asyncio.gather(*tasks)
    client.stop.assert_awaited_once()
    assert player.state == PlaybackState.STOPPED


SOUNDTOUCH_IDLE_URI = "qplay://"
SOUNDTOUCH_IDLE_TRACK_URI = "http://invalidHost.com/invalidUri.mp3"


def _soundtouch(rig_backend, client):
    """Bose SoundTouch: empty CurrentURI while playing; after Stop it drops to a
    standby placeholder (CurrentURI qplay://, PAUSED_PLAYBACK)."""
    rig_backend._is_sonos = False
    rig_backend._next_track_proxy_url = None
    rig_backend._playback_started_at = 0
    client.get_media_info.return_value = ""
    client.get_track_uri.return_value = QOBUZ_URI
    client.get_transport_info.return_value = "PLAYING"

    async def stop():
        client.get_media_info.return_value = SOUNDTOUCH_IDLE_URI
        client.get_track_uri.return_value = SOUNDTOUCH_IDLE_TRACK_URI
        client.get_transport_info.return_value = "PAUSED_PLAYBACK"
        return True

    client.stop.side_effect = stop


async def test_skip_on_renderer_idling_after_our_stop_keeps_the_session(rig):
    """A skip is Stop + Play. The SoundTouch shows its standby placeholder in
    between; that is not another source, so the proxy must play the next track
    instead of leaving the Qobuz session ("device disconnected")."""
    backend, client, _, manager, _ = rig
    _soundtouch(backend, client)
    await backend.stop(next_track_id="456")
    metadata = BackendTrackMetadata(track_id="456", title="Next", artist="Artist")
    await backend.play(NEXT_URI, metadata)
    client.set_av_transport_uri.assert_awaited_once()
    assert backend._current_proxy_url == NEXT_URI
    assert not backend._external_playback
    assert manager.is_renderer_active


async def test_poll_after_our_stop_does_not_release_idle_renderer(rig):
    backend, client, _, manager, _ = rig
    _soundtouch(backend, client)
    await backend.stop()
    assert not await backend.check_external_playback()
    assert manager.is_renderer_active
    # The app may start a new track on the stopped speaker.
    assert await backend.can_apply_remote_state()


async def test_idle_placeholder_after_our_stop_is_still_not_controllable(rig):
    backend, client, _, _, _ = rig
    _soundtouch(backend, client)
    await backend.stop()
    client.pause.reset_mock()
    await backend.pause()
    client.pause.assert_not_awaited()


@pytest.mark.parametrize("transport", ["PLAYING", "TRANSITIONING"])
async def test_foreign_source_playing_after_our_stop_is_a_takeover(rig, transport):
    backend, client, _, manager, _ = rig
    _soundtouch(backend, client)
    await backend.stop()
    client.get_media_info.return_value = SPOTIFY_URI
    client.get_transport_info.return_value = transport
    assert await backend.check_external_playback()
    assert not manager.is_renderer_active


async def test_natural_track_end_on_soundtouch_is_not_a_takeover(rig):
    """At the end of a track the SoundTouch reports STOPPED with CurrentURI
    qplay:// but keeps our URI in TrackURI; the queue must advance."""
    backend, client, _, manager, _ = rig
    _soundtouch(backend, client)
    client.get_media_info.return_value = SOUNDTOUCH_IDLE_URI
    client.get_track_uri.return_value = QOBUZ_URI
    client.get_transport_info.return_value = "STOPPED"
    assert not await backend.check_external_playback()
    assert manager.is_renderer_active
    assert await backend.can_apply_remote_state()
    metadata = BackendTrackMetadata(track_id="456", title="Next", artist="Artist")
    await backend.play(NEXT_URI, metadata)
    client.set_av_transport_uri.assert_awaited_once()


async def test_foreign_source_playing_with_stale_track_uri_is_a_takeover(rig):
    backend, client, _, manager, _ = rig
    _soundtouch(backend, client)
    client.get_media_info.return_value = SPOTIFY_URI
    client.get_track_uri.return_value = QOBUZ_URI
    client.get_transport_info.return_value = "PLAYING"
    assert await backend.check_external_playback()
    assert not manager.is_renderer_active


@pytest.mark.parametrize("transport", ["PAUSED_PLAYBACK", "STOPPED"])
async def test_foreign_source_paused_after_our_stop_is_a_takeover(rig, transport):
    """After our Stop only the standby placeholder is ours. Spotify or AirPlay
    paused on the speaker must still fail closed."""
    backend, client, _, manager, _ = rig
    _soundtouch(backend, client)
    await backend.stop()
    client.get_media_info.return_value = SPOTIFY_URI
    client.get_track_uri.return_value = SPOTIFY_URI
    client.get_transport_info.return_value = transport
    assert not await backend.can_apply_remote_state()
    assert await backend.check_external_playback()
    assert not manager.is_renderer_active


async def test_failed_track_uri_read_does_not_hide_a_takeover(rig):
    backend, client, _, manager, _ = rig
    _soundtouch(backend, client)
    client.get_media_info.return_value = SPOTIFY_URI
    client.get_track_uri.return_value = None
    client.get_transport_info.return_value = "STOPPED"
    assert not await backend.can_apply_remote_state()
    assert await backend.check_external_playback()
    assert not manager.is_renderer_active


async def test_idle_placeholder_without_our_stop_is_a_takeover(rig):
    """Only our own Stop explains the renderer dropping our track."""
    backend, client, _, manager, _ = rig
    _soundtouch(backend, client)
    client.get_media_info.return_value = SOUNDTOUCH_IDLE_URI
    client.get_track_uri.return_value = SOUNDTOUCH_IDLE_TRACK_URI
    client.get_transport_info.return_value = "PAUSED_PLAYBACK"
    assert await backend.check_external_playback()
    assert not manager.is_renderer_active


@pytest.mark.parametrize("uri", [None, ""])
async def test_unknown_source_blocks_snapshot_without_relinquishing_session(rig, uri):
    backend, client, player, manager, tasks = rig
    client.get_track_uri.return_value = uri
    await deliver_snapshot(manager, tasks, 2)
    assert_no_transport_writes(client)
    assert not backend._external_playback
    assert manager.is_renderer_active

    # A transient read failure must not require a new device selection.
    client.get_track_uri.return_value = QOBUZ_URI
    await deliver_snapshot(manager, tasks, 3)
    client.seek.assert_awaited_once_with(4000)
    assert player.state == PlaybackState.PAUSED


async def test_pending_track_load_cannot_replace_new_external_source(rig):
    backend, client, _, _, _ = rig
    metadata = BackendTrackMetadata(track_id="456", title="Next", artist="Artist")
    with pytest.raises(RuntimeError, match="External source"):
        await backend.play(NEXT_URI, metadata)
    assert_no_transport_writes(client)


async def test_loading_old_uri_is_not_mistaken_for_external_takeover(rig):
    import time

    backend, client, _, manager, _ = rig
    backend._playback_started_at = time.monotonic()
    assert not await backend.check_external_playback()
    with pytest.raises(RuntimeError, match="Cannot seek"):
        await backend.seek(4000)
    assert_no_transport_writes(client)
    client.get_track_uri.return_value = QOBUZ_URI
    await backend.seek(4000)
    client.seek.assert_awaited_once_with(4000)
    assert manager.is_renderer_active


async def test_inflight_uri_read_cannot_revoke_a_new_transport_load(rig):
    backend, client, _, manager, _ = rig
    reading = asyncio.Event()
    finish = asyncio.Event()

    async def read_uri():
        reading.set()
        await finish.wait()
        return SPOTIFY_URI

    client.get_track_uri.side_effect = read_uri
    check = asyncio.create_task(backend.check_external_playback())
    await reading.wait()
    backend._starting_playback = True
    finish.set()
    assert not await check
    assert manager.is_renderer_active


async def test_source_mismatch_logs_track_replacement_context(rig, caplog):
    backend, client, _, _, _ = rig
    client.get_track_uri.return_value = QOBUZ_URI
    caplog.set_level(logging.INFO, logger="qobuz_proxy.backends.dlna.backend")
    await backend.stop(next_track_id="789")

    # Model a renderer reporting the formerly armed track after Stop. These
    # diagnostics must distinguish this case from a foreign Spotify/AirPlay URI.
    client.get_track_uri.return_value = NEXT_URI
    await backend.check_external_playback()
    assert "replacement_track_id=789" in caplog.text
    assert "cleared_next=http://proxy/audio/456_2.flac" in caplog.text
    assert "observed=http://proxy/audio/456_2.flac" in caplog.text
    assert "expected=http://proxy/audio/123.flac" in caplog.text
    assert "cached_state=STOPPED" in caplog.text
    assert "last_successful_stop_age_s=None" not in caplog.text


@pytest.mark.parametrize(
    ("after_stop", "transport", "stop_succeeds", "allowed"),
    [
        (QOBUZ_URI, "STOPPED", True, True),
        (SPOTIFY_URI, "STOPPED", True, False),
        (QOBUZ_URI, "PLAYING", True, False),
        (QOBUZ_URI, "STOPPED", False, False),
        ("http://proxy:7120/audio/999.flac", "STOPPED", True, False),
    ],
)
async def test_replace_album_after_sonos_stop_resets_to_queue_head(
    rig, after_stop, transport, stop_succeeds, allowed
):
    backend, client, _, manager, _ = rig
    client.get_track_uri.return_value = QOBUZ_URI
    first = BackendTrackMetadata(track_id="123", title="First", artist="Artist")
    await backend.play(QOBUZ_URI, first)

    # Several gapless transitions later, the first entry remains in Sonos's queue.
    backend._current_proxy_url = NEXT_URI
    backend._playback_started_at = 0
    client.get_track_uri.return_value = NEXT_URI

    async def stop():
        client.get_track_uri.return_value = after_stop
        client.get_transport_info.return_value = transport
        return stop_succeeds

    client.stop.side_effect = stop
    await backend.stop(next_track_id="789")
    client.play_from_queue.reset_mock()
    replacement = BackendTrackMetadata(track_id="789", title="Replacement", artist="Artist")
    if not allowed:
        with pytest.raises(RuntimeError, match="External source"):
            await backend.play("http://proxy:7120/audio/789.flac", replacement)
        client.play_from_queue.assert_not_awaited()
        assert not manager.is_renderer_active
    else:
        await backend.play("http://proxy:7120/audio/789.flac", replacement)
        client.play_from_queue.assert_awaited_once_with(0)
        assert manager.is_renderer_active
        assert not backend._external_playback

        # Loading the replacement retires the old queue head.
        backend._playback_started_at = 0
        client.get_track_uri.return_value = QOBUZ_URI
        assert await backend.check_external_playback()


async def test_source_poll_during_stop_does_not_relinquish_queue(rig):
    backend, client, _, manager, _ = rig
    client.get_track_uri.return_value = QOBUZ_URI
    await backend.play(
        QOBUZ_URI, BackendTrackMetadata(track_id="123", title="First", artist="Artist")
    )
    backend._current_proxy_url = NEXT_URI
    backend._playback_started_at = 0
    client.get_track_uri.return_value = NEXT_URI

    async def stop():
        # Sonos changes its reported URI before the Stop response reaches us.
        client.get_track_uri.return_value = QOBUZ_URI
        assert not await backend.check_external_playback()
        assert manager.is_renderer_active
        return True

    client.stop.side_effect = stop
    await backend.stop(next_track_id="789")
    client.get_transport_info.return_value = "STOPPED"
    assert await backend.can_apply_remote_state()
    assert manager.is_renderer_active


@pytest.mark.parametrize(
    "uri",
    [
        "https://user:secret@cdn.example/private-id?token=secret#secret",
        "http://user:secret@proxy/audio/123.flac?token=secret",
        "x-sonos-spotify:secret",
        "http://[invalid-secret",
    ],
)
def test_source_diagnostics_do_not_expose_credentials_or_opaque_ids(uri):
    label = _source_label(uri)
    assert "secret" not in label
    assert "private-id" not in label
    assert "sha256=" in label


@pytest.mark.parametrize(
    ("uri", "transport", "allowed"),
    [
        ("", "STOPPED", True),
        ("", "NO_MEDIA_PRESENT", True),
        ("", "PLAYING", False),  # device-internal source (AirPlay, line-in)
        (None, "STOPPED", False),  # read failed: fail closed
        (SPOTIFY_URI, "STOPPED", False),
    ],
)
async def test_idle_renderer_without_a_uri_accepts_qobuz_again(rig, uri, transport, allowed):
    """GitHub #35: upmpdcli 1.9 clears CurrentURI when Qobuz stops it on
    deactivation, and the app's commands after switching back were refused."""
    backend, client, _, _, _ = rig
    backend._is_sonos = False
    backend._next_track_proxy_url = None
    client.get_media_info.return_value = uri
    client.get_track_uri.return_value = ""
    client.get_transport_info.return_value = transport
    assert await backend.can_apply_remote_state() is allowed


async def test_stopped_renderer_reporting_only_our_track_uri_accepts_qobuz(rig):
    backend, client, _, _, _ = rig
    backend._is_sonos = False
    client.get_media_info.return_value = ""
    client.get_track_uri.return_value = QOBUZ_URI
    client.get_transport_info.return_value = "STOPPED"
    assert await backend.can_apply_remote_state()
    assert not backend._external_playback
