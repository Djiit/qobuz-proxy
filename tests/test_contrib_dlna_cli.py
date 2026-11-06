"""contrib/dlna_cli.py and the fake renderer's SoundTouch quirks."""

import importlib.util
import io
import sys
import wave
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "contrib" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


dlna_cli = _load("dlna_cli")
fake_renderer = _load("fake_renderer")


def test_parse_script_splits_commands_and_arguments():
    assert dlna_cli.parse_script("play 1; wait 2.5 ;next;; status") == [
        ["play", "1"],
        ["wait", "2.5"],
        ["next"],
        ["status"],
    ]


def test_tone_wav_has_expected_format():
    data = dlna_cli.make_tone_wav(440.0, 1, sample_rate=8000)
    with wave.open(io.BytesIO(data)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (2, 2, 8000)
        assert w.getnframes() == 8000


def test_tone_flac_is_flac():
    pytest.importorskip("soundfile")
    assert dlna_cli.make_tone_flac(440.0, 1)[:4] == b"fLaC"


@pytest.mark.parametrize(
    ("header", "status", "body", "content_range"),
    [
        ("bytes=2-5", 206, b"2345", "bytes 2-5/10"),
        ("bytes=7-", 206, b"789", "bytes 7-9/10"),
        ("bytes=-3", 206, b"789", "bytes 7-9/10"),
        ("bytes=8-99", 206, b"89", "bytes 8-9/10"),
        ("bytes=10-", 416, b"", "bytes */10"),
    ],
)
async def test_tone_server_serves_ranges(header, status, body, content_range):
    server = dlna_cli.ToneServer({"t.wav": b"0123456789"}, "127.0.0.1", 0)
    app = web.Application()
    app.router.add_route("*", "/{name}", server._handle)
    async with TestServer(app) as ts:
        from aiohttp import ClientSession

        async with ClientSession() as s:
            async with s.get(ts.make_url("/t.wav"), headers={"Range": header}) as r:
                assert r.status == status
                assert await r.read() == body
                assert r.headers["Content-Range"] == content_range
            async with s.get(ts.make_url("/missing.wav")) as r:
                assert r.status == 404


def test_parse_line_reports_unbalanced_quotes():
    assert dlna_cli.parse_line('play "1') is None
    assert dlna_cli.parse_line("play 2") == ["play", "2"]


def test_flac_without_optional_deps_explains_the_fix(monkeypatch):
    monkeypatch.setitem(sys.modules, "soundfile", None)
    with pytest.raises(SystemExit, match="--format wav"):
        dlna_cli.make_tone_flac(440.0, 1)


@pytest.fixture
async def soundtouch():
    """The fake renderer with SoundTouch quirks, behind a real HTTP server."""
    renderer = fake_renderer.FakeRenderer(sink=fake_renderer.SINK_GMEDIARENDER, quirks="soundtouch")
    app = web.Application()
    app.router.add_get("/description.xml", renderer.handle_description)
    for svc in ("avtransport", "renderingcontrol", "connectionmanager"):
        app.router.add_post(f"/upnp/control/{svc}", renderer.handle_soap)
    async with TestServer(app, host="127.0.0.1") as ts:
        yield ts


async def test_cli_skips_on_soundtouch_without_external_takeover(soundtouch):
    """End to end: the skip sequence the Qobuz app triggers keeps control."""
    from qobuz_proxy.backends.dlna.backend import DLNABackend

    tones = dlna_cli.ToneServer({f"tone{n}.wav": b"RIFF" for n in (1, 2, 3)}, "127.0.0.1", 0)
    tracks = [dlna_cli.Track(n, 440.0, 30, f"tone{n}.wav") for n in (1, 2, 3)]
    backend = DLNABackend(
        "127.0.0.1",
        port=soundtouch.port,
        description_url=str(soundtouch.make_url("/description.xml")),
    )
    takeovers = []

    async def on_external():
        takeovers.append(True)

    backend.on_external_playback(on_external)
    assert await backend.connect()
    try:
        session = dlna_cli.Session(backend, tones, tracks)
        await session.play(1)
        backend._playback_started_at = 0  # past the start grace period
        await session.change(+1)
        backend._playback_started_at = 0
        await session.change(+1)
        backend._playback_started_at = 0
        await session.change(-1)
        assert backend._current_proxy_url.endswith("/tone2.wav")
        await backend.pause()
        assert await backend.resume()
        await backend.stop()
        assert not await backend.check_external_playback()
        assert takeovers == []
        assert not backend._external_playback
    finally:
        await backend.disconnect()
