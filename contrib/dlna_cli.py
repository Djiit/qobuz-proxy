#!/usr/bin/env python3
"""Drive a DLNA renderer through qobuz-proxy's DLNABackend, without Qobuz.

Serves short generated FLAC tones from this machine and sends them to the
renderer with the same backend code the proxy uses, so play/pause/next/
previous/stop can be tried against a real speaker (or contrib/fake_renderer.py)
and the renderer's raw UPnP state inspected after each command.

The renderer must be able to reach this machine on --serve-port.

Usage:
    uv run python contrib/dlna_cli.py --ip 192.168.1.152 --port 8091
    uv run python contrib/dlna_cli.py --ip 192.168.1.152 --port 8091 \\
        --run "play; wait 5; next; wait 5; prev; wait 5; pause; resume; stop"

Commands:
    play [n]     play track n (default: current, starts at 1)
    next / prev  load the next / previous track the way the player does
                 (stop the current one, then play the new one)
    pause        pause
    resume       resume
    stop         stop
    status       renderer UPnP state and backend state
    wait <s>     sleep s seconds (useful in --run)
    quit
"""

import argparse
import asyncio
import io
import logging
import math
import shlex
import socket
import struct
import sys
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from aiohttp import web

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qobuz_proxy.backends.dlna.backend import DLNABackend  # noqa: E402
from qobuz_proxy.backends.types import BackendTrackMetadata  # noqa: E402

logger = logging.getLogger("dlna_cli")

SAMPLE_RATE = 44100
# One tone per track so a track change is audible.
TONES_HZ = (440.0, 554.37, 659.25, 880.0)


@dataclass
class Track:
    number: int
    frequency: float
    duration_s: int
    path: str


def make_tone_wav(frequency: float, duration_s: int, sample_rate: int = SAMPLE_RATE) -> bytes:
    """16-bit stereo sine tone with a short fade so cuts are click free."""
    frames = sample_rate * duration_s
    fade = sample_rate // 50
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        samples = bytearray()
        for i in range(frames):
            gain = min(1.0, i / fade, (frames - i) / fade) * 0.3
            v = int(32767 * gain * math.sin(2 * math.pi * frequency * i / sample_rate))
            samples += struct.pack("<hh", v, v)
        w.writeframes(bytes(samples))
    return buf.getvalue()


def make_tone_flac(frequency: float, duration_s: int) -> bytes:
    """FLAC encode a tone; FLAC is what the proxy normally serves."""
    import numpy as np
    import soundfile as sf

    t = np.arange(SAMPLE_RATE * duration_s) / SAMPLE_RATE
    fade = np.minimum(1.0, np.minimum(t, duration_s - t) * 50)
    mono = (0.3 * fade * np.sin(2 * np.pi * frequency * t)).astype(np.float32)
    out = io.BytesIO()
    sf.write(out, np.stack([mono, mono], axis=1), SAMPLE_RATE, format="FLAC", subtype="PCM_16")
    return out.getvalue()


def parse_script(script: str) -> list[list[str]]:
    """Split "play 1; wait 2; next" into [["play", "1"], ["wait", "2"], ["next"]]."""
    return [shlex.split(part) for part in script.split(";") if part.strip()]


def local_ip_for(target: str) -> str:
    """Address of the interface this machine uses to reach ``target``."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect((target, 9))
        return str(s.getsockname()[0])
    finally:
        s.close()


class ToneServer:
    """Serves the generated tracks and logs every renderer request."""

    def __init__(self, tracks: dict[str, bytes], host: str, port: int):
        self._tracks = tracks
        self._host = host
        self._port = port
        self._runner: Optional[web.AppRunner] = None

    def url(self, path: str) -> str:
        return f"http://{self._host}:{self._port}/{path}"

    async def start(self) -> None:
        app = web.Application()
        app.router.add_route("*", "/{name}", self._handle)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        await web.TCPSite(self._runner, "0.0.0.0", self._port).start()

    async def stop(self) -> None:
        if self._runner:
            await self._runner.cleanup()

    async def _handle(self, request: web.Request) -> web.StreamResponse:
        name = request.match_info["name"]
        body = self._tracks.get(name)
        logger.info(
            "renderer %s %s %s range=%s ua=%s",
            request.remote,
            request.method,
            name,
            request.headers.get("Range", "-"),
            request.headers.get("User-Agent", "-"),
        )
        if body is None:
            return web.Response(status=404)
        ctype = "audio/flac" if name.endswith(".flac") else "audio/wav"
        start, end = 0, len(body) - 1
        status = 200
        rng = request.headers.get("Range", "")
        if rng.startswith("bytes="):
            first, _, last = rng[6:].partition("-")
            start = int(first or 0)
            end = int(last) if last else end
            status = 206
        headers = {"Accept-Ranges": "bytes", "Content-Type": ctype}
        if status == 206:
            headers["Content-Range"] = f"bytes {start}-{end}/{len(body)}"
        if request.method == "HEAD":
            headers["Content-Length"] = str(end - start + 1)
            return web.Response(status=status, headers=headers)
        return web.Response(status=status, body=body[start : end + 1], headers=headers)


class Session:
    def __init__(self, backend: DLNABackend, server: ToneServer, tracks: list[Track]):
        self.backend = backend
        self.server = server
        self.tracks = tracks
        self.current = 0
        self.playing = False
        self.t0 = time.monotonic()

    def _stamp(self) -> str:
        return f"[{time.monotonic() - self.t0:7.2f}s]"

    async def play(self, number: Optional[int] = None) -> None:
        if number is not None:
            self.current = max(0, min(len(self.tracks) - 1, number - 1))
        track = self.tracks[self.current]
        meta = BackendTrackMetadata(
            track_id=str(track.number),
            title=f"Tone {track.number} ({track.frequency:.0f} Hz)",
            artist="dlna_cli",
            album="qobuz-proxy test",
            duration_ms=track.duration_s * 1000,
            sample_rate=SAMPLE_RATE,
            bit_depth=16,
        )
        await self.backend.play(self.server.url(track.path), meta)
        self.playing = True

    async def change(self, delta: int) -> None:
        """Mirror the player: stop the outgoing track, then play the new one."""
        target = max(0, min(len(self.tracks) - 1, self.current + delta))
        if self.playing:
            await self.backend.stop(next_track_id=str(self.tracks[target].number))
        self.current = target
        await self.play()

    async def status(self) -> None:
        client = self.backend._client
        if not client:
            print("not connected")
            return
        transport = await client.get_transport_info()
        media_uri = await client.get_media_info()
        track_uri = await client.get_track_uri()
        position = await client.get_position_info()
        print(
            f"{self._stamp()} renderer: transport={transport} CurrentURI={media_uri!r} "
            f"TrackURI={track_uri!r} position_ms={position}"
        )
        print(
            f"{self._stamp()} backend:  state={self.backend._state.name} "
            f"external={self.backend._external_playback} "
            f"current={self.backend._current_proxy_url} track={self.current + 1}"
        )

    async def run(self, words: list[str]) -> bool:
        cmd, args = words[0].lower(), words[1:]
        print(f"{self._stamp()} > {' '.join(words)}")
        try:
            if cmd == "play":
                await self.play(int(args[0]) if args else None)
            elif cmd == "next":
                await self.change(+1)
            elif cmd in ("prev", "previous"):
                await self.change(-1)
            elif cmd == "pause":
                await self.backend.pause()
            elif cmd == "resume":
                print(f"{self._stamp()} resume accepted: {await self.backend.resume()}")
            elif cmd == "stop":
                await self.backend.stop()
                self.playing = False
            elif cmd == "status":
                pass
            elif cmd == "wait":
                await asyncio.sleep(float(args[0]) if args else 1.0)
            elif cmd in ("quit", "exit"):
                return False
            else:
                print(__doc__)
                return True
        except Exception as e:
            print(f"{self._stamp()} ! {cmd} failed: {e}")
        if cmd != "wait":
            await self.status()
        return True


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ip", required=True, help="renderer IP")
    parser.add_argument("--port", type=int, default=8091, help="renderer UPnP port")
    parser.add_argument("--description-url", default=None)
    parser.add_argument("--serve-port", type=int, default=7299)
    parser.add_argument("--serve-host", default=None, help="address the renderer fetches from")
    parser.add_argument("--tracks", type=int, default=3)
    parser.add_argument("--duration", type=int, default=30, help="seconds per track")
    parser.add_argument("--format", choices=["flac", "wav"], default="flac")
    parser.add_argument("--run", default=None, help='commands separated by ";"')
    parser.add_argument("-v", "--verbose", action="store_true", help="show backend debug logs")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    if not args.verbose:
        logging.getLogger("qobuz_proxy.backends.dlna.client").setLevel(logging.WARNING)

    tracks = []
    files = {}
    for n in range(1, args.tracks + 1):
        freq = TONES_HZ[(n - 1) % len(TONES_HZ)]
        path = f"tone{n}.{args.format}"
        if args.format == "flac":
            files[path] = make_tone_flac(freq, args.duration)
        else:
            files[path] = make_tone_wav(freq, args.duration)
        tracks.append(Track(n, freq, args.duration, path))

    host = args.serve_host or local_ip_for(args.ip)
    server = ToneServer(files, host, args.serve_port)
    await server.start()
    logger.info("serving %d tracks on http://%s:%d/", len(tracks), host, args.serve_port)

    backend = DLNABackend(args.ip, port=args.port, description_url=args.description_url)
    backend.on_external_playback(_on_external)
    if not await backend.connect():
        await server.stop()
        raise SystemExit(f"could not connect to renderer {args.ip}:{args.port}")

    session = Session(backend, server, tracks)
    try:
        await session.status()
        if args.run:
            for words in parse_script(args.run):
                if not await session.run(words):
                    break
        else:
            loop = asyncio.get_running_loop()
            while True:
                line = await loop.run_in_executor(None, input, "dlna> ")
                words = shlex.split(line)
                if words and not await session.run(words):
                    break
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        await backend.disconnect()
        await server.stop()


async def _on_external() -> None:
    logger.warning("backend reported an EXTERNAL SOURCE TAKEOVER (proxy would leave the session)")


if __name__ == "__main__":
    asyncio.run(main())
