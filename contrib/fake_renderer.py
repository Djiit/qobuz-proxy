#!/usr/bin/env python3
"""Fake DLNA MediaRenderer for local testing.

Emulates just enough UPnP/DLNA (device description + SOAP control endpoints)
for qobuz-proxy to connect, detect capabilities, and drive basic transport
state. No audio is played; PLAYING state advances a fake position clock.

--on-fetch-error makes Play fetch the start of the URI, as a real renderer
does, and picks how the transport reacts when that fetch fails:

  stop            go STOPPED (like upmpdcli + mpd)
  stay-playing    keep reporting PLAYING with the position frozen at 0
  transitioning   report TRANSITIONING indefinitely

Without it, Play never fetches the URI.

Profiles control what GetProtocolInfo advertises:

  gmediarender  FLAC with no rate/depth info (like gmrender-resurrect) —
                exercises the auto-quality "could not detect" fallback path
  hires         FLAC with DLNA.ORG_PN=FLAC_192 — exercises confirmed
                hi-res auto-detection

--quirks soundtouch reproduces how a Bose SoundTouch reports its transport:
CurrentURI stays empty while playing (only TrackURI shows the track), Stop
drops to a standby placeholder (CurrentURI qplay://, PAUSED_PLAYBACK) and
SetNextAVTransportURI is not supported.

Usage:
    python3 contrib/fake_renderer.py --port 49494 --profile gmediarender

Then point a speaker at it:
    speakers:
    - name: Fake Renderer
      backend: dlna
      max_quality: auto
      dlna_ip: 127.0.0.1
      dlna_port: 49494
"""

import argparse
import asyncio
import logging
import time
from typing import Optional

from aiohttp import ClientSession, ClientTimeout, web

logger = logging.getLogger("fake_renderer")

UDN = "uuid:FakeRender-1_0-000-000-001"

# The audio section of a real gmrender-resurrect (GStreamer) Sink: FLAC is
# advertised with no DLNA profile and no rate/depth attributes, so a parser
# cannot tell what the device actually supports.
SINK_GMEDIARENDER = (
    "http-get:*:audio/*:*,"
    "http-get:*:audio/L16;rate=44100;channels=2:*,"
    "http-get:*:audio/mpeg:*,"
    "http-get:*:audio/ogg:*,"
    "http-get:*:audio/x-flac:*,"
    "http-get:*:audio/x-wav:*,"
    "http-get:*:video/x-matroska:*"
)

# A device that states its FLAC support explicitly via DLNA profiles.
SINK_HIRES = (
    "http-get:*:audio/mpeg:DLNA.ORG_PN=MP3,"
    "http-get:*:audio/flac:DLNA.ORG_PN=FLAC,"
    "http-get:*:audio/flac:DLNA.ORG_PN=FLAC_192;DLNA.ORG_OP=01"
)

PROFILES = {
    "gmediarender": SINK_GMEDIARENDER,
    "hires": SINK_HIRES,
}

DESCRIPTION_XML = f"""<?xml version="1.0" encoding="utf-8"?>
<root xmlns="urn:schemas-upnp-org:device-1-0">
  <specVersion><major>1</major><minor>0</minor></specVersion>
  <device>
    <deviceType>urn:schemas-upnp-org:device:MediaRenderer:1</deviceType>
    <friendlyName>FakeRender</friendlyName>
    <manufacturer>qobuz-proxy contrib</manufacturer>
    <modelName>fake_renderer</modelName>
    <UDN>{UDN}</UDN>
    <serviceList>
      <service>
        <serviceType>urn:schemas-upnp-org:service:AVTransport:1</serviceType>
        <serviceId>urn:upnp-org:serviceId:AVTransport</serviceId>
        <controlURL>/upnp/control/avtransport</controlURL>
        <eventSubURL>/upnp/event/avtransport</eventSubURL>
        <SCPDURL>/upnp/scpd/avtransport</SCPDURL>
      </service>
      <service>
        <serviceType>urn:schemas-upnp-org:service:RenderingControl:1</serviceType>
        <serviceId>urn:upnp-org:serviceId:RenderingControl</serviceId>
        <controlURL>/upnp/control/renderingcontrol</controlURL>
        <eventSubURL>/upnp/event/renderingcontrol</eventSubURL>
        <SCPDURL>/upnp/scpd/renderingcontrol</SCPDURL>
      </service>
      <service>
        <serviceType>urn:schemas-upnp-org:service:ConnectionManager:1</serviceType>
        <serviceId>urn:upnp-org:serviceId:ConnectionManager</serviceId>
        <controlURL>/upnp/control/connectionmanager</controlURL>
        <eventSubURL>/upnp/event/connectionmanager</eventSubURL>
        <SCPDURL>/upnp/scpd/connectionmanager</SCPDURL>
      </service>
    </serviceList>
  </device>
</root>
"""


def _soap_response(service: str, action: str, args: dict) -> web.Response:
    body = "".join(f"<{k}>{v}</{k}>" for k, v in args.items())
    xml = (
        '<?xml version="1.0"?>'
        '<s:Envelope xmlns:s="http://schemas.xmlsoap.org/soap/envelope/" '
        's:encodingStyle="http://schemas.xmlsoap.org/soap/encoding/">'
        "<s:Body>"
        f'<u:{action}Response xmlns:u="{service}">{body}</u:{action}Response>'
        "</s:Body></s:Envelope>"
    )
    return web.Response(text=xml, content_type="text/xml")


def _fmt_hms(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"


SOUNDTOUCH_IDLE_URI = "qplay://"
SOUNDTOUCH_IDLE_TRACK_URI = "http://invalidHost.com/invalidUri.mp3"


class FakeRenderer:
    def __init__(
        self, sink: str, on_fetch_error: Optional[str] = None, quirks: Optional[str] = None
    ):
        self.sink = sink
        self.on_fetch_error = on_fetch_error
        self.quirks = quirks
        self._fetch_task: Optional[asyncio.Task[None]] = None
        self.volume = 39
        self.mute = 0
        self.state = "STOPPED"
        self.uri = ""
        self.next_uri = ""
        self._play_started: float = 0.0
        self._position_base: float = 0.0

    def _position_seconds(self) -> float:
        if self.state == "PLAYING" and self._play_started:
            return self._position_base + (time.monotonic() - self._play_started)
        return self._position_base

    async def handle_description(self, request: web.Request) -> web.Response:
        return web.Response(text=DESCRIPTION_XML, content_type="text/xml")

    async def handle_soap(self, request: web.Request) -> web.Response:
        soap_action = request.headers.get("SOAPAction", "").strip('"')
        service, _, action = soap_action.partition("#")
        body = await request.text()
        logger.info("SOAP %s", action)

        if action == "GetProtocolInfo":
            return _soap_response(service, action, {"Source": "", "Sink": self.sink})

        if action == "GetVolume":
            return _soap_response(service, action, {"CurrentVolume": self.volume})
        if action == "SetVolume":
            vol = _extract(body, "DesiredVolume")
            if vol and vol.isdigit():
                self.volume = int(vol)
            return _soap_response(service, action, {})
        if action == "GetMute":
            return _soap_response(service, action, {"CurrentMute": self.mute})
        if action == "SetMute":
            mute = _extract(body, "DesiredMute")
            self.mute = 1 if mute in ("1", "true") else 0
            return _soap_response(service, action, {})

        if action == "SetAVTransportURI":
            self.uri = _extract(body, "CurrentURI") or ""
            self._position_base = 0.0
            return _soap_response(service, action, {})
        if action == "SetNextAVTransportURI":
            if self.quirks == "soundtouch":
                return _soap_fault(401, "Invalid Action")
            self.next_uri = _extract(body, "NextURI") or ""
            return _soap_response(service, action, {})
        if action == "Play":
            self._play_started = time.monotonic()
            self.state = "PLAYING"
            if self.on_fetch_error and self.uri:
                if self._fetch_task:
                    self._fetch_task.cancel()
                self._fetch_task = asyncio.create_task(self._fetch(self.uri))
            return _soap_response(service, action, {})
        if action == "Pause":
            self._position_base = self._position_seconds()
            self.state = "PAUSED_PLAYBACK"
            return _soap_response(service, action, {})
        if action == "Stop":
            if self._fetch_task:
                self._fetch_task.cancel()
                self._fetch_task = None
            self.state = "STOPPED"
            self._position_base = 0.0
            if self.quirks == "soundtouch":
                self.state = "PAUSED_PLAYBACK"
                self.uri = ""
            return _soap_response(service, action, {})
        if action == "Seek":
            target = _extract(body, "Target") or "0:00:00"
            parts = [int(p) for p in target.split(":")]
            self._position_base = parts[0] * 3600 + parts[1] * 60 + parts[2]
            self._play_started = time.monotonic()
            return _soap_response(service, action, {})
        if action == "GetMediaInfo":
            media_uri = self.uri
            if self.quirks == "soundtouch":
                loaded = self.uri and self.state in ("PLAYING", "PAUSED_PLAYBACK")
                media_uri = "" if loaded else SOUNDTOUCH_IDLE_URI
            return _soap_response(
                service,
                action,
                {
                    "NrTracks": "1" if self.uri else "0",
                    "MediaDuration": "0:04:00",
                    "CurrentURI": _xml_escape(media_uri),
                    "CurrentURIMetaData": "",
                    "NextURI": _xml_escape(self.next_uri),
                    "NextURIMetaData": "",
                    "PlayMedium": "NETWORK",
                    "RecordMedium": "NOT_IMPLEMENTED",
                    "WriteStatus": "NOT_IMPLEMENTED",
                },
            )
        if action == "GetTransportInfo":
            return _soap_response(
                service,
                action,
                {
                    "CurrentTransportState": self.state,
                    "CurrentTransportStatus": "OK",
                    "CurrentSpeed": "1",
                },
            )
        if action == "GetPositionInfo":
            pos = _fmt_hms(self._position_seconds())
            return _soap_response(
                service,
                action,
                {
                    "Track": "1" if self.uri else "0",
                    "TrackDuration": "0:04:00",
                    "TrackURI": _xml_escape(
                        self.uri or (SOUNDTOUCH_IDLE_TRACK_URI if self.quirks else "")
                    ),
                    "RelTime": pos,
                    "AbsTime": pos,
                    "RelCount": "0",
                    "AbsCount": "0",
                },
            )

        logger.warning("Unhandled SOAP action: %s", soap_action)
        return web.Response(status=500, text="Unhandled action")

    async def _fetch(self, uri: str) -> None:
        """Read the start of the stream; on failure, react as configured."""
        try:
            async with ClientSession(timeout=ClientTimeout(total=60)) as session:
                async with session.get(uri) as resp:
                    if resp.status not in (200, 206):
                        raise RuntimeError(f"HTTP {resp.status}")
                    await resp.content.read(65536)
            logger.info("Fetched start of %s", uri)
            return
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("Fetch of %s failed (%s); reacting with %s", uri, e, self.on_fetch_error)
        if uri != self.uri:
            return
        if self.on_fetch_error == "stop":
            self.state = "STOPPED"
            self._position_base = 0.0
        elif self.on_fetch_error == "stay-playing":
            self._play_started = 0.0
            self._position_base = 0.0
        elif self.on_fetch_error == "transitioning":
            self.state = "TRANSITIONING"
            self._position_base = 0.0


def _soap_fault(code: int, description: str) -> web.Response:
    xml = (
        '<?xml version="1.0"?>'
        '<s:Envelope xmlns:s="http://schemas.xmlsoap.org/soap/envelope/" '
        's:encodingStyle="http://schemas.xmlsoap.org/soap/encoding/">'
        "<s:Body><s:Fault><faultcode>s:Client</faultcode><faultstring>UPnPError</faultstring>"
        '<detail><UPnPError xmlns="urn:schemas-upnp-org:control-1-0">'
        f"<errorCode>{code}</errorCode><errorDescription>{description}</errorDescription>"
        "</UPnPError></detail></s:Fault></s:Body></s:Envelope>"
    )
    return web.Response(status=500, text=xml, content_type="text/xml")


def _xml_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _extract(xml_text: str, tag: str) -> str:
    start = xml_text.find(f"<{tag}>")
    if start == -1:
        return ""
    start += len(tag) + 2
    end = xml_text.find(f"</{tag}>", start)
    return xml_text[start:end] if end != -1 else ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Fake DLNA renderer for qobuz-proxy testing")
    parser.add_argument("--port", type=int, default=49494)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="gmediarender")
    parser.add_argument(
        "--on-fetch-error", choices=["stop", "stay-playing", "transitioning"], default=None
    )
    parser.add_argument("--quirks", choices=["soundtouch"], default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    renderer = FakeRenderer(
        sink=PROFILES[args.profile], on_fetch_error=args.on_fetch_error, quirks=args.quirks
    )
    app = web.Application()
    app.router.add_get("/description.xml", renderer.handle_description)
    app.router.add_post("/upnp/control/avtransport", renderer.handle_soap)
    app.router.add_post("/upnp/control/renderingcontrol", renderer.handle_soap)
    app.router.add_post("/upnp/control/connectionmanager", renderer.handle_soap)

    logger.info("Fake renderer (%s profile) on http://127.0.0.1:%d", args.profile, args.port)
    web.run_app(app, host="127.0.0.1", port=args.port, print=None)


if __name__ == "__main__":
    main()
