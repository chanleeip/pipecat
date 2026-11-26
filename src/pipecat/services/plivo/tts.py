#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Plivo Koyal text-to-speech service implementation.

WebSocket integration with Plivo's realtime TTS API
(``wss://api.plivo.com/v1/Account/{auth_id}/TTS/Realtime/``). Each ``run_tts()``
call opens one Plivo reply (``context_id``) and closes it with ``finish: true``.
Consecutive replies on a connection continue each other's prosody server-side.
"""

import base64
import json
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field

from loguru import logger
from websockets.protocol import State

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import InterruptibleTTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.tracing.service_decorators import traced_tts
from pipecat.utils.types import NOT_GIVEN, NotGiven

KOYAL_MODEL = "koyal-v1"

# Sample rates the API accepts for raw pcm_s16le output.
_SUPPORTED_SAMPLE_RATES = (8000, 16000, 22050, 24000, 32000, 44100, 48000)

_LANGUAGE_MAP = {
    Language.BN: "bn",
    Language.EN: "en-IN",
    Language.EN_IN: "en-IN",
    Language.EN_US: "en-US",
    Language.EN_GB: "en-GB",
    Language.EN_AU: "en-AU",
    Language.GU: "gu",
    Language.HI: "hi",
    Language.KN: "kn",
    Language.MR: "mr",
    Language.TA: "ta",
    Language.TE: "te",
}


def language_to_koyal_language(language: Language | str) -> str:
    """Map a Language (or raw string) to a Koyal language code; unknown values mean ``auto``."""
    if isinstance(language, str) and not isinstance(language, Language):
        return language or "auto"
    return _LANGUAGE_MAP.get(language, "auto")


@dataclass
class PlivoTTSSettings(TTSSettings):
    """Settings for PlivoTTSService.

    Parameters:
        pace: Speaking pace multiplier (0.7-1.2).
        language_code: Koyal language code, or ``auto`` to follow the text.
    """

    pace: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    language_code: str | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)


class PlivoTTSService(InterruptibleTTSService):
    """Plivo Koyal real-time text-to-speech over WebSocket.

    Interruptions reconnect the socket, which drops any audio still queued for
    the cancelled reply.

    Example::

        tts = PlivoTTSService(
            auth_id="MA...",
            auth_token="...",
            settings=PlivoTTSService.Settings(voice="<voice_id>"),
        )
    """

    Settings = PlivoTTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        auth_id: str,
        auth_token: str,
        base_url: str = "wss://api.plivo.com",
        sample_rate: int | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Plivo Koyal TTS service.

        Args:
            auth_id: Plivo Auth ID (also part of the endpoint path).
            auth_token: Plivo Auth Token.
            base_url: Base WebSocket URL.
            sample_rate: Output sample rate in Hz; must be one the API supports.
            settings: Runtime-updatable settings (voice is a Plivo ``voice_id``).
            **kwargs: Passed to the parent InterruptibleTTSService.
        """
        default_settings = self.Settings(
            model=KOYAL_MODEL,
            voice=None,
            language=Language.EN,
            pace=None,
            language_code="auto",
        )
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            push_stop_frames=True,
            push_start_frame=True,
            pause_frame_processing=True,
            push_text_frames=True,
            sample_rate=sample_rate,
            settings=default_settings,
            **kwargs,
        )

        self._auth_id = auth_id
        self._auth_token = auth_token
        self._base_url = base_url.rstrip("/")
        self._receive_task = None
        # Plivo never lets a context_id be reused on a connection, so each
        # run_tts() call gets its own.
        self._reply_seq = 0

    def can_generate_metrics(self) -> bool:
        """Return True; this service reports TTFB and usage metrics."""
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert a Language enum to a Koyal language code."""
        return language_to_koyal_language(language)

    def _build_websocket_url(self) -> str:
        return f"{self._base_url}/v1/Account/{self._auth_id}/TTS/Realtime/"

    def _build_text_msg(self, text: str, plivo_context_id: str) -> dict:
        if self.sample_rate not in _SUPPORTED_SAMPLE_RATES:
            raise ValueError(f"Koyal does not support sample rate {self.sample_rate}")
        msg = {
            "type": "text",
            "context_id": plivo_context_id,
            "voice_id": self._settings.voice,
            "text": text,
            "finish": True,
            "model_id": self._settings.model or KOYAL_MODEL,
            "language": self._settings.language_code or "auto",
            "output_format": {
                "container": "raw",
                "encoding": "pcm_s16le",
                "sample_rate": self.sample_rate,
            },
        }
        if self._settings.pace is not None:
            msg["pace"] = self._settings.pace
        return msg

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the service and open the socket."""
        await super().setup(setup)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the service and close the socket."""
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the service and close the socket."""
        await super().cancel(frame)
        await self._disconnect()

    async def _connect(self):
        await super()._connect()
        await self._connect_websocket()
        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        await super()._disconnect()
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None
        await self._disconnect_websocket()

    async def _connect_websocket(self):
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return
            logger.debug("Connecting to Plivo TTS")
            token = base64.b64encode(f"{self._auth_id}:{self._auth_token}".encode()).decode()
            self._websocket = await self._websocket_connect(
                self._build_websocket_url(),
                additional_headers={"Authorization": f"Basic {token}"},
            )
            await self._call_event_handler("on_connected")
        except Exception as e:
            await self.push_error(error_msg=f"Plivo TTS connection error: {e}", exception=e)
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        try:
            await self.stop_all_metrics()
            if self._websocket:
                logger.debug("Disconnecting from Plivo TTS")
                await self._websocket.close()
        except Exception as e:
            await self.push_error(error_msg=f"Plivo TTS error closing websocket: {e}", exception=e)
        finally:
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _receive_messages(self):
        async for message in self._get_websocket():
            if isinstance(message, (bytes, bytearray)):
                # [2-byte little-endian context_id length][context_id][pcm audio]
                if len(message) < 2:
                    continue
                id_len = int.from_bytes(message[:2], "little")
                audio = bytes(message[2 + id_len :])
                if not audio:
                    continue
                await self.stop_ttfb_metrics()
                context_id = self.get_active_audio_context_id()
                await self.append_to_audio_context(
                    context_id,
                    TTSAudioRawFrame(
                        audio=audio,
                        sample_rate=self.sample_rate,
                        num_channels=1,
                        context_id=context_id,
                    ),
                )
                continue

            msg = json.loads(message)
            kind = msg.get("type")
            if kind == "done":
                await self.stop_all_metrics()
            elif kind == "error":
                context_id = self.get_active_audio_context_id()
                await self.push_frame(TTSStoppedFrame(context_id=context_id))
                await self.stop_all_metrics()
                await self.push_error(error_msg=f"Plivo TTS error: {msg.get('message', msg)}")
            elif kind == "warning":
                logger.warning(f"{self} warning: {msg.get('message', msg)}")

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Send ``text`` as one finished Plivo reply; audio arrives via the receive task."""
        try:
            if not self._websocket or self._websocket.state is State.CLOSED:
                await self._connect()

            try:
                self._reply_seq += 1
                plivo_context_id = f"{uuid.uuid4().hex[:12]}-{self._reply_seq}"
                msg = self._build_text_msg(text, plivo_context_id)
                await self._get_websocket().send(json.dumps(msg))
                await self.start_tts_usage_metrics(text)
            except Exception as e:
                yield ErrorFrame(error=f"Plivo TTS send error: {e}")
                yield TTSStoppedFrame(context_id=context_id)
                await self._disconnect()
                await self._connect()
                return
            yield None
        except Exception as e:
            yield ErrorFrame(error=f"Plivo TTS error: {e}")
