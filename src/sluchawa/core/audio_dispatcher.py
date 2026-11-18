import asyncio
from typing import Dict, Callable, Awaitable

import numpy as np
from faster_whisper import WhisperModel
from loguru import logger

"""
Receieves paths to audio from outer world and appends them to the queue.
"""


class WhisperConsumer:
    def __init__(self, model_size: str = "base", device: str = "cuda", compute_type: str = "float16"):
        logger.info(f"Loading model faster-whisper ({model_size})")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    async def process_audio(self, user_id: int, pcm_bytes: bytes):
        audio_np = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768
        logger.debug(f"Processing audio {user_id}: {audio_np.shape}")

        try:
            segments, info = await asyncio.to_thread(
                self.model.transcribe,
                audio_np,
                beam_size=5,
                language="pl",
                vad_filter=True,
            )

            transcript = " ".join([segment.text for segment in segments]).strip()

            if transcript:
                logger.info(f"{user_id}: {transcript}")

        except Exception as e:
            logger.error(f"Error processing audio {user_id}: {e}")


class AudioDispatcher:
    def __init__(self, consumer_callback: Callable[[int, bytes], Awaitable[None]], chunk_duration_ms: int = 3000):
        self._consumer_callback = consumer_callback
        self._chunk_duration_ms = chunk_duration_ms
        self.user_queues: Dict[int, asyncio.Queue[bytes]] = {}
        self.workers: Dict[int, asyncio.Task] = {}
        self.is_running = False
        self._loop = asyncio.get_running_loop()

    async def start(self):
        self.is_running = True
        logger.info(f"Starting audio dispatcher")

    async def stop(self):
        self.is_running = False

        for user_id, task in self.workers.items():
            task.cancel()

        if self.workers:
            await asyncio.gather(*self.workers.values(), return_exceptions=True)

        self.user_queues.clear()
        self.workers.clear()
        logger.info(f"Stopping audio dispatcher")

    def dispatch_packet(self, user_id: int, pcm_chunk: bytes):
        if not self.is_running:
            return

        if user_id not in self.user_queues:
            self._loop.call_soon_threadsafe(self._init_user_pipeline, user_id)

        queue = self.user_queues.get(user_id)

        if queue:
            try:
                queue.put_nowait(pcm_chunk)

            except asyncio.QueueFull:
                logger.warning(f"Queue full, dropping {user_id}")

    def user_pipeline(self, user_id: int):
        if user_id in self.user_queues:
            return

        queue = asyncio.Queue[bytes] = asyncio.Queue(maxsize=200)
        self.user_queues[user_id] = queue
        self.workers[user_id] = asyncio.create_task(self._process_stream(user_id, queue))

    async def _process_stream(self, user_id: int, queue: asyncio.Queue[bytes]):
        buffer = bytearray()
        bytes_per_seccond = 16000 * 2
        target_buffer_size = int(bytes_per_seccond * (self._chunk_duration_ms / 1000))

        try:
            while self.is_running:
                chunk = await queue.get()
                buffer.extend(chunk)

                if len(buffer) >= target_buffer_size:
                    chunk_data = bytes(buffer)
                    buffer.clear()
                    await self._consumer_callback(user_id, chunk_data)

                queue.task_done()

        except asyncio.CancelledError:
            logger.debug(f"Cancelled, dropping {user_id}")
        except Exception as e:
            logger.error(f"Error processing audio {user_id}: {e}")
