import asyncio
from idlelib.debugobj import dispatch

from sluchawa.core.audio_dispatcher import WhisperConsumer, AudioDispatcher


def main():
    """
    Create domain audio dispatcher
    Create event loop
    Create discord audio input adapter (passing the loop and dispatcher)
    Create discord notification adapter (passing the loop)
    Create domain audio processing (passing notificaion adapter)
    Run audio processing in separate thread

    Wait for input
    """

async def main():
    whisper_consumer = WhisperConsumer(model_size="small", device="cuda", compute_type="float16")
    dispatcher = AudioDispatcher(
        consumer_callback=whisper_consumer.process_audio,
        chunk_duration_ms=3000
    )
    await dispatcher.start()

    dummp_pcm_packet = b'\x00\x00' * 1600

    for _ in range(40):
        dispatcher.dispatch_packet(user_id=123, pcm_chunk=dummp_pcm_packet)
        await asyncio.sleep(0.1)

    await asyncio.sleep(5)
    await dispatcher.stop()

if __name__ == "__main__":
    main()
