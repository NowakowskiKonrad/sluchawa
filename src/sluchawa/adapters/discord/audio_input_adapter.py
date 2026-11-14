import uuid
from pathlib import Path
import Discord

from sluchawa.ports.audio_input_port import AudioInputPort

"""
That's where discord bot actually interacts with domain code.

After collecting all the audio required and getting stop command it should add paths to audio file to queue.
"""


class DiscordAudioInputAdapter:
    def __init__(self, domain_port: AudioInputPort, voice_client: discord.VoiceClient):
        self._domain_port = domain_port
        self._voice_client = voice_client
        self.session_id = str(uuid.uuid4())
        self._output_dir = Path(f"/tmp/sluchawa_records/{self.session_id}")
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def start_recording(self) -> None:
        self._voice_client.start_recording(
            discord.sinks.OggSink(),
            self.add_audio_to_queue
        )

    def stop_recording(self) -> None:
        self._voice_client.stop_recording()

    async def add_audio_to_queue(self, sink: discord.sinks.Sink) -> None:
        saved_paths: list[Path] = []

        for user_id, audio in sink.audio_data.items():
            file_path = self._output_dir / f"{user_id}.ogg"
            with open(file_path, "wb") as f:
                f.write(audio.file.read())

            saved_paths.append(file_path)

        self._domain_port.submit_audio(saved_paths)
