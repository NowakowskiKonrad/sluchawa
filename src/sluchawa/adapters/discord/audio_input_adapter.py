from sluchawa.ports.audio_input_port import AudioInputPort

"""
That's where discord bot actually interacts with domain code.

After collecting all the audio required and getting stop command it should add paths to audio file to queue.
"""


class DiscordAudioInputAdapter:
    def __init__(self, domain_port: AudioInputPort):
        self._domain_port = domain_port

    def add_audio_to_queue(self):
        # TODO: Implement proper appending to queue
        self._domain_port.submit_audio([])
