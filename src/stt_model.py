import torch
import torchaudio
from nemo.collections.asr.models import EncDecCTCModel
from nemo.collections.asr.modules.audio_preprocessing import (
    AudioToMelSpectrogramPreprocessor as NeMoAudioToMelSpectrogramPreprocessor,
)
from nemo.collections.asr.parts.preprocessing.features import (
    FilterbankFeaturesTA as NeMoFilterbankFeaturesTA,
)


class FilterbankFeaturesTA(NeMoFilterbankFeaturesTA):
    def __init__(self, mel_scale: str = "htk", wkwargs=None, **kwargs):
        if "window_size" in kwargs:
            del kwargs["window_size"]
        if "window_stride" in kwargs:
            del kwargs["window_stride"]

        super().__init__(**kwargs)

        self._mel_spec_extractor: torchaudio.transforms.MelSpectrogram = (
            torchaudio.transforms.MelSpectrogram(
                sample_rate=self._sample_rate,
                win_length=self.win_length,
                hop_length=self.hop_length,
                n_mels=kwargs["nfilt"],
                window_fn=self.torch_windows[kwargs["window"]],
                mel_scale=mel_scale,
                norm=kwargs["mel_norm"],
                n_fft=kwargs["n_fft"],
                f_max=kwargs.get("highfreq", None),
                f_min=kwargs.get("lowfreq", 0),
                wkwargs=wkwargs,
            )
        )


class AudioToMelSpectrogramPreprocessor(NeMoAudioToMelSpectrogramPreprocessor):
    def __init__(self, mel_scale: str = "htk", **kwargs):
        super().__init__(**kwargs)
        kwargs["nfilt"] = kwargs["features"]
        del kwargs["features"]
        self.featurizer = (
            FilterbankFeaturesTA(  
                mel_scale=mel_scale,
                **kwargs,
            )
        )

class STTModel:
    def __init__(self, config_path: str, weights_path: str, device: str) -> None:
        """Initialize STT model with configuration, weights and device settings."""
        self.config_path: str = config_path
        self.weights_path: str = weights_path
        self.device: str = device
        self.model = self.load_stt_model()

    def load_stt_model(self) -> EncDecCTCModel:
        """Load and configure the STT model with appropriate device settings."""
        model = EncDecCTCModel.from_config_file(self.config_path)
        ckpt = torch.load(self.weights_path, map_location="cpu")
        model.load_state_dict(ckpt, strict=False)
        model.eval()

        dev_type = self.device if isinstance(self.device, str) else self.device.type

        if dev_type == "cuda":
            model = model.to("cuda").half()
            model = torch.compile(model, mode="max-autotune")
        elif dev_type == "mps":
            model = model.to("mps").half()
        else:                                
            model = model.float()
        return model
    
    def transcribe(self, audio_path: str | list[str]) -> str:
        """
        Transcribe audio file using the loaded STT model.
        
        Args:
            audio_path: Path to the audio file to transcribe
            
        Returns:
            Transcribed text from the audio file
        """
        dev_type = self.device if isinstance(self.device, str) else self.device.type
        batch_size = 1 if dev_type == "cpu" else 4
        
        return self.model.transcribe(audio_path, batch_size=batch_size)