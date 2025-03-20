import gigaam
import torch


class STTModel:
    def __init__(self, model: str, *, fp16_encoder: bool, device: str) -> None:
        """
        Initialize the STTModel with the specified model, fp16 encoder flag, and device.

        Args:
            model (str): The name of the model to load.
            fp16_encoder (bool): Flag indicating whether to use fp16 encoding.
            device (str): The device to run the model on (e.g., 'cpu' or 'cuda').

        """
        self.model = model
        self.fp16_encoder = fp16_encoder
        self.device = device
        self.opt_model = self.load_stt_model()

    def load_stt_model(self):
        """Load the speech-to-text model and compile it for optimal performance."""
        model = gigaam.load_model(self.model, self.fp16_encoder, self.device)
        return torch.compile(model, mode="max-autotune")

    def __repr__(self) -> str:
        """Output the model when print(model)."""
        return repr(self.opt_model)

if __name__ == "__main__":
    model = STTModel(model="ctc", fp16_encoder=True, device="mps")
    print(model)
