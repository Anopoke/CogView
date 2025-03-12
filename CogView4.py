from typing import Dict, Any

import torch
from diffusers import CogView4Pipeline, CogView4Transformer2DModel, AutoencoderKL, FlowMatchEulerDiscreteScheduler
from transformers import GlmModel, BitsAndBytesConfig, PreTrainedTokenizerFast


# CogView4
class CogView4(CogView4Pipeline):
    def __init__(self, model_path: str, torch_dtype: torch.dtype, device_map: Dict[str, str]):
        """
        Initialize the CogView4 model by loading its components and passing them to the parent class constructor.

        :param model_path: The path to the directory containing the model files.
        :param torch_dtype: The data type for PyTorch tensors.
        :param device_map: A dictionary mapping component names to device strings.
        """
        self.model_path = model_path
        self.torch_dtype = torch_dtype
        self.device_map = device_map
        components = self.components()
        super().__init__(
            text_encoder=components['text_encoder'],
            tokenizer=components['tokenizer'],
            transformer=components['transformer'],
            vae=components['vae'],
            scheduler=components['scheduler']
        )

    def components(self) -> Dict[str, Any]:
        """
        Load and return the components of the CogView4 model.

        :return: A dictionary containing the model components.
        """
        # TextEncoder
        text_encoder = GlmModel.from_pretrained(
            f"{self.model_path}/text_encoder",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                # bnb_4bit_quant_type="nf4",
                # bnb_4bit_compute_dtype=TORCH_DTYPE,
                # bnb_4bit_use_double_quant=True
            ),
            device_map=self.device_map["text_encoder"],
            torch_dtype=self.torch_dtype,
        )
        # Tokenizer
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            f"{self.model_path}/tokenizer",
            device_map=self.device_map["tokenizer"],
            torch_dtype=self.torch_dtype,
        )
        # Transformer
        transformer = CogView4Transformer2DModel.from_pretrained(
            f"{self.model_path}/transformer",
            # quantization_config=BitsAndBytesConfig(
            #     bnb_4bit_quant_type="nf4",
            #     bnb_4bit_compute_dtype=self.torch_dtype,
            #     bnb_4bit_use_double_quant=True
            # ),
            torch_dtype=self.torch_dtype,
            device_map=self.device_map["transformer"],
        )
        # VAE
        vae = AutoencoderKL.from_pretrained(
            f"{self.model_path}/vae",
            torch_dtype=self.torch_dtype,
            device_map=self.device_map["vae"]
        )
        # Scheduler
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            f"{self.model_path}/scheduler",
            subfolder="scheduler_config",
            torch_dtype=self.torch_dtype,
            device_map=self.device_map["scheduler"]
        )

        # Return the components
        return {
            'text_encoder': text_encoder,
            'tokenizer': tokenizer,
            'transformer': transformer,
            'vae': vae,
            'scheduler': scheduler
        }
