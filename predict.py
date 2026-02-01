# Prediction interface for Replicate
from cog import BasePredictor, Input, Path
import os
import torch
import numpy as np
from PIL import Image

# Import your custom modules
# NOTE: Ensure the 'qwenimage' folder is present in this directory!
try:
    from qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
    from qwenimage.transformer_qwenimage import QwenImageTransformer2DModel
    from qwenimage.qwen_fa3_processor import QwenDoubleStreamAttnProcessorFA3
except ImportError:
    print("WARNING: 'qwenimage' module not found. Please ensure the local folder is copied here.")

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16

        print("Loading Qwen Image Edit Pipeline...")
        # Note: We rely on the qwenimage module being present. 
        # If it uses relative paths for 'prithivMLmods', they should be downloaded by HF Hub automatically.
        try:
            self.pipe = QwenImageEditPlusPipeline.from_pretrained(
                "Qwen/Qwen-Image-Edit-2509",
                transformer=QwenImageTransformer2DModel.from_pretrained(
                    "prithivMLmods/Qwen-Image-Edit-Rapid-AIO-V19",
                    torch_dtype=self.dtype,
                    device_map='cuda'
                ),
                torch_dtype=self.dtype
            ).to(self.device)
            
            try:
                self.pipe.transformer.set_attn_processor(QwenDoubleStreamAttnProcessorFA3())
                print("Flash Attention 3 Processor set successfully.")
            except Exception as e:
                print(f"Warning: Could not set FA3 processor: {e}")
                
        except Exception as e:
            print(f"Error loading model: {e}")
            # If we fail, it might be because 'qwenimage' isn't finding its imports.
            raise e

        # Load Adapters
        self.ADAPTER_SPECS = {
            "Extract-Outfit": {
                "repo": "prithivMLmods/QIE-2511-Extract-Outfit",
                "weights": "QIE-2511-Extract-Outfit-4200.safetensors",
                "adapter_name": "extract-outfit"
            },
        }
        self.LOADED_ADAPTERS = set()

    def update_dimensions(self, image):
        original_width, original_height = image.size
        # Limit to 1024 max dimension for GPU memory safety
        if original_width > original_height:
            new_width = 1024
            aspect_ratio = original_height / original_width
            new_height = int(new_width * aspect_ratio)
        else:
            new_height = 1024
            aspect_ratio = original_width / original_height
            new_width = int(new_height * aspect_ratio)
        
        # Must be divisible by 8 for VAE
        return (new_width // 8) * 8, (new_height // 8) * 8

    def predict(
        self,
        image: Path = Input(description="Input clothing photo"),
        seed: int = Input(description="Random seed", default=42),
        guidance_scale: float = Input(description="Guidance scale", default=3.0),
        steps: int = Input(description="Inference steps", default=25),
    ) -> Path:
        """Run a single prediction on the model"""
        
        # 1. Process Input
        pil_image = Image.open(image).convert("RGB")
        
        # 2. Load Adapter if needed (Extract-Outfit default)
        spec = self.ADAPTER_SPECS["Extract-Outfit"]
        adapter_name = spec["adapter_name"]
        
        if adapter_name not in self.LOADED_ADAPTERS:
            print(f"Loading Adapter: {adapter_name}")
            self.pipe.load_lora_weights(
                spec["repo"], 
                weight_name=spec["weights"], 
                adapter_name=adapter_name
            )
            self.LOADED_ADAPTERS.add(adapter_name)
        
        self.pipe.set_adapters([adapter_name], adapter_weights=[1.0])
        
        # 3. Setup Generator
        # Handle random seed if user passed 0 or None, though Cog passes default 42 usually.
        # If we really want random:
        if seed is None or seed < 0:
            seed = int.from_bytes(os.urandom(2), "big")
            
        print(f"Using seed: {seed}")
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # 4. Infer
        width, height = self.update_dimensions(pil_image)
        
        prompt = "Extract the clothing and create a flat mockup."
        negative_prompt = "worst quality, low quality, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, jpeg artifacts, signature, watermark, username, blurry"

        result = self.pipe(
            image=[pil_image],
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            generator=generator,
            true_cfg_scale=guidance_scale,
        ).images[0]

        # 5. Save and Return
        output_path = "/tmp/output.png"
        result.save(output_path)
        return Path(output_path)
