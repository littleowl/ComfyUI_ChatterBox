import os
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Optional
import tempfile

# Import directly from the chatterbox package
from .local_chatterbox.chatterbox.tts import ChatterboxTTS
from .local_chatterbox.chatterbox.vc import ChatterboxVC

from comfy.utils import ProgressBar

# Monkey patch torch.load to use MPS or CPU if map_location is not specified
original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    if 'map_location' not in kwargs:
        # Determine the appropriate device (MPS for Mac, else CPU)
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        kwargs['map_location'] = torch.device(device)
    return original_torch_load(*args, **kwargs)

torch.load = patched_torch_load

class ChatterboxTTSModel:
    def __init__(self, device: str = "cuda"):
        self.model = None
        self.device = device
        
    def load_model(self):
        self.model = ChatterboxTTS.from_pretrained(device=self.device)
        
    @property
    def sr(self):
        return self.model.sr if self.model else 24000
    
    def unload_model(self):
        self.model = None
        self.device = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache() # Clear CUDA cache if possible
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
            
    def generate(
        self,
        text: str,
        audio_prompt_path: Optional[str] = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8
        ) -> torch.Tensor:
        
        if self.model is None:
            self.load_model()
        
        wav = self.model.generate(
            text=text,
            audio_prompt_path=audio_prompt_path,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature,
        )
        return wav
    
class ChatterboxVCModel:
    def __init__(self, device: str = "cuda"):
        self.model = None
        self.device = device
        
    @property
    def sr(self):
        return self.model.sr if self.model else 24000
        
    def load_model(self):
        self.model = ChatterboxVC.from_pretrained(device=self.device)
        
    def unload_model(self):
        self.model = None
        self.device = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache() # Clear CUDA cache if possible
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
            
    def generate(
        self,
        audio,
        target_voice_path=None,
    ) -> torch.Tensor:
        if self.model is None:
            self.load_model()
        
        converted_wav = self.model.generate(
            audio=audio,
            target_voice_path=target_voice_path,
            )
        return converted_wav


class LoadChatterboxTTSModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "use_cpu": ("BOOLEAN", {"default": False}),
                "differed_loading": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("CHATTERBOX_TTS",)
    RETURN_NAMES = ("tts_model",)
    FUNCTION = "load_model"
    CATEGORY = "ChatterboxTTS"

    def load_model(self, use_cpu, differed_loading):
        device = "cpu" if use_cpu else ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
        tts_model = ChatterboxTTSModel(device=device)
        
        if not differed_loading:
            tts_model.load_model()
        return (tts_model,)

    def IS_CHANGED(s, use_cpu, differed_loading):
            return "use_cpu: {}, differed_loading: {}".format(use_cpu, differed_loading)


class LoadChatterboxVCModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "use_cpu": ("BOOLEAN", {"default": False}),
                "differed_loading": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("CHATTERBOX_VC",)
    RETURN_NAMES = ("vc_model",)
    FUNCTION = "load_model"
    CATEGORY = "ChatterboxTTS"

    def load_model(self, use_cpu, differed_loading):
        device = "cpu" if use_cpu else ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

        vc_model = ChatterboxVCModel(device=device)
        
        if not differed_loading:
            vc_model.load_model()
            
        return (vc_model,)
    
    def IS_CHANGED(s, use_cpu, differed_loading):
        return "use_cpu: {}, differed_loading: {}".format(use_cpu, differed_loading)


class AudioNodeBase:
    """Base class for audio nodes with common utilities."""
    
    @staticmethod
    def create_empty_tensor(audio, frame_rate, height, width, channels=None):
        """Create an empty tensor with dimensions based on audio duration."""
        audio_duration = audio['waveform'].shape[-1] / audio['sample_rate']
        num_frames = int(audio_duration * frame_rate)
        if channels is None:
            return torch.zeros((num_frames, height, width), dtype=torch.float32)
        else:
            return torch.zeros((num_frames, height, width, channels), dtype=torch.float32)

# Text-to-Speech node
class FL_ChatterboxTTSNode(AudioNodeBase):
    """
    ComfyUI node for Chatterbox Text-to-Speech functionality.
    """
    _tts_model = None
    _tts_device = None
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tts_model": ("CHATTERBOX_TTS",),
                "text": ("STRING", {"multiline": True, "default": "Hello, this is a test."}),
                "exaggeration": ("FLOAT", {"default": 0.5, "min": 0.25, "max": 2.0, "step": 0.05}),
                "cfg_weight": ("FLOAT", {"default": 0.5, "min": 0.2, "max": 1.0, "step": 0.05}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.05, "max": 5.0, "step": 0.05}),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "audio_prompt": ("AUDIO",),
            }
        }
    
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "message")
    FUNCTION = "generate_speech"
    CATEGORY = "ChatterBox"
    
    def generate_speech(
        self, 
        tts_model: ChatterboxTTSModel,
        text: str, 
        exaggeration: float, 
        cfg_weight: float, 
        temperature: float, 
        keep_model_loaded: bool = True,
        audio_prompt: Optional[torch.Tensor] = None, 
        ):
        """
        Generate speech from text.
        
        Args:
            text: The text to convert to speech.
            exaggeration: Controls emotion intensity (0.25-2.0).
            cfg_weight: Controls pace/classifier-free guidance (0.2-1.0).
            temperature: Controls randomness in generation (0.05-5.0).
            audio_prompt: AUDIO object containing the reference voice for TTS voice cloning.
            use_cpu: If True, forces CPU usage even if CUDA is available.
            keep_model_loaded: If True, keeps the model loaded in memory after generation.
            
        Returns:
            Tuple of (audio, message)
        """
        # Determine device to use
        # device = "cpu" if use_cpu else ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
        # if use_cpu:
        #     message = "Using CPU for inference (GPU disabled)"
        # elif torch.backends.mps.is_available() and device == "mps":
        #      message = "Using MPS (Mac GPU) for inference"
        # elif torch.cuda.is_available() and device == "cuda":
        #      message = "Using CUDA (NVIDIA GPU) for inference"
        # else:
        #     message = f"Using {device} for inference" # Should be CPU if no GPU found
        
        # Create temporary files for any audio inputs
        
        message = ""
        
        temp_files = []
        
        # Create a temporary file for the audio prompt if provided
        audio_prompt_path = None
        if audio_prompt is not None:
            try:
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_prompt:
                    audio_prompt_path = temp_prompt.name
                    temp_files.append(audio_prompt_path)
                
                # Save the audio prompt to the temporary file
                prompt_waveform = audio_prompt['waveform'].squeeze(0)
                torchaudio.save(audio_prompt_path, prompt_waveform, audio_prompt['sample_rate'])
                message += f"\nUsing provided audio prompt for voice cloning: {audio_prompt_path}"
                
                # Debug: Check if the file exists and has content
                if os.path.exists(audio_prompt_path):
                    file_size = os.path.getsize(audio_prompt_path)
                    message += f"\nAudio prompt file created successfully: {file_size} bytes"
                else:
                    message += f"\nWarning: Audio prompt file was not created properly"
            except Exception as e:
                message += f"\nError creating audio prompt file: {str(e)}"
                audio_prompt_path = None
        
        wav = None # Initialize wav to None
        audio_data = {"waveform": torch.zeros((1, 2, 1)), "sample_rate": 16000} # Initialize with empty audio
        pbar = ProgressBar(100) # Simple progress bar for overall process
        try:
            # Generate speech
            message += f"\nGenerating speech for: {text[:50]}..." if len(text) > 50 else f"\nGenerating speech for: {text}"
            if audio_prompt_path:
                message += f"\nUsing audio prompt: {audio_prompt_path}"
            
            tts_model.load_model()
            
            pbar.update_absolute(60) # Indicate generation started
            wav = tts_model.generate(
                text=text,
                audio_prompt_path=audio_prompt_path,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
            )
            pbar.update_absolute(90) # Indicate generation finished
            
            audio_data = {
                "waveform": wav.unsqueeze(0),  # Add batch dimension
                "sample_rate": tts_model.model.sr
            }
            message += f"\nSpeech generated successfully"
            # return (audio_data, message)
            
        except Exception as e:
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            if not keep_model_loaded:
                tts_model.unload_model()
            raise
        
        finally:
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            if not keep_model_loaded:
                tts_model.unload_model()
            
        
        pbar.update_absolute(100) # Ensure progress bar completes on success or error

        # Create audio data structure for the output
        audio_data = {
            "waveform": wav.unsqueeze(0),  # Add batch dimension
            "sample_rate": tts_model.sr
        }
        
        message += f"\nSpeech generated successfully"
        pbar.update_absolute(100) # Ensure progress bar completes on success
        
        return (audio_data, message)

# Voice Conversion node
class FL_ChatterboxVCNode(AudioNodeBase):
    """
    ComfyUI node for Chatterbox Voice Conversion functionality.
    """
    _vc_model = None
    _vc_device = None
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vc_model": ("CHATTERBOX_VC",),
                "input_audio": ("AUDIO",),
                "target_voice": ("AUDIO",),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
            },
        }
    
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "message")
    FUNCTION = "convert_voice"
    CATEGORY = "ChatterBox"
    
    def convert_voice(
        self, 
        vc_model, 
        input_audio, 
        target_voice, 
        keep_model_loaded=False
        ):
        """
        Convert the voice in an audio file to match a target voice.
        
        Args:
            input_audio: AUDIO object containing the audio to convert.
            target_voice: AUDIO object containing the target voice.
            use_cpu: If True, forces CPU usage even if CUDA is available.
            keep_model_loaded: If True, keeps the model loaded in memory after conversion.
            
        Returns:
            Tuple of (audio, message)
        """
        
        message = ""
        
        # Create temporary files for the audio inputs
        import tempfile
        temp_files = []
        
        # Create a temporary file for the input audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_input:
            input_audio_path = temp_input.name
            temp_files.append(input_audio_path)
        
        # Save the input audio to the temporary file
        input_waveform = input_audio['waveform'].squeeze(0)
        torchaudio.save(input_audio_path, input_waveform, input_audio['sample_rate'])
        
        # Create a temporary file for the target voice
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_target:
            target_voice_path = temp_target.name
            temp_files.append(target_voice_path)
        
        # Save the target voice to the temporary file
        target_waveform = target_voice['waveform'].squeeze(0)
        torchaudio.save(target_voice_path, target_waveform, target_voice['sample_rate'])
        
        pbar = ProgressBar(100) # Simple progress bar for overall process
        try:

            # Convert voice
            message += f"\nConverting voice to match target voice"
            
            vc_model.load_model()
            
            pbar.update_absolute(60) # Indicate conversion started
            converted_wav = vc_model.generate(
                audio=input_audio_path,
                target_voice_path=target_voice_path,
            )
            pbar.update_absolute(90) # Indicate conversion finished
            
        except Exception as e:
            message += f"\nAn unexpected error occurred during VC: {str(e)}"

            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            if not keep_model_loaded:
                vc_model.unload_model()
            raise
        
        finally:
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            if not keep_model_loaded:
                vc_model.unload_model()

        # Create audio data structure for the output
        audio_data = {
            "waveform": converted_wav.unsqueeze(0),  # Add batch dimension
            "sample_rate": vc_model.sr
        }
        
        message += f"\nVoice converted successfully"
        pbar.update_absolute(100) # Ensure progress bar completes on success
        
        return (audio_data, message)

# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "LoadChatterboxTTSModel": LoadChatterboxTTSModel,
    "LoadChatterboxVCModel": LoadChatterboxVCModel,
    "FL_ChatterboxTTS": FL_ChatterboxTTSNode,
    "FL_ChatterboxVC": FL_ChatterboxVCNode,
}

# Display names for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadChatterboxTTSModel": "Load Chatterbox TTS Model",
    "LoadChatterboxVCModel": "Load Chatterbox VC Model",
    "FL_ChatterboxTTS": "FL Chatterbox TTS",
    "FL_ChatterboxVC": "FL Chatterbox VC",
}