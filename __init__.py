from .egregora_audio_super_resolution import EgregoraAudioSuperResolution
from .egregora_fat_llama_gpu import EgregoraFatLlamaGPU
from .egregora_fat_llama_cpu import EgregoraFatLlamaCPU

NODE_CLASS_MAPPINGS = {
    "EgregoraAudioUpscaler": EgregoraAudioSuperResolution,     # FlashSR
    "EgregoraFatLlamaGPU": EgregoraFatLlamaGPU,               # GPU (CuPy)
    "EgregoraFatLlamaCPU": EgregoraFatLlamaCPU,               # CPU (FFTW)
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EgregoraAudioUpscaler": "🎧 Audio Super Resolution (FlashSR)",
    "EgregoraFatLlamaGPU": "🎛️ Spectral Enhance (Fat Llama — GPU)",
    "EgregoraFatLlamaCPU": "🎛️ Spectral Enhance (Fat Llama — CPU/FFTW)",
}