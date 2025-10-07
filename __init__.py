# Core nodes you already had
from .egregora_audio_super_resolution import EgregoraAudioSuperResolution
from .egregora_fat_llama_gpu import EgregoraFatLlamaGPU
from .egregora_fat_llama_cpu import EgregoraFatLlamaCPU

# Import and merge the new modules’ mappings
# (each of these files defines NODE_CLASS_MAPPINGS / NODE_DISPLAY_NAME_MAPPINGS)
try:
    from .egregora_audio_enhance_extras import (
        NODE_CLASS_MAPPINGS as ENHANCE_MAP,
        NODE_DISPLAY_NAME_MAPPINGS as ENHANCE_NAMES,
    )
except Exception:
    ENHANCE_MAP, ENHANCE_NAMES = {}, {}

try:
    from .egregora_audio_eval_pack import (
        NODE_CLASS_MAPPINGS as EVAL_MAP,
        NODE_DISPLAY_NAME_MAPPINGS as EVAL_NAMES,
    )
except Exception:
    EVAL_MAP, EVAL_NAMES = {}, {}

try:
    from .egregora_null_test_suite import (
        NODE_CLASS_MAPPINGS as NULL_MAP,
        NODE_DISPLAY_NAME_MAPPINGS as NULL_NAMES,
    )
except Exception:
    NULL_MAP, NULL_NAMES = {}, {}

# Base mappings (FlashSR + Fat Llama) just like before
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

# Merge in the rest (Enhance Extras + Eval Pack + Null Test Suite)
NODE_CLASS_MAPPINGS.update(ENHANCE_MAP)
NODE_CLASS_MAPPINGS.update(EVAL_MAP)
NODE_CLASS_MAPPINGS.update(NULL_MAP)

NODE_DISPLAY_NAME_MAPPINGS.update(ENHANCE_NAMES)
NODE_DISPLAY_NAME_MAPPINGS.update(EVAL_NAMES)
NODE_DISPLAY_NAME_MAPPINGS.update(NULL_NAMES)
