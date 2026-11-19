"""Native 外层优化层（Lumenoo）：Jamba 黑盒 + 外层优化层。

对应任务包第二节与第三节。版本：v0.1.0-dev（与主仓库一致）。
底座 Jamba 文件（modeling_jamba.py 等）一行不改，本包只做外层编排。
"""

from .backend import Backend, benchmark, check_correctness, detect_backend, resolve_backend
from .config import LUMENOO_TIERS, NativeConfig, lumenoo_config
from .copy_mechanism import (
    CopyDecoder,
    CopySpan,
    auto_construct_copy_labels,
    find_longest_overlap,
    format_copy_instruction,
    is_copyable,
    parse_copy_instruction,
)
from .generation import GenerationResult, NativeGenerator, load_jamba_native
from .memory_compression import (
    RuleScorer,
    ScoringHead,
    Segment,
    SelectiveMemoryCompressor,
    split_segments,
)
from .protected_zone import (
    InstructionAnchor,
    PersonaAdapter,
    PersonaSchemeA,
    ProtectedZone,
    ThinkingMemoryBlock,
)

__all__ = [
    "Backend",
    "benchmark",
    "check_correctness",
    "detect_backend",
    "resolve_backend",
    "LUMENOO_TIERS",
    "NativeConfig",
    "lumenoo_config",
    "CopyDecoder",
    "CopySpan",
    "auto_construct_copy_labels",
    "find_longest_overlap",
    "format_copy_instruction",
    "is_copyable",
    "parse_copy_instruction",
    "GenerationResult",
    "NativeGenerator",
    "load_jamba_native",
    "RuleScorer",
    "ScoringHead",
    "Segment",
    "SelectiveMemoryCompressor",
    "split_segments",
    "InstructionAnchor",
    "PersonaAdapter",
    "PersonaSchemeA",
    "ProtectedZone",
    "ThinkingMemoryBlock",
]

__version__ = "0.1.0.dev0"
