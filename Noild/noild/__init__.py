"""Noild — Role-Specialized Cascaded Experts（独立稀疏层，Apache 2.0）。

不依赖 Nooht；其他混合架构可独立接入。
版本：v0.1.0-dev（与主仓库一致）。
"""

from .config import CASCADE_PATHS, NoildConfig, get_cascade_path, scale_note
from .experts import (
    DomainAdapter,
    DraftExpert,
    ExpertMLP,
    GeneralExpert,
    MergeExpert,
    VerticalExpert,
)
from .layer import NoildLayer, NoildOutput
from .loading import ExpertHeatManager
from .mtp import MTPHead, build_mtp_head
from .router import DomainRouter, RejudgeController, RouterOutput
from .speculative import (
    draft_acceptance_rate,
    speculative_greedy_decode,
    verify_draft,
)
from .training import (
    STAGED_TRAINING_PLAN,
    alignment_check,
    draft_loss,
    future_token_targets,
    load_balance_loss,
    mean_layer_losses,
    mtp_loss,
    set_trainable,
    train_noild,
    upcycle_init,
)

__all__ = [
    "CASCADE_PATHS",
    "NoildConfig",
    "get_cascade_path",
    "scale_note",
    "DomainAdapter",
    "DraftExpert",
    "ExpertMLP",
    "GeneralExpert",
    "MergeExpert",
    "VerticalExpert",
    "NoildLayer",
    "NoildOutput",
    "ExpertHeatManager",
    "MTPHead",
    "build_mtp_head",
    "DomainRouter",
    "RejudgeController",
    "RouterOutput",
    "STAGED_TRAINING_PLAN",
    "alignment_check",
    "draft_loss",
    "future_token_targets",
    "load_balance_loss",
    "mean_layer_losses",
    "mtp_loss",
    "set_trainable",
    "train_noild",
    "upcycle_init",
    "draft_acceptance_rate",
    "speculative_greedy_decode",
    "verify_draft",
]

__version__ = "0.1.0.dev0"
