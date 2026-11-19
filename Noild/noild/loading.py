"""加载策略（基础版）：冷启动全量加载，按热度卸载，异步预加载。

对应任务包第四节「加载策略」：
- 默认冷启动全量加载，再按热度卸载冷门专家；
- 路由判出领域后异步预加载垂直专家；
- 本阶段只做基础版，深度优化不在范围。

本模块只管理「哪些专家已加载/热度」，不负责真正的显存搬运；
实际加载/卸载由调用方根据 `loaded` 集合执行。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class ExpertHeatManager:
    """专家热度管理器。

    Args:
        num_vertical: 垂直专家数。
        num_general: 通用专家数。
        heat_decay: 每步热度衰减系数（0~1）。
        unload_threshold: 热度低于此值且冷却期已过则卸载。
        cooldown_steps: 卸载冷却步数（刚用过的专家不立即卸载）。
        min_loaded: 至少保留加载的专家数（防止全卸载）。
    """

    num_vertical: int
    num_general: int = 2
    heat_decay: float = 0.95
    unload_threshold: float = 0.05
    cooldown_steps: int = 10
    min_loaded: int = 1
    heat: dict[str, float] = field(default_factory=dict)
    loaded: set[str] = field(default_factory=set)
    last_used_step: dict[str, int] = field(default_factory=dict)
    pending_preload: deque = field(default_factory=deque)

    def __post_init__(self):
        self.cold_start_all()

    # -- 专家 id 命名：vertical:{domain} / general:{i} --
    def vertical_id(self, domain: int) -> str:
        return f"vertical:{domain}"

    def general_id(self, i: int) -> str:
        return f"general:{i}"

    def all_expert_ids(self) -> list[str]:
        return [self.vertical_id(d) for d in range(self.num_vertical)] + [
            self.general_id(i) for i in range(self.num_general)
        ]

    def cold_start_all(self) -> None:
        """冷启动：全量加载。"""
        self.loaded = set(self.all_expert_ids())
        for eid in self.loaded:
            self.heat.setdefault(eid, 1.0)

    def register_route(self, domain_ids, step: int) -> list[str]:
        """路由判出领域后：提升热度，并异步预加载对应垂直专家。

        Returns:
            新加入预加载队列的专家 id 列表。
        """
        queued = []
        for d in domain_ids:
            eid = self.vertical_id(int(d))
            self.heat[eid] = self.heat.get(eid, 0.0) + 1.0
            self.last_used_step[eid] = step
            if eid not in self.loaded and eid not in self.pending_preload:
                self.pending_preload.append(eid)
                queued.append(eid)
        return queued

    def drain_preload(self) -> list[str]:
        """模拟异步预加载完成：把队列中的专家标记为已加载。"""
        done = []
        while self.pending_preload:
            eid = self.pending_preload.popleft()
            self.loaded.add(eid)
            done.append(eid)
        return done

    def step(self, current_step: int) -> list[str]:
        """每步：衰减热度，卸载冷门专家。返回本次卸载的专家 id 列表。"""
        for eid in list(self.heat):
            self.heat[eid] *= self.heat_decay
        unloaded = []
        for eid in sorted(self.loaded):
            if len(self.loaded) <= self.min_loaded:
                break
            if self.heat.get(eid, 0.0) >= self.unload_threshold:
                continue
            if current_step - self.last_used_step.get(eid, -10**9) < self.cooldown_steps:
                continue
            self.loaded.discard(eid)
            unloaded.append(eid)
        return unloaded

    def is_loaded(self, expert_id: str) -> bool:
        return expert_id in self.loaded
