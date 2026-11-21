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
    preload_delay: int = 2  # 异步预加载延迟步数：路由后 N 步才标记加载完成
    heat: dict[str, float] = field(default_factory=dict)
    loaded: set[str] = field(default_factory=set)
    last_used_step: dict[str, int] = field(default_factory=dict)
    pending_preload: deque = field(default_factory=deque)  # 元素: (expert_id, ready_step)

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

    def _register_use(self, expert_id: str, step: int) -> bool:
        self.heat[expert_id] = self.heat.get(expert_id, 0.0) + 1.0
        self.last_used_step[expert_id] = step
        pending_ids = {eid for eid, _ in self.pending_preload}
        if expert_id not in self.loaded and expert_id not in pending_ids:
            self.pending_preload.append((expert_id, step + self.preload_delay))
            return True
        return False

    def register_route(self, domain_ids, step: int) -> list[str]:
        """路由判出领域后提升热度并预加载对应垂直专家。"""
        queued = []
        for domain in domain_ids:
            expert_id = self.vertical_id(int(domain))
            if self._register_use(expert_id, step):
                queued.append(expert_id)
        return queued

    def register_general_use(self, expert_index: int, step: int) -> bool:
        """记录通用专家使用热度，并在已卸载时排队重新加载。"""
        if not 0 <= expert_index < self.num_general:
            raise IndexError(f"通用专家索引越界: {expert_index}")
        return self._register_use(self.general_id(expert_index), step)

    def drain_preload(self, current_step: int | None = None) -> list[str]:
        """处理已到期的异步预加载：到期专家标记为已加载。

        Args:
            current_step: 当前步；为 None 时（兼容旧调用）直接完成全部队列。
        """
        done = []
        remaining = deque()
        while self.pending_preload:
            eid, ready_step = self.pending_preload.popleft()
            if current_step is None or ready_step <= current_step:
                self.loaded.add(eid)
                done.append(eid)
            else:
                remaining.append((eid, ready_step))
        self.pending_preload = remaining
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
