"""专家热度调度与可插拔异步搬运。

默认不搬运模型权重，只追踪加载状态；要执行真实的 CPU/GPU 或存储搬运，
调用方必须同时提供 ``preload_fn`` 与 ``unload_fn``。回调在线程池中串行执行，
``ensure_loaded`` 会等待对应搬运完成，避免在专家仍未加载时进入前向。
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable


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
    preload_delay: int = 2  # 预加载调度的最小逻辑延迟步数
    preload_fn: Callable[[str], None] | None = field(default=None, repr=False)
    unload_fn: Callable[[str], None] | None = field(default=None, repr=False)
    heat: dict[str, float] = field(default_factory=dict)
    loaded: set[str] = field(default_factory=set)
    last_used_step: dict[str, int] = field(default_factory=dict)
    pending_preload: deque = field(default_factory=deque)  # 元素: (expert_id, ready_step)
    _executor: ThreadPoolExecutor | None = field(default=None, init=False, repr=False)
    _preload_futures: dict[str, Future] = field(default_factory=dict, init=False, repr=False)
    _unload_futures: dict[str, Future] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        if self.preload_delay < 0:
            raise ValueError("preload_delay 不得为负。")
        if (self.preload_fn is None) != (self.unload_fn is None):
            raise ValueError("实际专家搬运需同时提供 preload_fn 和 unload_fn。")
        self.cold_start_all()

    def _get_executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            # 串行执行同一管理器的模块搬运，避免卸载/重新加载竞争。
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="noild-expert-io")
        return self._executor

    def close(self, wait: bool = True) -> None:
        """关闭后台搬运线程；建议在释放模型时调用。搬运错误会在 wait=True 时传播。"""
        if self._executor is not None:
            self._executor.shutdown(wait=wait)
            if wait:
                self._executor = None
                futures = list(self._preload_futures.values()) + list(self._unload_futures.values())
                self._preload_futures.clear()
                self._unload_futures.clear()
                for future in futures:
                    future.result()

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
            if self.preload_fn is not None:
                self._preload_futures[expert_id] = self._get_executor().submit(
                    self.preload_fn, expert_id
                )
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
            if current_step is not None and ready_step > current_step:
                remaining.append((eid, ready_step))
                continue
            future = self._preload_futures.get(eid)
            if future is not None and not future.done():
                remaining.append((eid, ready_step))
                continue
            if future is not None:
                # Propagate transfer errors instead of falsely marking the expert loaded.
                future.result()
                self._preload_futures.pop(eid, None)
            unload_future = self._unload_futures.pop(eid, None)
            if unload_future is not None:
                unload_future.result()
            self.loaded.add(eid)
            done.append(eid)
        self.pending_preload = remaining
        return done

    def ensure_loaded(self, expert_id: str) -> None:
        """阻塞至某专家真实搬运完成；无搬运回调时只维护兼容状态。"""
        if expert_id in self.loaded:
            return
        if expert_id not in {eid for eid, _ in self.pending_preload}:
            if self.preload_fn is not None:
                self._preload_futures[expert_id] = self._get_executor().submit(
                    self.preload_fn, expert_id
                )
            self.pending_preload.append((expert_id, -10**18))
        future = self._preload_futures.get(expert_id)
        if future is not None:
            future.result()
            self._preload_futures.pop(expert_id, None)
        unload_future = self._unload_futures.pop(expert_id, None)
        if unload_future is not None:
            unload_future.result()
        self.pending_preload = deque(
            (eid, ready) for eid, ready in self.pending_preload if eid != expert_id
        )
        self.loaded.add(expert_id)

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
            if self.unload_fn is not None:
                self._unload_futures[eid] = self._get_executor().submit(self.unload_fn, eid)
            unloaded.append(eid)
        return unloaded

    def is_loaded(self, expert_id: str) -> bool:
        return expert_id in self.loaded
