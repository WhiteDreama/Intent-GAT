import os
import sys
import traceback  # 新增：用于打印报错堆栈
import multiprocessing as mp
from typing import Any, Callable, List, Optional, Sequence, Tuple

# === [修复关键点 1] 确保 Windows 子进程能找到 marl_project 包 ===
# 获取当前文件 (mp_env.py) 的上上级目录 (即项目根目录)
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)
# =============================================================

class _CloudpickleWrapper:
    """Minimal wrapper for making env factories picklable on Windows spawn."""

    def __init__(self, fn: Callable[[], Any]):
        self.fn = fn

    def __call__(self):
        return self.fn()


def _worker(remote, env_fn_wrapper: _CloudpickleWrapper):
    env = None
    try:
        env = env_fn_wrapper()
        while True:
            cmd, data = remote.recv()
            if cmd == "reset":
                # ... (reset 保持不变)
                kwargs = data or {}
                obs = env.reset(**kwargs)
                if isinstance(obs, tuple) and len(obs) == 2:
                    obs = obs[0]
                remote.send(obs)
            elif cmd == "step":
                # ... (step 保持不变)
                obs, rew, done, info = env.step(data)
                remote.send((obs, rew, done, info))
                
            # === [核心修复] 区分方法调用和属性获取 ===
            elif cmd == "call":
                method_name, args, kwargs = data
                target_attr = getattr(env, method_name)
                
                # 关键判断：如果是可调用的(函数/方法)，就加括号调用；否则直接返回属性值
                if callable(target_attr):
                    result = target_attr(*args, **kwargs)
                else:
                    result = target_attr
                    
                remote.send(result)
            # ==========================================
            
            elif cmd == "close":
                try:
                    env.close()
                except Exception:
                    pass
                remote.close()
                break
            else:
                raise RuntimeError(f"Unknown cmd: {cmd}")
    except Exception:
        print(f"\n[Worker Error] Process crashed! Check the trace below:")
        traceback.print_exc()
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        try:
            remote.close()
        except Exception:
            pass

class MultiProcEnv:
    """A tiny multi-process env manager for non-Gym, dict-based multi-agent observations."""

    def __init__(self, env_fns: Sequence[Callable[[], Any]]):
        self.num_envs = len(env_fns)
        if self.num_envs < 1:
            raise ValueError("env_fns must be non-empty")

        # Windows 上强制使用 spawn
        ctx = mp.get_context("spawn") if os.name == "nt" else mp.get_context()
        self.remotes: List[Any] = []
        self.ps: List[mp.Process] = []

        for fn in env_fns:
            parent_remote, worker_remote = ctx.Pipe(duplex=True)
            p = ctx.Process(target=_worker, args=(worker_remote, _CloudpickleWrapper(fn)))
            p.daemon = True
            p.start()
            worker_remote.close()
            self.remotes.append(parent_remote)
            self.ps.append(p)

        # best-effort: fetch action_space from env0
        try:
            self.action_space = self.call(0, "action_space")
        except Exception:
            self.action_space = None

    def reset(self, seeds: Optional[Sequence[Optional[int]]] = None) -> List[Any]:
        if seeds is None:
            seeds = [None] * self.num_envs
        if len(seeds) != self.num_envs:
            raise ValueError("seeds length must equal num_envs")

        for remote, seed in zip(self.remotes, seeds):
            if seed is None:
                remote.send(("reset", {}))
            else:
                remote.send(("reset", {"seed": int(seed)}))
        # 这里如果子进程挂了，recv() 就会报 EOFError 或 BrokenPipeError
        # 配合上面的 _worker 修复，你会先在控制台看到 traceback 打印
        return [remote.recv() for remote in self.remotes]

    def step(self, actions: Sequence[Any]) -> Tuple[List[Any], List[Any], List[Any], List[Any]]:
        if len(actions) != self.num_envs:
            raise ValueError("actions length must equal num_envs")
        for remote, act in zip(self.remotes, actions):
            remote.send(("step", act))
        results = [remote.recv() for remote in self.remotes]
        obs, rews, dones, infos = zip(*results)
        return list(obs), list(rews), list(dones), list(infos)

    def env_method(self, method_name: str, *args, **kwargs) -> List[Any]:
        for remote in self.remotes:
            remote.send(("call", (method_name, args, kwargs)))
        return [remote.recv() for remote in self.remotes]

    def call(self, env_idx: int, method_name: str, *args, **kwargs):
        self.remotes[env_idx].send(("call", (method_name, args, kwargs)))
        return self.remotes[env_idx].recv()

    def close(self):
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except Exception:
                pass
        for p in self.ps:
            try:
                p.join(timeout=2)
            except Exception:
                pass