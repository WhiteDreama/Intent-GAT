import os
import sys
import time  # 新增：用于reset retry的延迟
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
                # === [Linux修复] 添加reset重试逻辑，处理MetaDrive navigation bug ===
                kwargs = data or {}
                max_retries = 5
                reset_success = False
                
                for attempt in range(max_retries):
                    try:
                        # 如果有seed参数且不是第一次尝试，调整seed避免重复相同的bug场景
                        if "seed" in kwargs and attempt > 0:
                            original_seed = kwargs.get("seed", 0)
                            # 使用较小的增量(+100)并确保在MetaDrive有效范围内[43, 5043)
                            adjusted_seed = original_seed + 100 * attempt
                            # Wrap到有效范围：如果超出5043，从43开始循环
                            if adjusted_seed >= 5043:
                                adjusted_seed = 43 + (adjusted_seed - 43) % 5000
                            kwargs["seed"] = adjusted_seed
                            print(f"[Worker] Reset retry {attempt+1}/{max_retries} with adjusted seed: {kwargs['seed']}")
                        
                        obs = env.reset(**kwargs)
                        if isinstance(obs, tuple) and len(obs) == 2:
                            obs = obs[0]
                        remote.send(obs)
                        reset_success = True
                        break
                    except (ValueError, KeyError, IndexError, AssertionError) as e:
                        if attempt < max_retries - 1:
                            print(f"[Worker] Reset failed (attempt {attempt+1}/{max_retries}): {e}")
                            time.sleep(0.1)  # 短暂延迟再重试
                            continue
                        else:
                            # 所有重试都失败，发送异常给主进程
                            error_msg = f"Reset failed after {max_retries} attempts: {e}"
                            print(f"[Worker] {error_msg}")
                            remote.send(Exception(error_msg))
                            reset_success = True  # 标记为已处理
                            break
                
                if not reset_success:
                    remote.send(Exception("Unknown reset failure"))
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
        
        # === [Linux修复] 主进程处理worker发回的Exception或EOFError ===
        results = []
        for i, remote in enumerate(self.remotes):
            try:
                obs = remote.recv()
                # 检查worker是否返回了Exception对象
                if isinstance(obs, Exception):
                    print(f"[MultiProcEnv] Worker {i} reset failed: {obs}")
                    # 尝试用调整后的seed再次reset，确保在有效范围[43, 5043)内
                    original_seed = seeds[i] if seeds[i] is not None else 43
                    adjusted_seed = original_seed + 500
                    # Wrap到有效范围
                    if adjusted_seed >= 5043:
                        adjusted_seed = 43 + (adjusted_seed - 43) % 5000
                    print(f"[MultiProcEnv] Retrying worker {i} with seed {adjusted_seed}...")
                    remote.send(("reset", {"seed": adjusted_seed}))
                    obs = remote.recv()
                    if isinstance(obs, Exception):
                        raise RuntimeError(f"Worker {i} reset failed twice: {obs}")
                results.append(obs)
            except (EOFError, BrokenPipeError) as e:
                # Worker进程彻底崩溃
                raise RuntimeError(f"Worker {i} crashed during reset: {e}. Check worker logs above.")
        
        return results

    def step(self, actions: Sequence[Any]) -> Tuple[List[Any], List[Any], List[Any], List[Any]]:
        if len(actions) != self.num_envs:
            raise ValueError("actions length must equal num_envs")
        for remote, act in zip(self.remotes, actions):
            remote.send(("step", act))
        results = []
        for i, remote in enumerate(self.remotes):
            try:
                result = remote.recv()
                results.append(result)
            except (EOFError, BrokenPipeError, ConnectionResetError) as e:
                raise RuntimeError(
                    f"Worker {i} crashed during step: {e}. "
                    f"Check worker logs above for traceback."
                ) from e
        obs, rews, dones, infos = zip(*results)
        return list(obs), list(rews), list(dones), list(infos)

    def env_method(self, method_name: str, *args, **kwargs) -> List[Any]:
        for remote in self.remotes:
            remote.send(("call", (method_name, args, kwargs)))
        results = []
        for i, remote in enumerate(self.remotes):
            try:
                results.append(remote.recv())
            except (EOFError, BrokenPipeError, ConnectionResetError) as e:
                raise RuntimeError(
                    f"Worker {i} crashed during env_method('{method_name}'): {e}. "
                    f"Check worker logs above for traceback."
                ) from e
        return results

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