import os
import multiprocessing as mp
from typing import Any, Callable, List, Optional, Sequence, Tuple


class _CloudpickleWrapper:
    """Minimal wrapper for making env factories picklable on Windows spawn."""

    def __init__(self, fn: Callable[[], Any]):
        self.fn = fn

    def __call__(self):
        return self.fn()


def _worker(remote, env_fn_wrapper: _CloudpickleWrapper):
    env = env_fn_wrapper()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "reset":
                # data can be dict of kwargs or None
                kwargs = data or {}
                obs = env.reset(**kwargs)
                # GraphEnvWrapper.reset returns (obs, info)
                if isinstance(obs, tuple) and len(obs) == 2:
                    obs = obs[0]
                remote.send(obs)
            elif cmd == "step":
                obs, rew, done, info = env.step(data)
                remote.send((obs, rew, done, info))
            elif cmd == "call":
                method_name, args, kwargs = data
                result = getattr(env, method_name)(*args, **kwargs)
                remote.send(result)
            elif cmd == "close":
                try:
                    env.close()
                except Exception:
                    pass
                remote.close()
                break
            else:
                raise RuntimeError(f"Unknown cmd: {cmd}")
    except EOFError:
        # Parent died
        try:
            env.close()
        except Exception:
            pass


class MultiProcEnv:
    """A tiny multi-process env manager for non-Gym, dict-based multi-agent observations.

    - Each worker holds one GraphEnvWrapper instance.
    - Main process sends per-env action dicts, receives per-env (obs, reward, done, info) dicts.

    This intentionally does NOT try to stack/flatten observations.
    """

    def __init__(self, env_fns: Sequence[Callable[[], Any]]):
        self.num_envs = len(env_fns)
        if self.num_envs < 1:
            raise ValueError("env_fns must be non-empty")

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
