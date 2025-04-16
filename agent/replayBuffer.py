


from typing import Any, Optional, Union, NamedTuple
import numpy as np
import psutil
import warnings
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import BaseBuffer
# from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize


class ReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    next_action_masks: th.Tensor


class ReplayBuffer(BaseBuffer):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3, modified to store action masks.

    :param buffer_size: Max number of elements in the buffer
    :param observation_space: Observation space
    :param action_space: Action space (assumed discrete for action masks)
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces memory usage by almost a factor of two,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
        Cannot be used in combination with handle_timeout_termination.
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    observations: np.ndarray
    next_observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    timeouts: np.ndarray
    next_action_masks: np.ndarray  # New attribute for action masks

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        # Adjust buffer size
        self.buffer_size = max(buffer_size // n_envs, 1)

        # Check that the replay buffer can fit into memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        # Ensure action_space is discrete since action masks are typically used with discrete actions
        if not isinstance(action_space, spaces.Discrete):
            raise ValueError("Action masks are only supported for discrete action spaces.")
        self.action_space_n = action_space.n  # Number of possible actions

        # Validate memory optimization and timeout handling compatibility
        if optimize_memory_usage and handle_timeout_termination:
            raise ValueError(
                "ReplayBuffer does not support optimize_memory_usage = True "
                "and handle_timeout_termination = True simultaneously."
            )
        self.optimize_memory_usage = optimize_memory_usage

        # Initialize existing arrays

        self.action_dim = 1

        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=observation_space.dtype)
        if not optimize_memory_usage:
            self.next_observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=observation_space.dtype)
        self.actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=self._maybe_cast_dtype(action_space.dtype)
        )
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

    

        # Initialize next_action_masks array with shape (buffer_size, n_envs, action_space.n)
        self.next_action_masks = np.zeros((self.buffer_size, self.n_envs, self.action_space_n), dtype=np.float32)

        # Memory usage check
        if psutil is not None:
            total_memory_usage: float = (
                self.observations.nbytes +
                self.actions.nbytes +
                self.rewards.nbytes +
                self.dones.nbytes +
                self.next_action_masks.nbytes  # Include next_action_masks in memory calculation
            )
            if not optimize_memory_usage:
                total_memory_usage += self.next_observations.nbytes
            if total_memory_usage > mem_available:
                total_memory_usage /= 1e9  # Convert to GB
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]]
    ) -> None:
        # Reshape for discrete observations if necessary
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape actions to handle multi-dim and discrete action spaces
        action = action.reshape((self.n_envs, self.action_dim))

        # Store observations, actions, rewards, and dones
        self.observations[self.pos] = np.array(obs)
        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs)
        else:
            self.next_observations[self.pos] = np.array(next_obs)
        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.dones[self.pos] = np.array(done)

        # Extract and store action masks from infos (default to all valid if not provided)
        next_action_masks = np.array(
            [info.get("next_action_masks", np.ones(self.action_space_n, dtype=np.float32)) for info in infos]
        )
        self.next_action_masks[self.pos] = next_action_masks

        # Handle timeouts if enabled
        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        # Update position and full status
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        """
        Sample elements from the replay buffer, including action masks.
        Custom sampling for memory-efficient variant to avoid invalid transitions.

        :param batch_size: Number of elements to sample
        :param env: Associated VecEnv to normalize observations/rewards
        :return: ReplayBufferSamples including action masks
        """
        if not self.optimize_memory_usage:
            return super().sample(batch_size=batch_size, env=env)
        # Avoid sampling the current position in memory-efficient mode
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        # Sample environment indices
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        # Get next observations based on memory optimization
        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        # Prepare data tuple including action masks
        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
            self.next_action_masks[batch_inds, env_indices, :],  # Include action masks
        )
        # Note: ReplayBufferSamples must be modified to accept next_action_masks as an additional field
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))

    @staticmethod
    def _maybe_cast_dtype(dtype: np.typing.DTypeLike) -> np.typing.DTypeLike:
        """
        Cast np.float64 to np.float32, keep other dtypes unchanged.
        See GH#1572 for more information.

        :param dtype: Original dtype
        :return: Adjusted dtype
        """
        if dtype == np.float64:
            return np.float32
        return dtype