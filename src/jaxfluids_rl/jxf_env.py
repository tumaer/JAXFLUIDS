from abc import abstractmethod
from collections import defaultdict
from enum import StrEnum
from typing import Any, Optional, TYPE_CHECKING
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import gymnasium as gym

from jaxfluids import InputManager, InitializationManager, SimulationManager
from jaxfluids.callbacks.base_callback import Callback
from jaxfluids.data_types import JaxFluidsBuffers
from jaxfluids.data_types.ml_buffers import ParametersSetup, CallablesSetup
from jaxfluids.data_types.information import WallClockTimes

if TYPE_CHECKING:
    from matplotlib.figure import Figure


class RenderMode(StrEnum):
    SHOW = "SHOW"
    SAVE = "SAVE"


class JAXFluidsEnv(gym.Env):
    """Generic wrapper around gym.Env which implements
    basic functionality to use JAX-Fluids in RL applications.
    """

    def __init__(
            self,
            env_config: dict,
            case_setup_dict: dict,
            numerical_setup_dict: dict,
            callbacks: Callback | list[Callback] | None = None
        ) -> None:
        
        self.env_config = env_config

        output_root = Path(env_config.get("output_dir", "outputs"))
        run_name = env_config.get("run_name", datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.output_dir = output_root / self.__class__.__name__ / run_name
        self.log_dir = self.output_dir / "logs"
        self.render_dir = self.output_dir / "renders"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.logger = self._init_logger(
            env_config.get("log_level", "INFO"),
            env_config.get("log_to_file", True),
        )
        self.log_every_steps = env_config.get("log_every_steps", 10)

        input_manager = InputManager(case_setup_dict, numerical_setup_dict)
        self.init_manager = InitializationManager(input_manager)
        self.sim_manager = SimulationManager(input_manager, callbacks)

        self.jxf_is_parallel = self.sim_manager.domain_information.is_parallel
        self.jxf_global_device_count = self.sim_manager.domain_information.global_device_count

        self.restart_file_path: Path | None = getattr(self, "restart_file_path", None)

        self.action_callable_setup = CallablesSetup()
        self.action_params_setup = ParametersSetup()

        self.state: tuple[JaxFluidsBuffers, dict[str, Any]] | None = None
        self.env_step = 0
        self.episode_idx = 0
        self.wall_clock_times = WallClockTimes()
        self._render_frame_indices: dict[str, int] = {}

        self.history: dict[str, list[Any]] = {}

        steps_per_action = env_config.get("steps_per_action", None)
        if not isinstance(steps_per_action, int):
            raise ValueError(f"steps_per_action needs to be of type int. Got {type(steps_per_action)}.")
        if steps_per_action <= 0:
            raise ValueError(f"steps_per_action needs to be larger 0. Got {steps_per_action}.")
        self.steps_per_action = steps_per_action

        self.default_action_reset: np.ndarray | None = None

        self.action_space: gym.spaces.Box | None = None
        self.observation_space: gym.spaces.Box | None = None  

        render_mode = env_config.get("render_mode", None)
        if render_mode is not None:
            render_mode = RenderMode(render_mode.upper())
        self.render_mode = render_mode

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:

        self.action_params_setup = self._convert_action_for_jxf(action)
        
        jxf_buffers, callback_dict = self._run_jxf(self.steps_per_action, None)
        self._set_state(jxf_buffers, callback_dict)

        self.env_step += 1

        observation = self._get_obs()
        reward = self._get_reward(action)
        info = self._get_info()
        terminated = self._is_terminated(action, jxf_buffers, info)
        truncated = self._is_truncated(jxf_buffers, info)

        self._log_step(observation, reward, info)

        self._after_step(
            action,
            observation,
            reward,
            terminated,
            truncated,
            info,
            jxf_buffers
        )

        return observation, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        self.env_step = 0
        self.episode_idx += 1
        self.wall_clock_times = WallClockTimes()

        self.action_params_setup = self._convert_action_for_jxf(self.default_action_reset)

        jxf_buffers = self.init_manager.initialization(
            user_restart_file_path=self.restart_file_path,
            ml_parameters=self.action_params_setup,
            ml_callables=self.action_callable_setup,
        )

        # Overwrite time control variables from the restart file
        time_control_variables = jxf_buffers.time_control_variables
        time_control_variables = time_control_variables._replace(
            physical_simulation_time=0.0,
            end_time=-1,
            end_step=-1,
        )
        jxf_buffers = jxf_buffers._replace(time_control_variables=time_control_variables)

        self._set_state(jxf_buffers, {})

        observation = self._get_obs()
        info = self._get_info()

        self._reset_history()

        self._log_reset()

        return observation, info

    def close(self) -> None:
        pass

    @abstractmethod
    def _convert_action_for_jxf(self, action: np.ndarray) -> ParametersSetup:
        pass

    @abstractmethod
    def _get_obs(self) -> np.ndarray:
        pass

    @abstractmethod
    def _get_reward(self, action: np.ndarray) -> float:
        pass

    @abstractmethod
    def _is_terminated(
            self,
            action: np.ndarray,
            jxf_buffers: JaxFluidsBuffers,
            info: dict[str, Any]
        ) -> bool:
        pass

    @abstractmethod
    def _is_truncated(
            self,
            jxf_buffers: JaxFluidsBuffers,
            info: dict[str, Any]
        ) -> bool:
        pass

    def _get_info(self) -> dict[str, Any]:
        return {}

    def _run_jxf(self, num_steps: int, dt: float | None) -> tuple[JaxFluidsBuffers, dict[str, Any]]:
        
        jxf_buffers, callback_dict = self._require_state()

        sim_manager = self.sim_manager
        for _ in range(num_steps): 

            start_step = sim_manager.synchronize_and_clock(
                jxf_buffers.simulation_buffers.material_fields.primitives
            )

            control_flow_params = sim_manager.compute_control_flow_params(
                jxf_buffers.time_control_variables,
                jxf_buffers.step_information
            )

            # NOTE CALLBACK
            jxf_buffers, callback_dict = sim_manager._callback(
                "before_step_start",
                jxf_buffers=jxf_buffers,
                callback_dict=callback_dict
            )

            # PERFORM INTEGRATION STEP
            jxf_buffers, callback_dict_step = sim_manager.do_integration_step(
                jxf_buffers,
                control_flow_params,
                self.action_params_setup,
                self.action_callable_setup
            )

            # NOTE CALLBACK - AFTER_STEP_END
            # This callback receives the callback_dict_step
            jxf_buffers, callback_dict = sim_manager._callback(
                "after_step_end",
                jxf_buffers=jxf_buffers,
                callback_dict=callback_dict,
                callback_dict_step=callback_dict_step
            )
            
            # CLOCK INTEGRATION STEP
            end_step = sim_manager.synchronize_and_clock(
                jxf_buffers.simulation_buffers.material_fields.primitives
            )
            wall_clock_step = end_step - start_step

            # COMPUTE WALL CLOCK TIMES FOR TIME STEP
            self.wall_clock_times = sim_manager.compute_wall_clock_time(
                wall_clock_step,
                self.wall_clock_times,
                jxf_buffers.time_control_variables.simulation_step
            )
       
        return jxf_buffers, callback_dict

    def _log_step(
            self,
            observation: np.ndarray,
            reward: float,
            info: dict[str, Any]
        ) -> None:

        if self.env_step % self.log_every_steps == 0:
            jxf_buffers, _ = self._require_state()

            time_control_variables = jxf_buffers.time_control_variables

            self.logger.info(
                f"ep={self.episode_idx} env_step={self.env_step} sim_step={time_control_variables.simulation_step} "
                f"sim_time={time_control_variables.physical_simulation_time:.3e} "
                f"wct/step/cell={self.wall_clock_times.mean_step_per_cell * 1e9:.2f}ns"
            )

    def _log_reset(self) -> None:
        self.logger.info("Env reset.")

    def _set_spaces(self, action_space: gym.spaces.Box, observation_space: gym.spaces.Box) -> None:
        self.action_space = action_space
        self.observation_space = observation_space

    def _set_state(
            self,
            jxf_buffers: JaxFluidsBuffers,
            callback_dict: dict[str, Any] | None = None,
        ) -> None:
        self.state = (jxf_buffers, callback_dict or {})

    def _require_state(self) -> tuple[JaxFluidsBuffers, dict[str, Any]]:
        if self.state is None:
            raise RuntimeError(
                f"{self.__class__.__name__}: state not initialized. "
                "Call reset() before step()/render()."
            )
        return self.state

    def _reset_history(self) -> None:
        self.history = defaultdict(list)

    def _append_history(self, **values: Any) -> None:
        for key, value in values.items():
            self.history[key].append(value)

    def _after_step(
            self,
            action: np.ndarray,
            observation: np.ndarray,
            reward: float,
            terminated: bool,
            truncated: bool,
            info: dict[str, Any],
            jxf_buffers: JaxFluidsBuffers,
        ) -> None:
            """Override in derived env if needed."""
            pass

    def _save_render_figure(self, fig: "Figure", stem: str = "flowfield") -> None:
        self.render_dir.mkdir(parents=True, exist_ok=True)
        if stem not in self._render_frame_indices:
            self._render_frame_indices[stem] = 0
        path = self.render_dir / f"{stem}_{self._render_frame_indices[stem]:05d}.png"
        fig.savefig(path, dpi=int(self.env_config.get("render_dpi", 300)), bbox_inches="tight")
        self._render_frame_indices[stem] += 1
        self.logger.debug(f"Saved render frame: {path}")

    def _init_logger(self, level: str = "INFO", log_to_file: bool = True) -> logging.Logger:
        logger = logging.getLogger(f"{self.__class__.__name__}.{id(self)}")
        logger.handlers.clear()
        logger.propagate = False
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))

        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)

        if log_to_file:
            fh = logging.FileHandler(self.log_dir / "env.log")
            fh.setFormatter(fmt)
            logger.addHandler(fh)

        return logger

    @property
    def _spaces_set(self) -> bool:
        return (
            self.action_space is not None
            and self.observation_space is not None
        ) 