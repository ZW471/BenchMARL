#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass, MISSING


@dataclass
class TaskConfig:
    max_steps: int = MISSING
    n_agents_holonomic: int = MISSING
    n_agents_diff_drive: int = MISSING
    n_agents_car: int = MISSING
    shared_rew: bool = MISSING
    lidar_range: float = MISSING
    agent_radius: float = MISSING
    comms_rendering_range: float = MISSING
    n_obstacles: int = MISSING