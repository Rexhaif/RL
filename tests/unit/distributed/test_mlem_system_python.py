# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys

from nemo_rl.distributed.ray_actor_environment_registry import get_actor_python_env


def test_system_python_override_applies_to_all_registered_actors(monkeypatch):
    monkeypatch.setenv("NEMO_RL_PY_EXECUTABLES_SYSTEM", "1")

    assert (
        get_actor_python_env("nemo_rl.environments.nemo_gym.NemoGym")
        == sys.executable
    )
    assert (
        get_actor_python_env("nemo_rl.algorithms.async_utils.ReplayBuffer")
        == sys.executable
    )

