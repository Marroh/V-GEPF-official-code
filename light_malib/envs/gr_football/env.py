# Copyright 2022 Digital Brain Laboratory, Yan Song and He jiang
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
import base64
import io
import logging
import time

import imageio
import ray
import requests
import torch
import wandb
from ray import serve
from io import BytesIO
import clip

from light_malib.utils.episode import EpisodeKey
from ..base_env import BaseEnv

from gfootball import env as gfootball_official_env
from gfootball.env.football_env import FootballEnv
# except ImportError as e:
#     raise e(
#         "Please install Google football evironment before use: https://github.com/google-research/football"
#     ) from None

from .state import State
from .tools.tracer import MatchTracer
import numpy as np
from .rewarder_basic import Rewarder as Rewarder_basic
from .rewarder_academy import Rewarder as Rewarder_academy
from .stats_basic import StatsCaculator
from .game_graph.game_graph import GameGraph

from light_malib.utils.timer import global_timer
from light_malib.utils.logger import Logger
from light_malib.registry import registry
from light_malib.vlm.utils import RealTimeDrawer, vLLMAgent, AnyBuffer


def register_new_scenarios():
    import sys
    import pkgutil
    import os
    import importlib
    path = os.path.join(os.path.dirname(__file__), "scenarios")
    for _, module_name, _ in pkgutil.walk_packages(path=[path]):
        module = importlib.import_module("light_malib.envs.gr_football.scenarios.{}".format(module_name))
        sys.modules["gfootball.scenarios.{}".format(module_name)] = module

def clip(x, ma, mi):
    if x > ma:
        x = ma
    if x < mi:
        x = mi
    return x

def map_coordinates_to_xG(xG, coord):
    m, n, c = xG.shape
    x, y = coord
    x = int((x + 1) / 2)
    y = int(abs(y) / 0.42)
    x = clip(x, m - 1, 0)
    y = clip(y, n - 1, 0)
    value = xG[x, y, 0]
    return value

def map_coordinates_to_xT(xT, coordinates):
    # Get the shape of the array
    rows, cols = xT.shape

    # Initialize a list to store the results
    values = []

    # Iterate over the list of coordinates
    for x, y in coordinates:
        # Map the coordinates to array indices
        col_idx = int((x + 1) / 2 * (cols - 1))  # Map x to the range 0 to cols-1
        row_idx = int((0.42 - y) / 0.84 * (rows - 1))  # Map y to the range 0 to rows-1, note the reversed direction

        # Clip the indices to be within array bounds
        col_idx = max(0, min(cols - 1, col_idx))
        row_idx = max(0, min(rows - 1, row_idx))
        # Get the corresponding array value
        values.append(xT[row_idx, col_idx])

    return values


@registry.registered(registry.ENV, "gr_football")
class GRFootballEnv(BaseEnv):
    def __init__(self, id, seed, cfg):
        super().__init__(id, seed)
        self.cfg = cfg
        scenario_config = self.cfg["scenario_config"]

        # check: it is to add extra space, which is wrongly parsed by the official env.
        offcial_rewards = scenario_config["rewards"].split(",")
        for official_reward in offcial_rewards:
            assert official_reward in ["scoring", "checkpoints"], official_reward

        scenario_name = scenario_config["env_name"]
        # assert scenario_name in [
        #     "5_vs_5",
        #     "10_vs_10_kaggle",
        # ], "Because of some bugs in envs, only these scenarios are supported now. See README"
        # scenario_config["other_config_options"]["game_engine_random_seed"]=int(seed)

        register_new_scenarios()

        self._env: FootballEnv = gfootball_official_env.create_environment(
            **scenario_config
        )
        self.agent_ids = ["agent_0", "agent_1"]
        self.num_players = {
            "agent_0": scenario_config["number_of_left_players_agent_controls"],
            "agent_1": scenario_config["number_of_right_players_agent_controls"],
        }
        self.slices = {
            "agent_0": slice(0, self.num_players["agent_0"]),
            "agent_1": slice(self.num_players["agent_0"], None),
        }
        for num in self.num_players.values():
            assert num > 0, "jh: if built-in ai is wanted, use built_in model."
        self.feature_encoders = {"agent_0": None, "agent_1": None}
        self.num_actions = 19
        if 'academy' in scenario_name:
            self.rewarder = Rewarder_academy(self.cfg.reward_config)
        else:
            self.rewarder = Rewarder_basic(self.cfg.reward_config)

        self.stats_calculators = {
            "agent_0": StatsCaculator(),
            "agent_1": StatsCaculator(),
        }

        # ============================
        #   Common setting for Potential Rew
        # ============================

        # # self.vlm_critic = serve.connect()
        # self.gamma = 0.995
        # self.last_pb_reward = 0
        # # self.vllm = vLLMAgent(model='miniCPM')

        # # ============================
        # #   Variables for V-GEPOR
        # # ============================

        self.drawer = RealTimeDrawer()
        self.recorder = ray.get_actor("Recorder")

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        self.video_buffer = AnyBuffer(30)

        self.rollout_manger = ray.get_actor("RolloutManager")

        # # ==========================================
        # #   Variables for Vanilla Potential Reward
        # # ==========================================

        # self.xG_map = np.load('./potential_rew/pre_save_rew/eg_list.npy')
        # self.xT_map = np.load('./potential_rew/pre_save_rew/WorldCup_xT_5.npy')

        # self.last_xt_reward = [0 for _ in range(11)]


    @property
    def num_players_total(self):
        return sum(self.num_players.values())

    def reset(self, custom_reset_config):
        self.feature_encoders = custom_reset_config["feature_encoders"]
        self.main_agent_id = custom_reset_config["main_agent_id"]  #for tracer
        self.rollout_length = custom_reset_config["rollout_length"]

        self.step_ctr = 0
        self.done = False

        global_timer.record("env_step_start")
        observations = self._env.reset()
        global_timer.time("env_step_start", "env_step_end", "env_step")

        if '5_vs_5' in self.cfg["scenario_config"]['env_name'] or '4_vs_4' in self.cfg["scenario_config"]['env_name']:
            self.states = [State(n_player=5) for i in range(self.num_players_total)]
        else:
            self.states = [State() for i in range(self.num_players_total)]

        self.tracer = MatchTracer()
        self.tracer.update_settings(
            {
                "n_left_control": self.num_players["agent_0"],
                "n_right_control": self.num_players["agent_1"],
            }
        )

        for stats_calculator in self.stats_calculators.values():
            stats_calculator.reset()

        assert len(observations) == len(self.states)
        for o, s in zip(observations, self.states):
            s.update_obs(o)

        if self.main_agent_id == 'agent_0':
            self.tracer.update(observations[0:1])
        elif self.main_agent_id == 'agent_1':
            self.tracer.update(observations[-1:])
        else:
            raise NotImplementedError

        encoded_observations, action_masks = self.encode()
        dones = {k: np.zeros((v, 1), dtype=bool) for k, v in self.num_players.items()}

        team_0_state = self.states[0].get_team_states()
        team_1_state = self.states[-1].get_team_states()
        team_state = {agent_id: self.states[(-1) ** idx].get_team_states()
                      for idx, agent_id in enumerate(self.agent_ids)}  #for QMix train

        rets = {
            agent_id: {
                EpisodeKey.NEXT_OBS: encoded_observations[agent_id],
                EpisodeKey.ACTION_MASK: action_masks[agent_id],
                EpisodeKey.DONE: dones[agent_id],
                EpisodeKey.GLOBAL_STATE: team_state[agent_id][np.newaxis, ...]
            }
            for agent_id in self.agent_ids
        }
        return rets

    def step(self, actions):
        self.step_ctr += 1
        # self.logger.info(f'>> Step: {self.step_ctr}')

        actions = np.concatenate(
            [actions[agent_id][EpisodeKey.ACTION] for agent_id in self.agent_ids],
            axis=0,
        ).flatten()

        global_timer.record("env_core_step_start")
        observations, rewards, done, info = self._env.step(actions)

        # ray.util.pdb.set_trace()
        # # ===============================
        # #  Casual PB rew start ~
        # # ==============================
        # rewards = np.array(rewards)
        
        # left_team = observations[0]['left_team']
        # ball = observations[0]['ball'][:2]
        
        # # # Calculate potential-based reward
        # xg = map_coordinates_to_xG(self.xG_map, ball)
        # xt = map_coordinates_to_xT(self.xT_map, left_team)
        
        # xg_pb_reward = self.gamma * xg - self.last_pb_reward
        # self.last_pb_reward = xg_pb_reward
        # rewards += xg_pb_reward * 0.01
        
        # for idx in range(11):
        #     xT_pb_reward = self.gamma * xt[idx] - self.last_xt_reward[idx]
        #     self.last_xt_reward[idx] = xT_pb_reward
        #     rewards[idx] += xT_pb_reward * 0.02

        # save frames before vLLM guiding
        # ===============================
        #    V-GEPF trigger replay image saving for vLLM
        # =================================
        print(f'>> Rollout epoch: {ray.get(self.rollout_manger.get_rollout_epoch.remote())}')
        if ((ray.get(self.rollout_manger.get_rollout_epoch.remote()) + 1) % 20 == 0
                and ray.get(self.rollout_manger.get_rollout_epoch.remote()) != 0
                and not ray.get(self.recorder.get.remote('images'))
                and self.step_ctr % 3 == 0):
            print(f'>> Recording video...')
            if not done and not self.video_buffer.is_full():
                img = self.drawer.draw_from_obs(observations[0], simple=True)
                self.video_buffer.add(img)
                print(f'>> Video buffer length {len(self.video_buffer.data)}')
            else:
                ray.get(self.recorder.add.remote({'images': self.video_buffer.data}, 1))

                # # note: test
                # with imageio.get_writer(f'/home/trl/football/videos/{time.time()}.mp4', fps=25) as writer:
                #     for image in self.video_buffer.data:
                #         # Convert PIL Image to NumPy array
                #         frame = imageio.core.util.Array(np.array(image))
                #         # Append the frame to the video
                #         writer.append_data(frame)

                print(f">> Video saved!")
                self.video_buffer.clear()

        # official rewards should be shared
        for agent_id in self.agent_ids:
            rewards[self.slices[agent_id]] = np.sum(rewards[self.slices[agent_id]])

        self.done = done

        global_timer.time("env_core_step_start", "env_core_step_end", "env_core_step")

        assert len(observations) == len(self.states) and len(actions) == len(
            self.states
        )
        for o, a, s in zip(observations, actions, self.states):
            s.update_action(a)
            s.update_obs(o)
        if self.main_agent_id == 'agent_0':
            self.tracer.update(observations[0:1])
        elif self.main_agent_id == 'agent_1':
            self.tracer.update(observations[-1:])
        else:
            raise NotImplementedError

        global_timer.record("reward_start")
        rewards = self.get_reward(rewards)
        global_timer.time("reward_start", "reward_end", "reward")

        global_timer.record("stats_start")
        self.update_episode_stats(rewards)
        global_timer.time("stats_start", "stats_end", "stats")

        global_timer.record("feature_start")
        encoded_observations, action_masks = self.encode()
        global_timer.time("feature_start", "feature_end", "feature")

        if info["score_reward"]:  # if score, done  to segment the long-horizon traj
            done = True

        dones = {
            k: np.full((v, 1), fill_value=done, dtype=bool)
            for k, v in self.num_players.items()
        }
        team_0_state = self.states[0].get_team_states()
        team_1_state = self.states[-1].get_team_states()
        team_state = {agent_id: self.states[(-1) ** idx].get_team_states()
                      for idx, agent_id in enumerate(self.agent_ids)}

        rets = {
            agent_id: {
                EpisodeKey.NEXT_OBS: encoded_observations[agent_id],
                EpisodeKey.ACTION_MASK: action_masks[agent_id],
                EpisodeKey.REWARD: rewards[agent_id],
                EpisodeKey.DONE: dones[agent_id],
                EpisodeKey.GLOBAL_STATE: team_state[agent_id][np.newaxis, ...]
            }
            for agent_id in self.agent_ids
        }
        return rets

    def get_AssistInfo(self):
        game_graph = GameGraph(self.tracer)
        left_team_assist_t = []
        left_team_goal_t = []
        left_team_loseball_t = []
        left_team_haltloseball_t = []
        left_team_gainball_t = []
        for g_t, g in game_graph.goals.items():  # for each goal events
            if g.team == 0 and g.score == 1:  # if our goals from ower team
                shot_player = g.player
                shot_t = g.out_step
                left_team_goal_t.append({"t": shot_t, "player": shot_player})
                # search for last pass
                previous_pass = np.where(
                    np.array(list(game_graph.passings.keys())) < shot_t
                )[0]
                if len(previous_pass) > 0:
                    previous_pass = previous_pass[-1]
                    previous_pass = list(game_graph.passings.keys())[previous_pass]
                    previous_pass = game_graph.passings[previous_pass]
                    if previous_pass.team == 0:  # this pass is what we want
                        pass_player = previous_pass.player
                        pass_t = previous_pass.out_step
                        if pass_player != 0:
                            left_team_assist_t.append(
                                {"t": pass_t, "player": pass_player}
                            )
        for l_t, l in game_graph.losing_balls.items():
            if l.team == 0:
                if l.next_game_mode == 0:  # out team get intercepted
                    lose_ball_player = l.player
                    lose_ball_t = l.out_step
                    if lose_ball_player != 0:
                        left_team_loseball_t.append(
                            {"t": lose_ball_t, "player": lose_ball_player}
                        )
                    # also search for last passer

                else:
                    halt_loseball_player = l.player
                    halt_loseball_t = l.out_step
                    passer_node = l.extra_info["passer_node"]
                    if passer_node is not None:
                        passer_player = passer_node.owned_player
                        passer_pass_t = passer_node.e_step
                        if passer_player != 0:  # passer also get penalty
                            left_team_haltloseball_t.append(
                                {"t": passer_pass_t, "player": passer_player}
                            )

                    if halt_loseball_player != 0:
                        left_team_haltloseball_t.append(
                            {"t": halt_loseball_t, "player": halt_loseball_player}
                        )
            elif l.team == 1 and l.next_game_mode == 0:
                gain_ball_t = l.step + 1
                gain_ball_node = game_graph.step2node[gain_ball_t]
                assert gain_ball_node.owned_team == 0
                gain_ball_player = gain_ball_node.owned_player
                # assert gain_ball_player != 0
                if gain_ball_player != 0:
                    left_team_gainball_t.append(
                        {"t": gain_ball_t, "player": gain_ball_player}
                    )

        return {
            "goal_info": left_team_goal_t,
            "assist_info": left_team_assist_t,
            "loseball_info": left_team_loseball_t,
            "halt_loseball_info": left_team_haltloseball_t,
            "gainball_info": left_team_gainball_t,
        }

    def is_terminated(self):
        if self.done:
            return True

        return self.step_ctr >= self.rollout_length - 1

    def split(self, arr):
        ret = {agent_id: arr[self.slices[agent_id]] for agent_id in self.agent_ids}
        return ret

    def get_reward(self, rewards):
        rewards = np.array(
            [
                [self.rewarder.calc_reward(reward, state)]
                for reward, state in zip(rewards, self.states)
            ],
            dtype=float,
        )

        rewards = self.split(rewards)
        return rewards

    def encode(self):
        states = self.split(self.states)
        encoded_observations = {
            agent_id: np.array(
                self.feature_encoders[agent_id].encode(states[agent_id]),
                dtype=np.float32,
            )
            for agent_id in self.agent_ids
        }
        action_masks = {
            agent_id: encoded_observations[agent_id][..., : self.num_actions]
            for agent_id in self.agent_ids
        }
        return encoded_observations, action_masks

    def update_episode_stats(self, rewards):
        """
        we only count statistics for main agent now
        """
        states = self.split(self.states)
        for agent_id in self.agent_ids:
            for idx, state in enumerate(states[agent_id]):
                self.stats_calculators[agent_id].calc_stats(
                    state, rewards[agent_id][idx][0], idx
                )

    def get_episode_stats(self):
        return {
            agent_id: self.stats_calculators[agent_id].stats
            for agent_id in self.agent_ids
        }

        for aid in return_stats.keys():
            return_stats[aid]['episode_length'] = self.step_ctr

        return return_stats

    def render(self, mode="human"):
        assert mode == "human"
        self._env.render()
