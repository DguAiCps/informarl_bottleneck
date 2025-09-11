"""
Clean Bottleneck Environment for InforMARL
기존 복잡한 환경에서 waypoint, YAML 의존성 제거, GPU 배치처리만 유지
"""
import torch
import numpy as np
from torch_geometric.data import Data
from typing import List, Tuple, Dict, Any

from ..utils.types import Agent2D, Landmark2D, Obstacle2D
from .map import create_agents_and_landmarks, create_obstacles
from .physics import execute_continuous_action, update_positions, batch_execute_actions_gpu, batch_update_positions_gpu
from .reward import calculate_rewards
from .graph_builder_informarl import build_informarl_graph_observations, batch_build_informarl_graph_gpu
from .render import BottleneckRenderer


class CleanBottleneckEnv:
    """깔끔한 병목 환경 - InforMARL 핵심만 유지"""
    
    def __init__(self, num_agents: int = 4, corridor_width: float = 20.0, 
                 corridor_height: float = 10.0, bottleneck_width: float = 1.2,
                 bottleneck_pos: float = 10.0, agent_radius: float = 0.5,
                 sensing_radius: float = 3.0, max_speed: float = 1.0,
                 max_timesteps: int = 300, device=None, use_gpu_physics: bool = True,
                 include_obstacles: bool = True):
        
        self.num_agents = num_agents
        self.corridor_width = corridor_width
        self.corridor_height = corridor_height
        self.bottleneck_width = bottleneck_width
        self.bottleneck_pos = bottleneck_pos
        self.agent_radius = agent_radius
        self.sensing_radius = sensing_radius
        self.max_speed = max_speed
        self.max_timesteps = max_timesteps
        self.use_gpu_physics = use_gpu_physics
        self.include_obstacles = include_obstacles
        
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 환경 상태
        self.agents: List[Agent2D] = []
        self.landmarks: List[Landmark2D] = []
        self.obstacles: List[Obstacle2D] = []
        self.timestep = 0
        self.success_count = 0
        self.collision_count = 0
        self.agent_success_flags = []
        
        # 렌더러
        self.renderer = BottleneckRenderer()
        
        print(f"Clean InforMARL Environment initialized:")
        print(f"  - Device: {self.device}")
        print(f"  - GPU Physics: {self.use_gpu_physics}")
        print(f"  - Include Obstacles: {self.include_obstacles}")
    
    def reset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """환경 리셋"""
        self.timestep = 0
        self.success_count = 0
        self.collision_count = 0
        
        # 벽 여백 계산
        obstacle_spacing = (self.agent_radius * 2) / 3
        obstacle_radius = obstacle_spacing / 2
        wall_margin = obstacle_radius * 3
        
        # 환경 객체 생성 (기존 검증된 함수 사용)
        self.agents, self.landmarks = create_agents_and_landmarks(
            self.num_agents, self.corridor_width, self.corridor_height,
            self.agent_radius, wall_margin
        )
        
        # 장애물 생성 (옵션)
        if self.include_obstacles:
            self.obstacles = create_obstacles(
                self.corridor_width, self.corridor_height,
                self.bottleneck_pos, self.bottleneck_width,
                self.agent_radius
            )
        else:
            self.obstacles = []
        
        self.agent_success_flags = [False] * self.num_agents
        
        # 그래프 관측 생성
        return self._get_graph_observations()
    
    def step(self, actions: List[List[float]]) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], List[float], bool, Dict]:
        """환경 스텝"""
        self.timestep += 1
        
        # 행동 적용
        if self.use_gpu_physics:
            try:
                # GPU 배치 처리
                new_velocities = batch_execute_actions_gpu(self.agents, actions, self.device)
                collision_count = batch_update_positions_gpu(
                    self.agents, new_velocities, self.obstacles,
                    self.corridor_width, self.corridor_height,
                    self.bottleneck_pos, self.bottleneck_width, self.device
                )
            except Exception as e:
                print(f"GPU physics failed, using CPU: {e}")
                # CPU 백업
                for i, action in enumerate(actions):
                    execute_continuous_action(self.agents[i], action)
                collision_count = update_positions(
                    self.agents, self.obstacles, self.corridor_width, self.corridor_height,
                    self.bottleneck_pos, self.bottleneck_width
                )
        else:
            # CPU 처리
            for i, action in enumerate(actions):
                execute_continuous_action(self.agents[i], action)
            collision_count = update_positions(
                self.agents, self.obstacles, self.corridor_width, self.corridor_height,
                self.bottleneck_pos, self.bottleneck_width
            )
        
        self.collision_count += collision_count
        
        # 보상 계산
        rewards = calculate_rewards(self.agents, self.landmarks)
        
        # 성공 체크
        self._check_success()
        
        # 종료 조건
        done = self._is_done()
        
        # 정보
        info = {
            'timestep': self.timestep,
            'success_count': self.success_count,
            'collision_count': self.collision_count,
            'success_rate': (self.success_count / self.num_agents) * 100
        }
        
        return self._get_graph_observations(), rewards, done, info
    
    def _get_graph_observations(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """InforMARL 그래프 관측 생성 - entity types 포함"""
        if self.use_gpu_physics:
            # GPU 가속 그래프 생성
            return batch_build_informarl_graph_gpu(
                self.agents, self.landmarks, self.obstacles, 
                self.sensing_radius, self.device
            )
        else:
            # CPU 그래프 생성
            return build_informarl_graph_observations(
                self.agents, self.landmarks, self.obstacles, 
                self.sensing_radius
            )
    
    def get_local_observation(self, agent_id: int) -> List[float]:
        """에이전트의 로컬 관측"""
        agent = self.agents[agent_id]
        target = self.landmarks[agent.target_id]
        
        obs = [
            agent.x / self.corridor_width,      # 정규화된 위치
            agent.y / self.corridor_height,
            agent.vx / agent.max_speed,         # 정규화된 속도
            agent.vy / agent.max_speed,
            (target.x - agent.x) / self.corridor_width,  # 목표 상대 위치
            (target.y - agent.y) / self.corridor_height
        ]
        
        return obs
    
    def _check_success(self):
        """성공 체크"""
        for i, agent in enumerate(self.agents):
            target = self.landmarks[agent.target_id]
            if agent.get_distance_to(target.x, target.y) < target.radius:
                if not self.agent_success_flags[i]:
                    self.agent_success_flags[i] = True
                    self.success_count += 1
    
    def _is_done(self) -> bool:
        """종료 조건"""
        # 시간 초과
        if self.timestep >= self.max_timesteps:
            return True
        
        # 모든 에이전트 성공
        if self.success_count >= self.num_agents:
            return True
        
        return False
    
    def render(self):
        """환경 렌더링"""
        self.renderer.render(
            self.agents, self.landmarks, self.obstacles,
            self.corridor_width, self.corridor_height,
            self.bottleneck_pos, self.bottleneck_width,
            self.timestep, self.success_count, self.collision_count,
            show_obstacles=self.include_obstacles
        )
    
    def close(self):
        """환경 종료"""
        self.renderer.close()