"""
Reward calculation for bottleneck navigation
기존 reward.py에서 waypoint 의존성 제거, 순수 거리 기반 보상
"""
import numpy as np
import math
from typing import List
from ..utils.types import Agent2D, Landmark2D


def calculate_rewards(agents: List[Agent2D], landmarks: List[Landmark2D]) -> List[float]:
    """순수 거리 기반 보상 계산 - waypoint 제거"""
    rewards = []
    
    for agent in agents:
        reward = 0.0
        target = landmarks[agent.target_id]
        distance = agent.get_distance_to(target.x, target.y)
        
        # 목표 도달 보상
        if distance < target.radius:
            reward += 1000.0
        else:
            # 거리 기반 보상 (가까울수록 높은 보상) - 강화
            max_distance = 30.0
            reward += (max_distance - distance) * 0.1  # 강한 거리 인센티브
        
        # 충돌 패널티
        for other_agent in agents:
            if other_agent.id != agent.id:
                agent_dist = agent.get_distance_to_agent(other_agent)
                if agent_dist < (agent.radius + other_agent.radius) * 1.5:
                    reward -= 50.0  # 충돌 패널티 강화
        
        # 속도 보상 (너무 느리면 패널티)
        speed = math.sqrt(agent.vx**2 + agent.vy**2)
        if speed < 0.1:
            reward -= 0.5
        
        # 벽 충돌 패널티 추가
        corridor_width = 20.0
        corridor_height = 10.0
        wall_margin = 1.0  # 벽 근처 위험 구역
        
        if (agent.x < wall_margin or agent.x > corridor_width - wall_margin or 
            agent.y < wall_margin or agent.y > corridor_height - wall_margin):
            reward -= 10.0  # 강한 벽 패널티
        
        # 시간 패널티 (빨리 끝내도록 장려)
        reward -= 0.01
        
        rewards.append(reward)
    
    return rewards


def calculate_simple_rewards(agents: List[Agent2D], landmarks: List[Landmark2D]) -> List[float]:
    """가장 단순한 보상 - 목표 도달만 고려"""
    rewards = []
    
    for agent in agents:
        target = landmarks[agent.target_id]
        distance = agent.get_distance_to(target.x, target.y)
        
        if distance < target.radius:
            reward = 10.0  # 성공
        else:
            reward = -0.01  # 기본 시간 패널티
        
        rewards.append(reward)
    
    return rewards