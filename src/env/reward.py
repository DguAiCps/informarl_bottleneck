"""
Reward calculation for bottleneck navigation
기존 reward.py에서 waypoint 의존성 제거, 순수 거리 기반 보상
"""
import numpy as np
import math
from typing import List
from ..utils.types import Agent2D, Landmark2D


def calculate_rewards(agents: List[Agent2D], landmarks: List[Landmark2D]) -> List[float]:
    """원본 InforMARL과 동일한 보상 방식"""
    
    individual_rewards = []
    for agent in agents:
        reward = 0.0
        target = landmarks[agent.target_id]
        distance = agent.get_distance_to(target.x, target.y)
        
        # 수정된 보상 방식 - 거리 패널티 약화
        if distance < target.radius:
            reward += 10.0  # 목표 도달 보상 강화
        else:
            reward -= distance * 0.1  # 거리 패널티 대폭 약화 (기존 대비 1/10)
        
        # 원본 InforMARL 충돌 패널티 (collision_rew = 5)
        for other_agent in agents:
            if other_agent.id != agent.id:
                agent_dist = agent.get_distance_to_agent(other_agent)
                if agent_dist < (agent.radius + other_agent.radius):
                    reward -= 5.0  # 원본과 동일
        
        individual_rewards.append(reward)
    return individual_rewards
    
    # NEW JOINT REWARD CODE (논문 방식):

    #individual_rewards = []
"""    
    # 각 에이전트의 개별 보상 계산 (합산하기 위해)
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
        
        # 충돌 패널티 (에이전트별로 계산하지만 중복 방지)
        for other_agent in agents:
            if other_agent.id > agent.id:  # 중복 방지: 각 충돌을 한 번만 계산
                agent_dist = agent.get_distance_to_agent(other_agent)
                if agent_dist < (agent.radius + other_agent.radius) * 1.5:
                    reward -= 25.0  # 총 50을 두 에이전트가 나눠 가짐
        
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
        
        individual_rewards.append(reward)
    
    # Joint reward: 모든 에이전트 보상의 합
    joint_reward = sum(individual_rewards)
    
    # 모든 에이전트가 동일한 총합 보상을 받음
    joint_rewards = [joint_reward] * len(agents)
    
    return joint_rewards
"""

def calculate_simple_rewards(agents: List[Agent2D], landmarks: List[Landmark2D]) -> List[float]:
    """가장 단순한 보상 - 목표 도달만 고려 (Joint reward 방식)"""
    
    # ORIGINAL INDIVIDUAL REWARD CODE (주석처리 - 되돌리기 위해 보존):
    # rewards = []
    # for agent in agents:
    #     target = landmarks[agent.target_id]
    #     distance = agent.get_distance_to(target.x, target.y)
    #     
    #     if distance < target.radius:
    #         reward = 10.0  # 성공
    #     else:
    #         reward = -0.01  # 기본 시간 패널티
    #     
    #     rewards.append(reward)
    # return rewards
    
    # NEW JOINT REWARD CODE (논문 방식):
    individual_rewards = []
    
    for agent in agents:
        target = landmarks[agent.target_id]
        distance = agent.get_distance_to(target.x, target.y)
        
        if distance < target.radius:
            reward = 10.0  # 성공
        else:
            reward = -0.01  # 기본 시간 패널티
        
        individual_rewards.append(reward)
    
    # Joint reward: 모든 에이전트 보상의 합
    joint_reward = sum(individual_rewards)
    
    # 모든 에이전트가 동일한 총합 보상을 받음
    joint_rewards = [joint_reward] * len(agents)
    
    return joint_rewards