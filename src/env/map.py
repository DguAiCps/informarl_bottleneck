"""
Map generation for bottleneck environment
기존 map.py에서 필요한 부분만 추출
"""
import numpy as np
from typing import List, Tuple
from ..utils.types import Agent2D, Landmark2D, Obstacle2D


def create_agents_and_landmarks(num_agents: int, corridor_width: float, corridor_height: float,
                               agent_radius: float, wall_margin: float) -> Tuple[List[Agent2D], List[Landmark2D]]:
    """에이전트와 목표 지점 생성"""
    agents = []
    landmarks = []
    
    # 목표 지점 생성 (벽에서 충분히 떨어뜨리기)
    for i in range(num_agents):
        if i % 2 == 0:  # L->R 이동하는 에이전트
            target_x = np.random.uniform(corridor_width - 3.0, corridor_width - wall_margin)
        else:  # R->L 이동하는 에이전트
            target_x = np.random.uniform(wall_margin, 3.0)
        
        target_y = np.random.uniform(wall_margin, corridor_height - wall_margin)
        
        landmark = Landmark2D(id=i, x=target_x, y=target_y)
        landmarks.append(landmark)
    
    # 에이전트 생성 (목표와 반대편에서 시작)
    for i in range(num_agents):
        if i % 2 == 0:  # L->R 에이전트는 왼쪽에서 시작
            start_x = np.random.uniform(wall_margin, 3.0)
        else:  # R->L 에이전트는 오른쪽에서 시작
            start_x = np.random.uniform(corridor_width - 3.0, corridor_width - wall_margin)
        
        start_y = np.random.uniform(wall_margin, corridor_height - wall_margin)
        max_speed = np.random.uniform(1.0, 2.0)
        
        agent = Agent2D(
            id=i, x=start_x, y=start_y, vx=0.0, vy=0.0,
            radius=agent_radius, target_id=i, max_speed=max_speed
        )
        agents.append(agent)
    
    return agents, landmarks


def create_obstacles(corridor_width: float, corridor_height: float,
                    bottleneck_position: float, bottleneck_width: float,
                    agent_radius: float) -> List[Obstacle2D]:
    """병목 벽을 작은 obstacle 노드들로 생성 - InforMARL 방식"""
    obstacles = []
    obstacle_id = 0
    
    # 벽 obstacle 크기 설정 (작게)
    wall_obstacle_radius = 0.15
    wall_spacing = wall_obstacle_radius * 1.5
    
    center_y = corridor_height / 2
    upper_wall_start = center_y + bottleneck_width / 2  # 5.6
    lower_wall_end = center_y - bottleneck_width / 2    # 4.4
    wall_left = bottleneck_position - 0.5   # 9.5
    wall_right = bottleneck_position + 0.5  # 10.5
    
    # === 병목 벽의 테두리만 obstacle로 표현 ===
    
    # 위쪽 회색 벽의 아래 경계선 (병목 통로 위쪽 경계)
    x_positions = np.arange(wall_left, wall_right, wall_spacing)
    for x in x_positions:
        obstacle = Obstacle2D(
            id=obstacle_id,
            x=x,
            y=upper_wall_start,  # y = 5.6 (병목 통로 위쪽 경계)
            radius=wall_obstacle_radius
        )
        obstacles.append(obstacle)
        obstacle_id += 1
    
    # 아래쪽 회색 벽의 위 경계선 (병목 통로 아래쪽 경계)
    x_positions = np.arange(wall_left, wall_right, wall_spacing)
    for x in x_positions:
        obstacle = Obstacle2D(
            id=obstacle_id,
            x=x,
            y=lower_wall_end,  # y = 4.4 (병목 통로 아래쪽 경계)
            radius=wall_obstacle_radius
        )
        obstacles.append(obstacle)
        obstacle_id += 1
    
    # 병목 벽의 좌우 경계선
    y_positions_upper = np.arange(upper_wall_start, corridor_height, wall_spacing)
    y_positions_lower = np.arange(0, lower_wall_end, wall_spacing)
    
    # 왼쪽 경계
    for y in list(y_positions_upper) + list(y_positions_lower):
        obstacle = Obstacle2D(
            id=obstacle_id,
            x=wall_left,  # x = 9.5
            y=y,
            radius=wall_obstacle_radius
        )
        obstacles.append(obstacle)
        obstacle_id += 1
    
    # 오른쪽 경계
    for y in list(y_positions_upper) + list(y_positions_lower):
        obstacle = Obstacle2D(
            id=obstacle_id,
            x=wall_right,  # x = 10.5
            y=y,
            radius=wall_obstacle_radius
        )
        obstacles.append(obstacle)
        obstacle_id += 1
    
    # === 환경 테두리 벽들도 obstacle로 추가 ===
    boundary_spacing = wall_spacing * 2  # 테두리는 좀 더 sparse하게
    
    # 위쪽 테두리 (y = corridor_height)
    x_positions = np.arange(0, corridor_width, boundary_spacing)
    for x in x_positions:
        obstacle = Obstacle2D(
            id=obstacle_id,
            x=x,
            y=corridor_height - wall_obstacle_radius,
            radius=wall_obstacle_radius
        )
        obstacles.append(obstacle)
        obstacle_id += 1
    
    # 아래쪽 테두리 (y = 0)
    x_positions = np.arange(0, corridor_width, boundary_spacing)
    for x in x_positions:
        obstacle = Obstacle2D(
            id=obstacle_id,
            x=x,
            y=wall_obstacle_radius,
            radius=wall_obstacle_radius
        )
        obstacles.append(obstacle)
        obstacle_id += 1
    
    # 왼쪽 테두리 (x = 0)
    y_positions = np.arange(0, corridor_height, boundary_spacing)
    for y in y_positions:
        obstacle = Obstacle2D(
            id=obstacle_id,
            x=wall_obstacle_radius,
            y=y,
            radius=wall_obstacle_radius
        )
        obstacles.append(obstacle)
        obstacle_id += 1
    
    # 오른쪽 테두리 (x = corridor_width)
    y_positions = np.arange(0, corridor_height, boundary_spacing)
    for y in y_positions:
        obstacle = Obstacle2D(
            id=obstacle_id,
            x=corridor_width - wall_obstacle_radius,
            y=y,
            radius=wall_obstacle_radius
        )
        obstacles.append(obstacle)
        obstacle_id += 1
    
    return obstacles