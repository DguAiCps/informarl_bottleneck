"""
Original InforMARL Graph Construction
원본과 완전히 동일한 방향성 그래프 + entity type embedding
"""
import torch
import numpy as np
import math
from typing import List, Tuple

from ..utils.types import Agent2D, Landmark2D, Obstacle2D, ENTITY_TYPES


def build_informarl_graph_observations(agents: List[Agent2D], landmarks: List[Landmark2D], 
                                      obstacles: List[Obstacle2D], sensing_radius: float,
                                      max_nodes: int = 30) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    원본 InforMARL 방식 그래프 생성
    
    Returns:
        node_obs: [num_agents, max_nodes, 6] 노드 특성
        adj: [num_agents, max_nodes, max_nodes] 인접 행렬 (방향성 있음)
        entity_types: [num_agents, max_nodes] entity type IDs
    """
    num_agents = len(agents)
    
    # 고정 크기 배열들 초기화
    node_obs = np.zeros((num_agents, max_nodes, 6), dtype=np.float32)
    adj = np.zeros((num_agents, max_nodes, max_nodes), dtype=np.float32)
    entity_types = np.zeros((num_agents, max_nodes), dtype=np.int32)
    
    # 각 에이전트를 ego agent로 하여 그래프 생성
    for ego_idx, ego_agent in enumerate(agents):
        entities = []  # [(entity_type_id, entity_obj, node_features), ...]
        
        # === 1. 센싱 범위 내 에이전트들 수집 ===
        for agent in agents:
            distance = ego_agent.get_distance_to_agent(agent)
            if distance <= sensing_radius:
                # 상대적 위치/속도 (sensing_radius로 정규화)
                rel_x = (agent.x - ego_agent.x) / sensing_radius
                rel_y = (agent.y - ego_agent.y) / sensing_radius
                rel_vx = agent.vx / agent.max_speed
                rel_vy = agent.vy / agent.max_speed
                
                # 해당 에이전트의 목표 위치 (에이전트 기준)
                target = landmarks[agent.target_id] 
                goal_x = (target.x - agent.x) / sensing_radius
                goal_y = (target.y - agent.y) / sensing_radius
                
                features = [rel_x, rel_y, rel_vx, rel_vy, goal_x, goal_y]
                entities.append((ENTITY_TYPES["agent"], agent, features))
        
        # === 2. 센싱 범위 내 landmark들 수집 ===
        for landmark in landmarks:
            distance = ego_agent.get_distance_to(landmark.x, landmark.y)
            if distance <= sensing_radius:
                rel_x = (landmark.x - ego_agent.x) / sensing_radius
                rel_y = (landmark.y - ego_agent.y) / sensing_radius
                
                features = [rel_x, rel_y, 0.0, 0.0, 0.0, 0.0]  # 정지 상태
                entities.append((ENTITY_TYPES["landmark"], landmark, features))
        
        # === 3. 센싱 범위 내 obstacle들 수집 ===
        for obstacle in obstacles:
            distance = ego_agent.get_distance_to(obstacle.x, obstacle.y)
            if distance <= sensing_radius:
                rel_x = (obstacle.x - ego_agent.x) / sensing_radius
                rel_y = (obstacle.y - ego_agent.y) / sensing_radius
                
                features = [rel_x, rel_y, 0.0, 0.0, 0.0, 0.0]  # 정지 상태
                entities.append((ENTITY_TYPES["obstacle"], obstacle, features))
        
        # === 4. 최대 노드 수 제한 (거리 순으로 선택) ===
        if len(entities) > max_nodes:
            entities_with_dist = []
            for entity_type, entity_obj, features in entities:
                if hasattr(entity_obj, 'x'):
                    dist = ego_agent.get_distance_to(entity_obj.x, entity_obj.y)
                else:
                    dist = ego_agent.get_distance_to_agent(entity_obj)
                entities_with_dist.append((dist, entity_type, entity_obj, features))
            
            # 거리순 정렬 후 가까운 것들만 선택
            entities_with_dist.sort(key=lambda x: x[0])
            entities = [(entity_type, entity_obj, features) 
                       for _, entity_type, entity_obj, features in entities_with_dist[:max_nodes]]
        
        # === 5. 고정 크기 배열에 데이터 채우기 ===
        num_actual_nodes = len(entities)
        for i, (entity_type, entity_obj, features) in enumerate(entities):
            node_obs[ego_idx, i] = features
            entity_types[ego_idx, i] = entity_type
        
        # === 6. InforMARL 방향성 인접 행렬 생성 ===
        for i in range(num_actual_nodes):
            for j in range(num_actual_nodes):
                if i == j:
                    continue  # 자기 자신과는 연결 안 함
                
                entity_i_type, _, _ = entities[i]
                entity_j_type, _, _ = entities[j]
                
                # 원본 InforMARL 연결 규칙:
                # - Agent ↔ Agent: 양방향 연결 (정보 교환)
                # - Agent → Landmark: 단방향 연결 (에이전트가 목표 관측)
                # - Agent → Obstacle: 단방향 연결 (에이전트가 장애물 관측)
                
                if entity_i_type == ENTITY_TYPES["agent"] and entity_j_type == ENTITY_TYPES["agent"]:
                    # Agent ↔ Agent: 양방향 연결
                    adj[ego_idx, i, j] = 1.0
                    adj[ego_idx, j, i] = 1.0
                
                elif entity_i_type == ENTITY_TYPES["agent"] and entity_j_type in [ENTITY_TYPES["landmark"], ENTITY_TYPES["obstacle"]]:
                    # Agent → Target: 단방향 연결 (agent가 target을 관측)
                    adj[ego_idx, i, j] = 1.0
                    # 반대 방향은 연결하지 않음 (adj[ego_idx, j, i] = 0.0)
    
    return node_obs, adj, entity_types


def batch_build_informarl_graph_gpu(agents: List[Agent2D], landmarks: List[Landmark2D], 
                                   obstacles: List[Obstacle2D], sensing_radius: float, 
                                   device, max_nodes: int = 30) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """GPU 버전 InforMARL 그래프 생성"""
    # CPU에서 계산 후 GPU로 이동
    node_obs, adj, entity_types = build_informarl_graph_observations(
        agents, landmarks, obstacles, sensing_radius, max_nodes
    )
    
    # GPU로 이동
    node_obs_gpu = torch.tensor(node_obs, dtype=torch.float32, device=device)
    adj_gpu = torch.tensor(adj, dtype=torch.float32, device=device)
    entity_types_gpu = torch.tensor(entity_types, dtype=torch.long, device=device)
    
    return node_obs_gpu, adj_gpu, entity_types_gpu


# 기존 인터페이스와 호환성을 위한 래퍼
def build_graph_observations_fixed_size(agents: List[Agent2D], landmarks: List[Landmark2D], 
                                       obstacles: List[Obstacle2D], sensing_radius: float,
                                       include_obstacles: bool = True, max_nodes: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """기존 인터페이스 호환 - entity_types는 제외하고 반환"""
    if not include_obstacles:
        obstacles = []  # obstacle 제외
    
    node_obs, adj, entity_types = build_informarl_graph_observations(
        agents, landmarks, obstacles, sensing_radius, max_nodes
    )
    
    return node_obs, adj