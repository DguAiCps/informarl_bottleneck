"""
UniMP Graph Builder with Edge Features
논문과 동일하게 엣지 특성(거리 정보) 포함
"""
import torch
import numpy as np
import math
from typing import List, Tuple

from ..utils.types import Agent2D, Landmark2D, Obstacle2D, ENTITY_TYPES


def build_unimp_graph_observations(agents: List[Agent2D], landmarks: List[Landmark2D], 
                                   obstacles: List[Obstacle2D], sensing_radius: float,
                                   max_nodes: int = 30) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    UniMP용 그래프 생성 (엣지 특성 포함)
    
    Returns:
        node_obs: [num_agents, max_nodes, 6] 노드 특성
        adj: [num_agents, max_nodes, max_nodes] 인접 행렬 (방향성)
        entity_types: [num_agents, max_nodes] entity type IDs
        edge_features: [num_agents, max_nodes, max_nodes, 1] 엣지 특성 (정규화된 거리)
    """
    num_agents = len(agents)
    
    # 고정 크기 배열들 초기화
    node_obs = np.zeros((num_agents, max_nodes, 6), dtype=np.float32)
    adj = np.zeros((num_agents, max_nodes, max_nodes), dtype=np.float32)
    entity_types = np.zeros((num_agents, max_nodes), dtype=np.int32)
    edge_features = np.zeros((num_agents, max_nodes, max_nodes, 1), dtype=np.float32)  # 엣지 특성 추가
    
    # 각 에이전트를 ego agent로 하여 그래프 생성
    for ego_idx, ego_agent in enumerate(agents):
        entities = []  # [(entity_type_id, entity_obj, node_features, distance), ...]
        
        # === 1. 센싱 범위 내 에이전트들 수집 ===
        for agent in agents:
            distance = ego_agent.get_distance_to_agent(agent)
            if distance <= sensing_radius:
                # 상대적 위치/속도 (sensing_radius로 정규화)
                rel_x = (agent.x - ego_agent.x) / sensing_radius
                rel_y = (agent.y - ego_agent.y) / sensing_radius
                rel_vx = agent.vx / agent.max_speed
                rel_vy = agent.vy / agent.max_speed
                
                # 해당 에이전트의 목표 위치 (ego 에이전트 기준) - 논문 정의
                target = landmarks[agent.target_id] 
                goal_x = (target.x - ego_agent.x) / sensing_radius  # p^{goal,j}_i
                goal_y = (target.y - ego_agent.y) / sensing_radius
                
                features = [rel_x, rel_y, rel_vx, rel_vy, goal_x, goal_y]
                entities.append((ENTITY_TYPES["agent"], agent, features, distance))
        
        # === 2. 센싱 범위 내 landmark들 수집 ===
        for landmark in landmarks:
            distance = ego_agent.get_distance_to(landmark.x, landmark.y)
            if distance <= sensing_radius:
                rel_x = (landmark.x - ego_agent.x) / sensing_radius
                rel_y = (landmark.y - ego_agent.y) / sensing_radius
                
                # landmark의 경우 goal 정보는 자기 자신 위치
                features = [rel_x, rel_y, 0.0, 0.0, rel_x, rel_y]  # pgoal,j ≡ pji
                entities.append((ENTITY_TYPES["landmark"], landmark, features, distance))
        
        # === 3. 센싱 범위 내 obstacle들 수집 ===
        for obstacle in obstacles:
            distance = ego_agent.get_distance_to(obstacle.x, obstacle.y)
            if distance <= sensing_radius:
                rel_x = (obstacle.x - ego_agent.x) / sensing_radius
                rel_y = (obstacle.y - ego_agent.y) / sensing_radius
                
                # obstacle의 경우도 goal 정보는 자기 자신 위치
                features = [rel_x, rel_y, 0.0, 0.0, rel_x, rel_y]  # pgoal,j ≡ pji
                entities.append((ENTITY_TYPES["obstacle"], obstacle, features, distance))
        
        # === 4. 최대 노드 수 제한 (거리 순으로 선택) ===
        if len(entities) > max_nodes:
            entities = sorted(entities, key=lambda x: x[3])[:max_nodes]  # 거리로 정렬 후 선택
        
        # === 5. 노드 특성 및 entity type 채우기 ===
        for i, (entity_type, entity_obj, features, distance) in enumerate(entities):
            node_obs[ego_idx, i, :] = features
            entity_types[ego_idx, i] = entity_type
        
        # === 6. 인접 행렬 및 엣지 특성 생성 (논문의 방향성 규칙) ===
        for i, (entity_type_i, entity_obj_i, _, dist_i) in enumerate(entities):
            for j, (entity_type_j, entity_obj_j, _, dist_j) in enumerate(entities):
                if i == j:
                    continue  # 자기 자신과는 연결 안 함
                
                # 거리 계산
                if hasattr(entity_obj_i, 'x') and hasattr(entity_obj_j, 'x'):
                    # 두 엔티티 간 거리
                    dx = entity_obj_j.x - entity_obj_i.x
                    dy = entity_obj_j.y - entity_obj_i.y
                    edge_distance = math.sqrt(dx*dx + dy*dy)
                elif hasattr(entity_obj_i, 'get_distance_to_agent'):
                    # agent to agent
                    edge_distance = entity_obj_i.get_distance_to_agent(entity_obj_j)
                elif hasattr(entity_obj_i, 'get_distance_to'):
                    # agent to non-agent
                    edge_distance = entity_obj_i.get_distance_to(entity_obj_j.x, entity_obj_j.y)
                else:
                    edge_distance = sensing_radius  # 기본값
                
                # 논문의 방향성 규칙:
                # - agent-agent: 양방향 (bidirectional)
                # - agent-non-agent: 단방향 (unidirectional, non-agent → agent)
                
                if entity_type_i == ENTITY_TYPES["agent"] and entity_type_j == ENTITY_TYPES["agent"]:
                    # Agent to Agent: 양방향
                    adj[ego_idx, i, j] = 1.0
                    adj[ego_idx, j, i] = 1.0
                    
                    # 엣지 특성: 정규화된 거리
                    normalized_dist = min(edge_distance / sensing_radius, 1.0)
                    edge_features[ego_idx, i, j, 0] = normalized_dist
                    edge_features[ego_idx, j, i, 0] = normalized_dist
                    
                elif entity_type_i == ENTITY_TYPES["agent"] and entity_type_j != ENTITY_TYPES["agent"]:
                    # Non-agent to Agent: 단방향 (j → i)
                    adj[ego_idx, j, i] = 1.0  # non-agent j가 agent i에게 메시지 전송
                    
                    # 엣지 특성
                    normalized_dist = min(edge_distance / sensing_radius, 1.0)
                    edge_features[ego_idx, j, i, 0] = normalized_dist
                    
                elif entity_type_i != ENTITY_TYPES["agent"] and entity_type_j == ENTITY_TYPES["agent"]:
                    # Non-agent to Agent: 단방향 (i → j)
                    adj[ego_idx, i, j] = 1.0  # non-agent i가 agent j에게 메시지 전송
                    
                    # 엣지 특성
                    normalized_dist = min(edge_distance / sensing_radius, 1.0)
                    edge_features[ego_idx, i, j, 0] = normalized_dist
                
                # Non-agent to Non-agent: 연결 안 함 (논문 규칙)
    
    return node_obs, adj, entity_types, edge_features


def batch_build_unimp_graph_gpu(agents_list: List[List[Agent2D]], landmarks_list: List[List[Landmark2D]], 
                                obstacles_list: List[List[Obstacle2D]], sensing_radius: float,
                                max_nodes: int = 30, device=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    GPU 배치 처리를 위한 UniMP 그래프 생성
    
    Args:
        agents_list: [batch_size, num_agents] 에이전트 리스트들
        landmarks_list: [batch_size, num_landmarks] 랜드마크 리스트들  
        obstacles_list: [batch_size, num_obstacles] 장애물 리스트들
        sensing_radius: 센싱 반지름
        max_nodes: 최대 노드 수
        device: GPU 디바이스
        
    Returns:
        node_obs: [batch_size * num_agents, max_nodes, 6]
        adj: [batch_size * num_agents, max_nodes, max_nodes]
        entity_types: [batch_size * num_agents, max_nodes]
        edge_features: [batch_size * num_agents, max_nodes, max_nodes, 1]
    """
    batch_size = len(agents_list)
    if batch_size == 0:
        return None, None, None, None
    
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    all_node_obs = []
    all_adj = []
    all_entity_types = []
    all_edge_features = []
    
    for b in range(batch_size):
        node_obs, adj, entity_types, edge_features = build_unimp_graph_observations(
            agents_list[b], landmarks_list[b], obstacles_list[b], sensing_radius, max_nodes
        )
        
        all_node_obs.append(node_obs)
        all_adj.append(adj)
        all_entity_types.append(entity_types)
        all_edge_features.append(edge_features)
    
    # 배치로 합치기
    node_obs_batch = np.concatenate(all_node_obs, axis=0)  # [batch_size * num_agents, max_nodes, 6]
    adj_batch = np.concatenate(all_adj, axis=0)            # [batch_size * num_agents, max_nodes, max_nodes]
    entity_types_batch = np.concatenate(all_entity_types, axis=0)  # [batch_size * num_agents, max_nodes]
    edge_features_batch = np.concatenate(all_edge_features, axis=0)  # [batch_size * num_agents, max_nodes, max_nodes, 1]
    
    # GPU 텐서로 변환
    node_obs_tensor = torch.from_numpy(node_obs_batch).float().to(device)
    adj_tensor = torch.from_numpy(adj_batch).float().to(device)
    entity_types_tensor = torch.from_numpy(entity_types_batch).long().to(device)
    edge_features_tensor = torch.from_numpy(edge_features_batch).float().to(device)
    
    return node_obs_tensor, adj_tensor, entity_types_tensor, edge_features_tensor