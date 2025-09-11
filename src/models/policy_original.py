"""
Actor-Critic models matching original InforMARL architecture
원본과 동일한 구조: local_obs + graph_features concatenation
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

from .gnn_original import ActorGNN, CriticGNN


class OriginalActor(nn.Module):
    """원본 InforMARL Actor: local_obs + graph_features -> action"""
    
    def __init__(self, local_obs_dim: int = 6, gnn_output_dim: int = 64, 
                 action_dim: int = 2, hidden_dim: int = 64, 
                 gnn_config: dict = None):
        super().__init__()
        
        self.local_obs_dim = local_obs_dim
        self.gnn_output_dim = gnn_output_dim
        self.action_dim = action_dim
        
        # GNN for graph feature extraction
        if gnn_config is None:
            gnn_config = {
                'node_input_dim': 6,
                'hidden_dim': 64,
                'num_layers': 2,
                'actor_graph_aggr': 'node'
            }
        self.gnn = ActorGNN(**gnn_config)
        
        # MLP for action prediction
        combined_input_dim = local_obs_dim + gnn_output_dim
        self.mlp = nn.Sequential(
            nn.Linear(combined_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * 2)  # mean + log_std
        )
        
        # Action bounds
        self.action_scale = 1.0
        self.action_bias = 0.0
        
    def forward(self, local_obs, node_obs, adj, entity_types=None, agent_id=None):
        """
        Forward pass
        
        Args:
            local_obs: [batch, local_obs_dim] 로컬 관측 (위치, 속도, 목표방향)
            node_obs: [batch, max_nodes, node_features] 그래프 노드 특성  
            adj: [batch, max_nodes, max_nodes] 인접 행렬 (방향성 있음)
            entity_types: [batch, max_nodes] entity type IDs
            agent_id: [batch] 에이전트 ID
            
        Returns:
            action_mean: [batch, action_dim]
            action_log_std: [batch, action_dim] 
        """
        # 1. GNN으로 그래프 특성 추출
        graph_features = self.gnn(node_obs, adj, entity_types, agent_id)  # [batch, gnn_output_dim]
        
        # 2. local_obs + graph_features concatenation (핵심!)
        combined_input = torch.cat([local_obs, graph_features], dim=1)  # [batch, local_obs_dim + gnn_output_dim]
        
        # 3. MLP로 액션 분포 파라미터 예측
        output = self.mlp(combined_input)  # [batch, action_dim * 2]
        
        # 4. mean과 log_std 분리
        action_mean, action_log_std = torch.split(output, self.action_dim, dim=1)
        
        # 5. Action scaling
        action_mean = self.action_scale * torch.tanh(action_mean) + self.action_bias
        action_log_std = torch.clamp(action_log_std, min=-20, max=2)
        action_std = torch.exp(action_log_std)
        
        return action_mean, action_std


class OriginalCritic(nn.Module):
    """원본 InforMARL Critic: centralized observation + global graph features -> value"""
    
    def __init__(self, centralized_obs_dim: int, gnn_output_dim: int = 64, 
                 hidden_dim: int = 64, gnn_config: dict = None):
        super().__init__()
        
        self.centralized_obs_dim = centralized_obs_dim
        self.gnn_output_dim = gnn_output_dim
        
        # GNN for global graph feature extraction  
        if gnn_config is None:
            gnn_config = {
                'node_input_dim': 6,
                'hidden_dim': 64,
                'num_layers': 2,
                'actor_graph_aggr': 'global'  # Critic는 global aggregation
            }
        self.gnn = CriticGNN(**gnn_config)
        
        # MLP for value prediction
        combined_input_dim = centralized_obs_dim + gnn_output_dim
        self.mlp = nn.Sequential(
            nn.Linear(combined_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(), 
            nn.Linear(hidden_dim, 1)  # Single value output
        )
        
    def forward(self, centralized_obs, node_obs, adj, entity_types=None, agent_id=None):
        """
        Forward pass
        
        Args:
            centralized_obs: [batch, centralized_obs_dim] 모든 에이전트의 정보를 합친 중앙화된 관측
            node_obs: [batch, max_nodes, node_features] 그래프 노드 특성
            adj: [batch, max_nodes, max_nodes] 인접 행렬 (방향성 있음)
            entity_types: [batch, max_nodes] entity type IDs
            agent_id: [batch] 에이전트 ID (사용되지 않을 수도 있음)
            
        Returns:
            value: [batch, 1] State value
        """
        # 1. GNN으로 global 그래프 특성 추출
        global_graph_features = self.gnn(node_obs, adj, entity_types, agent_id)  # [batch, gnn_output_dim]
        
        # 2. centralized_obs + global_graph_features concatenation
        combined_input = torch.cat([centralized_obs, global_graph_features], dim=1)
        
        # 3. MLP로 value 예측
        value = self.mlp(combined_input)  # [batch, 1]
        
        return value


def create_centralized_obs(agents_local_obs):
    """
    모든 에이전트의 local observation을 하나로 합치는 함수
    
    Args:
        agents_local_obs: List[np.ndarray] or [batch, num_agents, local_obs_dim]
        
    Returns:
        centralized_obs: [batch, num_agents * local_obs_dim]
    """
    if isinstance(agents_local_obs, list):
        # List of arrays인 경우
        return np.concatenate(agents_local_obs, axis=-1)
    else:
        # Batch tensor인 경우
        batch_size, num_agents, obs_dim = agents_local_obs.shape
        return agents_local_obs.reshape(batch_size, -1)  # [batch, num_agents * obs_dim]