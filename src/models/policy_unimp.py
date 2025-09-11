"""
UniMP Actor-Critic models with attention mechanism
올바른 InforMARL 구조: 각 에이전트가 자신의 그래프 생성 → 공유 GNN
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

from .gnn_unimp import UniMPGNN


class InforMARLModels:
    """InforMARL의 공유 GNN과 Actor/Critic 네트워크"""
    
    def __init__(self, local_obs_dim: int = 6, gnn_output_dim: int = 64, 
                 action_dim: int = 2, actor_hidden_dim: int = 64, 
                 critic_hidden_dim: int = 64, gnn_config: dict = None, device=None):
        
        self.device = device or torch.device('cpu')
        
        # 공유 GNN (하나만)
        if gnn_config is None:
            gnn_config = {
                'node_input_dim': 6,
                'hidden_dim': gnn_output_dim,
                'num_layers': 2,
                'num_heads': 4,
                'num_entity_types': 3,
                'embedding_size': 2,
                'edge_dim': 1,
                'dropout': 0.1,
                'actor_graph_aggr': 'node'
            }
        self.gnn = UniMPGNN(**gnn_config).to(self.device)
        
        # Actor 네트워크 (하나만)
        self.actor = ActorNet(
            input_dim=local_obs_dim + gnn_output_dim,
            action_dim=action_dim,
            hidden_dim=actor_hidden_dim
        ).to(self.device)
        
        # Critic 네트워크 (하나만) - GNN 평균만 사용
        self.critic = CriticNet(
            input_dim=gnn_output_dim,  # GNN 평균만!
            hidden_dim=critic_hidden_dim
        ).to(self.device)
    
    def get_all_parameters(self):
        """모든 네트워크의 파라미터 반환"""
        return list(self.gnn.parameters()) + list(self.actor.parameters()) + list(self.critic.parameters())


class ActorNet(nn.Module):
    """단순한 Actor MLP - local_obs + gnn_features -> action"""
    
    def __init__(self, input_dim: int, action_dim: int = 2, hidden_dim: int = 64):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, action_dim * 2)  # mean + log_std
        )
        
        # Action bounds
        self.action_scale = 1.0
        self.action_bias = 0.0
        
        # Log std bounds
        self.log_std_min = -20
        self.log_std_max = 2
    
    def forward(self, combined_input):
        """
        Args:
            combined_input: [batch_size, local_obs_dim + gnn_output_dim]
        Returns:
            action_mean, action_std: [batch_size, action_dim]
        """
        output = self.mlp(combined_input)
        action_mean, log_std = torch.chunk(output, 2, dim=-1)
        
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        action_std = torch.exp(log_std)
        action_mean = torch.tanh(action_mean) * self.action_scale + self.action_bias
        
        return action_mean, action_std


class CriticNet(nn.Module):
    """단순한 Critic MLP - avg_gnn_features만 -> value"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, avg_gnn_features):
        """
        Args:
            avg_gnn_features: [batch_size, gnn_output_dim] 모든 에이전트 GNN 특성의 평균
        Returns:
            value: [batch_size, 1]
        """
        return self.mlp(avg_gnn_features)


def create_centralized_obs(local_obs_batch):
    """
    지역 관찰들을 중앙화된 관찰로 변환
    
    Args:
        local_obs_batch: [batch_size, num_agents, local_obs_dim] 또는 [num_agents, local_obs_dim]
        
    Returns:
        centralized_obs: [batch_size, num_agents * local_obs_dim] 중앙화된 관찰
    """
    if local_obs_batch.dim() == 2:
        # [num_agents, local_obs_dim] -> [1, num_agents * local_obs_dim]
        return local_obs_batch.flatten().unsqueeze(0)
    elif local_obs_batch.dim() == 3:
        # [batch_size, num_agents, local_obs_dim] -> [batch_size, num_agents * local_obs_dim]
        batch_size, num_agents, local_obs_dim = local_obs_batch.shape
        return local_obs_batch.view(batch_size, num_agents * local_obs_dim)
    else:
        raise ValueError(f"Unexpected local_obs_batch shape: {local_obs_batch.shape}")


# 편의를 위한 별칭 (호환성)
AttentionActor = ActorNet
AttentionCritic = CriticNet