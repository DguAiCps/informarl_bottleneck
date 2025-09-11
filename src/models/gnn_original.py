"""
Graph Neural Network implementation matching original InforMARL
원본과 동일한 고정크기 입력 처리 방식
"""
import torch
import torch.nn as nn
import numpy as np


class OriginalStyleGNN(nn.Module):
    """원본 InforMARL과 동일한 GNN 구현"""
    
    def __init__(self, node_input_dim: int = 6, hidden_dim: int = 64, 
                 num_layers: int = 2, num_entity_types: int = 3, 
                 embedding_size: int = 2, actor_graph_aggr: str = "node"):
        super().__init__()
        
        self.node_input_dim = node_input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.actor_graph_aggr = actor_graph_aggr
        
        # Entity type embedding (agent=0, landmark=1, obstacle=2)
        self.entity_embedding = nn.Embedding(num_entity_types, embedding_size)
        
        # Input projection layer
        self.input_proj = nn.Linear(node_input_dim + embedding_size, hidden_dim)
        
        # GNN layers (simple message passing)
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            self.gnn_layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        self.activation = nn.ReLU()
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Output dimension
        self.out_dim = hidden_dim
        
    def forward(self, node_obs, adj, entity_types=None, agent_id=None):
        """
        Forward pass
        
        Args:
            node_obs: [batch_size, max_nodes, node_features] 고정크기 노드 특성
            adj: [batch_size, max_nodes, max_nodes] 인접 행렬 (방향성 있음)
            entity_types: [batch_size, max_nodes] entity type IDs (agent=0, landmark=1, obstacle=2)
            agent_id: [batch_size] 에이전트 ID (어떤 에이전트의 관점인지)
            
        Returns:
            node_embeddings: [batch_size, hidden_dim] 또는 [batch_size, max_nodes, hidden_dim]
        """
        batch_size, max_nodes, node_features = node_obs.shape
        
        # Entity type 처리
        if entity_types is None:
            # 없으면 모두 agent로 가정
            entity_types = torch.zeros(batch_size, max_nodes, dtype=torch.long, device=node_obs.device)
        
        entity_embeds = self.entity_embedding(entity_types)  # [batch, max_nodes, embedding_size]
        
        # Node features + entity embedding concat
        x = torch.cat([node_obs, entity_embeds], dim=-1)  # [batch, max_nodes, node_features + embedding_size]
        
        # Input projection
        x = self.input_proj(x)  # [batch, max_nodes, hidden_dim]
        x = self.activation(x)
        
        # GNN message passing layers
        for layer in self.gnn_layers:
            # Message aggregation: 인접한 노드들의 특성을 평균
            # adj: [batch, max_nodes, max_nodes] -> normalize
            adj_norm = adj / (adj.sum(dim=-1, keepdim=True) + 1e-6)  # 정규화
            
            # Message passing: x_new = adj * x * W
            messages = torch.bmm(adj_norm, x)  # [batch, max_nodes, hidden_dim]
            x_new = layer(messages)  # [batch, max_nodes, hidden_dim]
            
            # Residual connection + activation
            x = self.activation(x + x_new)
            x = self.layer_norm(x)
        
        # Output aggregation
        if self.actor_graph_aggr == "node":
            # 각 에이전트의 첫 번째 노드 (ego agent) 특성만 반환
            if agent_id is not None:
                # agent_id별로 해당하는 노드 선택 (단순화: 첫 번째 노드로 가정)
                ego_embeddings = x[:, 0, :]  # [batch, hidden_dim]
            else:
                ego_embeddings = x[:, 0, :]  # [batch, hidden_dim]
            return ego_embeddings
            
        elif self.actor_graph_aggr == "global":
            # Global average pooling (마스킹 고려)
            # adj의 합이 0인 노드는 빈 노드 (패딩)
            node_mask = (adj.sum(dim=-1) > 0).float()  # [batch, max_nodes]
            
            # 마스킹된 평균 계산
            masked_x = x * node_mask.unsqueeze(-1)  # [batch, max_nodes, hidden_dim]
            global_embedding = masked_x.sum(dim=1) / (node_mask.sum(dim=1, keepdim=True) + 1e-6)
            
            return global_embedding  # [batch, hidden_dim]
        
        else:
            raise ValueError(f"Unknown aggregation type: {self.actor_graph_aggr}")


class CriticGNN(OriginalStyleGNN):
    """Critic용 GNN - global aggregation"""
    
    def __init__(self, **kwargs):
        kwargs['actor_graph_aggr'] = 'global'  # Critic는 항상 global
        super().__init__(**kwargs)


class ActorGNN(OriginalStyleGNN):
    """Actor용 GNN - node aggregation"""
    
    def __init__(self, **kwargs):
        kwargs['actor_graph_aggr'] = 'node'  # Actor는 항상 node
        super().__init__(**kwargs)