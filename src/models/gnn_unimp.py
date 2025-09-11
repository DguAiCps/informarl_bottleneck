"""
UniMP (Unified Message Passing) GNN implementation
논문과 동일한 attention 메커니즘 구현
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class UniMPLayer(nn.Module):
    """
    UniMP layer with multi-head attention
    논문 공식: x'_i = W1 * x_i + Σ α_ij * W2 * x_j
    α_ij = softmax((W3 * x_i) * (W4 * x_j + W5 * e_ij) / √c)
    """
    
    def __init__(self, hidden_dim: int = 64, num_heads: int = 4, edge_dim: int = 1, dropout: float = 0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.edge_dim = edge_dim
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Paper notation: W1, W2, W3, W4, W5
        self.W1 = nn.Linear(hidden_dim, hidden_dim)  # Self transform
        self.W2 = nn.Linear(hidden_dim, hidden_dim)  # Message transform
        self.W3 = nn.Linear(hidden_dim, hidden_dim)  # Query
        self.W4 = nn.Linear(hidden_dim, hidden_dim)  # Key  
        self.W5 = nn.Linear(edge_dim, hidden_dim)    # Edge transform
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x, adj, edge_features=None):
        """
        Args:
            x: [batch_size, max_nodes, hidden_dim] 노드 특성
            adj: [batch_size, max_nodes, max_nodes] 인접 행렬 (방향성)
            edge_features: [batch_size, max_nodes, max_nodes, edge_dim] 엣지 특성
            
        Returns:
            x_updated: [batch_size, max_nodes, hidden_dim] 업데이트된 노드 특성
        """
        batch_size, max_nodes, hidden_dim = x.shape
        
        # Self transform: W1 * x_i
        self_transform = self.W1(x)  # [batch, max_nodes, hidden_dim]
        
        # Multi-head attention
        # Query: W3 * x_i
        Q = self.W3(x).view(batch_size, max_nodes, self.num_heads, self.head_dim)  # [batch, nodes, heads, head_dim]
        Q = Q.transpose(1, 2)  # [batch, heads, nodes, head_dim]
        
        # Key: W4 * x_j + W5 * e_ij
        K = self.W4(x).view(batch_size, max_nodes, self.num_heads, self.head_dim)  # [batch, nodes, heads, head_dim]
        K = K.transpose(1, 2)  # [batch, heads, nodes, head_dim]
        
        # Edge contribution: W5 * e_ij
        if edge_features is not None:
            edge_contrib = self.W5(edge_features)  # [batch, nodes, nodes, hidden_dim]
            edge_contrib = edge_contrib.view(batch_size, max_nodes, max_nodes, self.num_heads, self.head_dim)
            edge_contrib = edge_contrib.permute(0, 3, 1, 2, 4)  # [batch, heads, nodes, nodes, head_dim]
            
            # K에 edge contribution 추가
            K = K.unsqueeze(2) + edge_contrib  # [batch, heads, nodes, nodes, head_dim]
        else:
            K = K.unsqueeze(2).expand(-1, -1, max_nodes, -1, -1)  # [batch, heads, nodes, nodes, head_dim]
        
        # Attention weights: α_ij = softmax((Q * K) / √c)
        scale = 1.0 / math.sqrt(self.head_dim)
        Q_expanded = Q.unsqueeze(3)  # [batch, heads, nodes, 1, head_dim]
        
        # Dot product attention
        attn_scores = (Q_expanded * K).sum(dim=-1) * scale  # [batch, heads, nodes, nodes]
        
        # Mask out non-adjacent nodes (adj == 0)
        adj_mask = adj.unsqueeze(1).expand(-1, self.num_heads, -1, -1)  # [batch, heads, nodes, nodes]
        attn_scores = attn_scores.masked_fill(adj_mask == 0, -1e9)
        
        # Softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)  # [batch, heads, nodes, nodes]
        attn_weights = self.dropout(attn_weights)
        
        # Message aggregation: Σ α_ij * W2 * x_j
        V = self.W2(x).view(batch_size, max_nodes, self.num_heads, self.head_dim)  # [batch, nodes, heads, head_dim]
        V = V.transpose(1, 2)  # [batch, heads, nodes, head_dim]
        
        # Weighted aggregation
        messages = torch.matmul(attn_weights, V)  # [batch, heads, nodes, head_dim]
        
        # Concatenate heads
        messages = messages.transpose(1, 2).contiguous()  # [batch, nodes, heads, head_dim]
        messages = messages.view(batch_size, max_nodes, hidden_dim)  # [batch, nodes, hidden_dim]
        
        # Final update: x'_i = W1 * x_i + messages
        x_updated = self_transform + messages
        
        # Residual connection + Layer norm
        x_updated = self.layer_norm(x + x_updated)
        
        return x_updated


class UniMPGNN(nn.Module):
    """
    UniMP Graph Neural Network
    논문의 attention 메커니즘을 완전히 구현
    """
    
    def __init__(self, node_input_dim: int = 6, hidden_dim: int = 64, 
                 num_layers: int = 2, num_heads: int = 4, num_entity_types: int = 3, 
                 embedding_size: int = 2, edge_dim: int = 1, dropout: float = 0.1,
                 actor_graph_aggr: str = "node"):
        super().__init__()
        
        self.node_input_dim = node_input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.actor_graph_aggr = actor_graph_aggr
        
        # Entity type embedding
        self.entity_embedding = nn.Embedding(num_entity_types, embedding_size)
        
        # Input projection
        self.input_proj = nn.Linear(node_input_dim + embedding_size, hidden_dim)
        
        # UniMP layers
        self.unimp_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.unimp_layers.append(
                UniMPLayer(hidden_dim, num_heads, edge_dim, dropout)
            )
        
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Output dimension
        self.out_dim = hidden_dim
        
    def forward(self, node_obs, adj, entity_types=None, edge_features=None, agent_id=None):
        """
        Args:
            node_obs: [batch_size, max_nodes, node_features] 노드 특성
            adj: [batch_size, max_nodes, max_nodes] 인접 행렬 (방향성)
            entity_types: [batch_size, max_nodes] entity type IDs
            edge_features: [batch_size, max_nodes, max_nodes, edge_dim] 엣지 특성 (거리 등)
            agent_id: [batch_size] 에이전트 ID
            
        Returns:
            aggregated_features: [batch_size, hidden_dim] 집약된 특성
        """
        batch_size, max_nodes, node_features = node_obs.shape
        
        # Entity type embedding
        if entity_types is None:
            entity_types = torch.zeros(batch_size, max_nodes, dtype=torch.long, device=node_obs.device)
        
        entity_embeds = self.entity_embedding(entity_types)  # [batch, max_nodes, embedding_size]
        
        # Input: node features + entity embedding
        x = torch.cat([node_obs, entity_embeds], dim=-1)  # [batch, max_nodes, node_features + embedding_size]
        
        # Input projection
        x = self.input_proj(x)  # [batch, max_nodes, hidden_dim]
        x = self.activation(x)
        x = self.dropout(x)
        
        # UniMP layers
        for layer in self.unimp_layers:
            x = layer(x, adj, edge_features)
        
        # Output aggregation
        if self.actor_graph_aggr == "node":
            # Actor: ego agent의 특성만 반환 (첫 번째 노드)
            ego_embeddings = x[:, 0, :]  # [batch, hidden_dim]
            return ego_embeddings
            
        elif self.actor_graph_aggr == "global":
            # Critic: 전체 노드의 평균 (논문 Section 3.3)
            # 마스킹: adj의 합이 0인 노드는 패딩
            node_mask = (adj.sum(dim=-1) > 0).float()  # [batch, max_nodes]
            
            # Global mean pooling: X_agg = (1/N) * Σ x_agg^(i)
            masked_x = x * node_mask.unsqueeze(-1)  # [batch, max_nodes, hidden_dim]
            global_embedding = masked_x.sum(dim=1) / (node_mask.sum(dim=1, keepdim=True) + 1e-6)
            
            return global_embedding  # [batch, hidden_dim]
        
        else:
            raise ValueError(f"Unknown aggregation type: {self.actor_graph_aggr}")


class UniMPActorGNN(UniMPGNN):
    """Actor용 UniMP GNN - node aggregation"""
    
    def __init__(self, **kwargs):
        kwargs['actor_graph_aggr'] = 'node'
        super().__init__(**kwargs)


class UniMPCriticGNN(UniMPGNN):
    """Critic용 UniMP GNN - global aggregation"""
    
    def __init__(self, **kwargs):
        kwargs['actor_graph_aggr'] = 'global'
        super().__init__(**kwargs)