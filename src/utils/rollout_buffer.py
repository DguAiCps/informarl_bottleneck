"""
Step-wise Rollout Buffer for PPO
스텝 단위 학습을 위한 롤아웃 버퍼 구현
"""
import torch
import numpy as np
from typing import Dict, List, Tuple, Any


class RolloutBuffer:
    """PPO를 위한 스텝 단위 롤아웃 버퍼"""
    
    def __init__(self, buffer_size: int, num_agents: int, local_obs_dim: int, 
                 action_dim: int, max_nodes: int = 30, edge_dim: int = 1,
                 gamma: float = 0.99, gae_lambda: float = 0.95, device=None):
        
        self.buffer_size = buffer_size
        self.num_agents = num_agents
        self.local_obs_dim = local_obs_dim
        self.action_dim = action_dim
        self.max_nodes = max_nodes
        self.edge_dim = edge_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device or torch.device('cpu')
        
        self.ptr = 0
        self.path_start_idx = 0
        self.max_size = buffer_size
        
        # 버퍼 초기화
        self.reset()
    
    def reset(self):
        """버퍼 초기화"""
        # 각 에이전트별 데이터 저장
        self.local_obs = np.zeros((self.max_size, self.num_agents, self.local_obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.max_size, self.num_agents, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.max_size, self.num_agents), dtype=np.float32)
        self.values = np.zeros((self.max_size, self.num_agents), dtype=np.float32)
        self.log_probs = np.zeros((self.max_size, self.num_agents), dtype=np.float32)
        self.dones = np.zeros((self.max_size,), dtype=np.bool_)
        
        # 그래프 데이터 (에이전트별)
        self.node_obs = np.zeros((self.max_size, self.num_agents, self.max_nodes, 6), dtype=np.float32)
        self.adj = np.zeros((self.max_size, self.num_agents, self.max_nodes, self.max_nodes), dtype=np.float32)
        self.entity_types = np.zeros((self.max_size, self.num_agents, self.max_nodes), dtype=np.int32)
        self.edge_features = np.zeros((self.max_size, self.num_agents, self.max_nodes, self.max_nodes, self.edge_dim), dtype=np.float32)
        
        # 중앙화된 관찰 (Critic용)
        # centralized_obs 제거 - InforMARL에서는 사용하지 않음
        
        # GAE 계산용
        self.advantages = np.zeros((self.max_size, self.num_agents), dtype=np.float32)
        self.returns = np.zeros((self.max_size, self.num_agents), dtype=np.float32)
        
        self.ptr = 0
        self.path_start_idx = 0
    
    def store(self, local_obs: List[List[float]], actions: List[List[float]], 
              rewards: List[float], values: List[float], log_probs: List[float],
              node_obs: np.ndarray, adj: np.ndarray, entity_types: np.ndarray,
              edge_features: np.ndarray):
        """스텝 데이터 저장"""
        assert self.ptr < self.max_size, "Buffer overflow"
        
        # 리스트를 numpy 배열로 변환
        self.local_obs[self.ptr] = np.array(local_obs)
        self.actions[self.ptr] = np.array(actions)
        self.rewards[self.ptr] = np.array(rewards)
        self.values[self.ptr] = np.array(values)
        self.log_probs[self.ptr] = np.array(log_probs)
        
        # 그래프 데이터
        self.node_obs[self.ptr] = node_obs
        self.adj[self.ptr] = adj
        self.entity_types[self.ptr] = entity_types
        self.edge_features[self.ptr] = edge_features
        # centralized_obs 저장하지 않음
        
        self.ptr += 1
    
    def finish_path(self, last_value: float = 0.0):
        """에피소드 종료 시 호출하여 advantage와 return 계산"""
        path_slice = slice(self.path_start_idx, self.ptr)
        
        # 각 에이전트별로 GAE 계산
        for agent_idx in range(self.num_agents):
            rewards = self.rewards[path_slice, agent_idx]
            values = self.values[path_slice, agent_idx]
            
            # 마지막 value 추가 (bootstrap value)
            values_with_bootstrap = np.append(values, last_value)
            
            # GAE 계산
            deltas = rewards + self.gamma * values_with_bootstrap[1:] - values_with_bootstrap[:-1]
            
            advantages = np.zeros_like(rewards)
            last_gae_lam = 0
            for t in reversed(range(len(rewards))):
                last_gae_lam = deltas[t] + self.gamma * self.gae_lambda * last_gae_lam
                advantages[t] = last_gae_lam
            
            # Returns = advantages + values
            returns = advantages + values
            
            self.advantages[path_slice, agent_idx] = advantages
            self.returns[path_slice, agent_idx] = returns
        
        self.path_start_idx = self.ptr
    
    def get(self) -> Dict[str, torch.Tensor]:
        """버퍼의 현재 데이터를 반환 (학습용)"""
        assert self.ptr > 0, "Buffer is empty"
        
        # Advantage 정규화 (전체 에이전트에 대해)
        advantages_flat = self.advantages[:self.ptr].flatten()
        adv_mean = advantages_flat.mean()
        adv_std = advantages_flat.std()
        self.advantages[:self.ptr] = (self.advantages[:self.ptr] - adv_mean) / (adv_std + 1e-8)
        
        # 원본 InforMARL 방식: 구조를 유지하면서 flatten
        episode_length = self.ptr
        num_agents = self.num_agents
        batch_size = episode_length * num_agents

        # Agent ID 생성 (원본 InforMARL 방식)
        agent_ids = np.tile(np.arange(num_agents), episode_length)  # [0,1,2,3, 0,1,2,3, ...]

        data = {
            'local_obs': torch.from_numpy(self.local_obs[:self.ptr].reshape(batch_size, -1)).float().to(self.device),
            'actions': torch.from_numpy(self.actions[:self.ptr].reshape(batch_size, -1)).float().to(self.device),
            'advantages': torch.from_numpy(self.advantages[:self.ptr].reshape(batch_size)).float().to(self.device),
            'returns': torch.from_numpy(self.returns[:self.ptr].reshape(batch_size)).float().to(self.device),
            'old_log_probs': torch.from_numpy(self.log_probs[:self.ptr].reshape(batch_size)).float().to(self.device),
            'old_values': torch.from_numpy(self.values[:self.ptr].reshape(batch_size)).float().to(self.device),

            # 그래프 데이터
            'node_obs': torch.from_numpy(self.node_obs[:self.ptr].reshape(batch_size, self.max_nodes, -1)).float().to(self.device),
            'adj': torch.from_numpy(self.adj[:self.ptr].reshape(batch_size, self.max_nodes, self.max_nodes)).float().to(self.device),
            'entity_types': torch.from_numpy(self.entity_types[:self.ptr].reshape(batch_size, self.max_nodes)).long().to(self.device),
            'edge_features': torch.from_numpy(self.edge_features[:self.ptr].reshape(batch_size, self.max_nodes, self.max_nodes, self.edge_dim)).float().to(self.device),

            # Agent ID 추가 (중요!)
            'agent_ids': torch.from_numpy(agent_ids).long().to(self.device),
        }
        
        return data
    
    def size(self) -> int:
        """현재 버퍼 크기 반환"""
        return self.ptr
    
    def is_full(self) -> bool:
        """버퍼가 가득 찼는지 확인"""
        return self.ptr >= self.max_size

    def has_data(self) -> bool:
        """버퍼에 데이터가 있는지 확인"""
        return self.ptr > self.path_start_idx


class StepWiseSampler:
    """스텝 단위 미니배치 샘플링 클래스 - 논문 의도에 맞게 동일 타임스텝 에이전트들을 함께 샘플링"""

    def __init__(self, total_steps: int, num_agents: int, mini_batch_size: int):
        self.total_steps = total_steps
        self.num_agents = num_agents
        self.mini_batch_size = mini_batch_size

        # 자동 계산
        assert mini_batch_size % num_agents == 0, f"Mini-batch size {mini_batch_size} must be divisible by num_agents {num_agents}"
        self.steps_per_batch = mini_batch_size // num_agents
        self.num_mini_batches = total_steps // self.steps_per_batch

    def __iter__(self):
        """스텝 단위로 미니배치 인덱스 반환"""
        # 스텝을 랜덤하게 섞기
        shuffled_steps = torch.randperm(self.total_steps)

        for i in range(self.num_mini_batches):
            # 선택된 스텝들
            start_step = i * self.steps_per_batch
            end_step = start_step + self.steps_per_batch
            selected_steps = shuffled_steps[start_step:end_step]

            # 각 스텝의 모든 에이전트 인덱스 생성
            indices = []
            for step in selected_steps:
                base_idx = step * self.num_agents
                for agent in range(self.num_agents):
                    indices.append(base_idx + agent)

            yield torch.tensor(indices, dtype=torch.long)