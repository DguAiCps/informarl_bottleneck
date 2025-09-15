"""
Step-wise Learning UniMP InforMARL
스텝 단위 학습으로 더 안정적이고 효율적인 훈련
"""
import sys
import os
import torch
import numpy as np
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.env.bottleneck_env import CleanBottleneckEnv
from src.models.policy_unimp import InforMARLModels
from src.env.graph_builder_unimp import build_unimp_graph_observations
from src.utils.rollout_buffer import RolloutBuffer, MiniBatchSampler

# =================== 하드코딩된 설정 ===================
# 환경 설정
NUM_AGENTS = 4
CORRIDOR_WIDTH = 20.0
CORRIDOR_HEIGHT = 10.0  
BOTTLENECK_WIDTH = 2.0
BOTTLENECK_POS = 10.0
AGENT_RADIUS = 0.5
SENSING_RADIUS = 7.0
MAX_SPEED = 1.0
MAX_TIMESTEPS = 300

# 스텝 단위 훈련 설정
TOTAL_TIMESTEPS = 2000000  # 원본 InforMARL과 동일
ROLLOUT_LENGTH = 2048      # 롤아웃 버퍼 크기 (스텝 수)
LEARNING_RATE = 3e-4
MINI_BATCH_SIZE = 64       # 미니배치 크기
PPO_EPOCHS = 4            # PPO 업데이트 횟수
UPDATE_FREQUENCY = ROLLOUT_LENGTH  # 롤아웃 버퍼가 가득 찰 때마다 업데이트
EVAL_FREQUENCY = 2048    # 평가 빈도 (스텝 단위) - 매 업데이트마다

# 모델 설정 (UniMP 기반)
LOCAL_OBS_DIM = 6
ACTION_DIM = 2
CENTRALIZED_OBS_DIM = NUM_AGENTS * LOCAL_OBS_DIM
GNN_OUTPUT_DIM = 64
ACTOR_HIDDEN_DIM = 64
CRITIC_HIDDEN_DIM = 64

# UniMP 하이퍼파라미터
NUM_HEADS = 4
NUM_LAYERS = 2
DROPOUT = 0.1

# PPO 하이퍼파라미터
CLIP_EPSILON = 0.2
VALUE_LOSS_COEF = 0.5
ENTROPY_COEF = 0.01
GAMMA = 0.99
GAE_LAMBDA = 0.95
MAX_GRAD_NORM = 0.5

# GPU 및 성능 설정
USE_GPU_PHYSICS = True
INCLUDE_OBSTACLES = True
DEVICE = torch.device('cuda')


def main():
    """메인 실행 함수 - 스텝 단위 학습"""
    print("=== Step-wise UniMP InforMARL 시작 ===")
    print(f"Device: {DEVICE}")
    print(f"에이전트 수: {NUM_AGENTS}")
    print(f"총 스텝: {TOTAL_TIMESTEPS:,}")
    print(f"롤아웃 길이: {ROLLOUT_LENGTH}")
    print(f"미니배치 크기: {MINI_BATCH_SIZE}")
    print(f"PPO epochs: {PPO_EPOCHS}")
    print()
    
    # 환경 초기화
    env = CleanBottleneckEnv(
        num_agents=NUM_AGENTS,
        corridor_width=CORRIDOR_WIDTH,
        corridor_height=CORRIDOR_HEIGHT,
        bottleneck_width=BOTTLENECK_WIDTH,
        bottleneck_pos=BOTTLENECK_POS,
        agent_radius=AGENT_RADIUS,
        sensing_radius=SENSING_RADIUS,
        max_speed=MAX_SPEED,
        max_timesteps=MAX_TIMESTEPS,
        device=DEVICE,
        use_gpu_physics=USE_GPU_PHYSICS,
        include_obstacles=INCLUDE_OBSTACLES
    )
    
    # InforMARL 모델 초기화 (GNN + Actor + Critic)
    gnn_config = {
        'node_input_dim': 6,
        'hidden_dim': GNN_OUTPUT_DIM,
        'num_layers': NUM_LAYERS,
        'num_heads': NUM_HEADS,
        'num_entity_types': 3,
        'embedding_size': 2,
        'edge_dim': 1,
        'dropout': DROPOUT
    }
    
    models = InforMARLModels(
        local_obs_dim=LOCAL_OBS_DIM,
        gnn_output_dim=GNN_OUTPUT_DIM,
        action_dim=ACTION_DIM,
        actor_hidden_dim=ACTOR_HIDDEN_DIM,
        critic_hidden_dim=CRITIC_HIDDEN_DIM,
        gnn_config=gnn_config,
        device=DEVICE
    )
    
    # 옵티마이저 (모든 네트워크 포함)
    optimizer = torch.optim.Adam(
        models.get_all_parameters(),
        lr=LEARNING_RATE,
        eps=1e-5
    )
    
    # 롤아웃 버퍼 초기화
    buffer = RolloutBuffer(
        buffer_size=ROLLOUT_LENGTH,
        num_agents=NUM_AGENTS,
        local_obs_dim=LOCAL_OBS_DIM,
        action_dim=ACTION_DIM,
        max_nodes=30,
        edge_dim=1,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        device=DEVICE
    )
    
    # 훈련 상태 변수
    global_step = 0
    episode_count = 0
    episode_rewards = []
    episode_success_rates = []
    episode_lengths = []
    
    # 환경 리셋
    env.reset()
    episode_reward = 0
    episode_length = 0
    
    print("스텝 단위 훈련 시작...")
    
    while global_step < TOTAL_TIMESTEPS:
        # 롤아웃 수집
        for rollout_step in range(ROLLOUT_LENGTH):
            # 모든 에이전트의 local observation 수집
            local_obs_list = []
            for i in range(NUM_AGENTS):
                local_obs = env.get_local_observation(i)
                local_obs_list.append(local_obs)
            
            # 배치 형태로 변환
            local_obs_batch = torch.tensor(local_obs_list, dtype=torch.float32).to(DEVICE)
            
            # UniMP 그래프 데이터 생성
            node_obs_np, adj_np, entity_types_np, edge_features_np = build_unimp_graph_observations(
                env.agents, env.landmarks, env.obstacles, SENSING_RADIUS
            )
            
            # 텐서 변환
            node_obs_batch = torch.tensor(node_obs_np, dtype=torch.float32).to(DEVICE)
            adj_batch = torch.tensor(adj_np, dtype=torch.float32).to(DEVICE)
            entity_types_batch = torch.tensor(entity_types_np, dtype=torch.long).to(DEVICE)
            edge_features_batch = torch.tensor(edge_features_np, dtype=torch.float32).to(DEVICE)
            
            # 중앙화된 observation은 사용하지 않음 (InforMARL에서는 GNN 평균만 사용)
            
            with torch.no_grad():
                # 1. 각 에이전트의 그래프를 GNN에 통과시켜 특성 추출
                # node_obs_batch: [num_agents, max_nodes, 6]
                # adj_batch: [num_agents, max_nodes, max_nodes] 
                # 각 에이전트별로 자신의 그래프 처리
                
                all_agent_gnn_features = []
                for agent_id in range(NUM_AGENTS):
                    # 해당 에이전트의 그래프만 추출
                    agent_node_obs = node_obs_batch[agent_id:agent_id+1]      # [1, max_nodes, 6]
                    agent_adj = adj_batch[agent_id:agent_id+1]                # [1, max_nodes, max_nodes]
                    agent_entity_types = entity_types_batch[agent_id:agent_id+1]  # [1, max_nodes]
                    agent_edge_features = edge_features_batch[agent_id:agent_id+1]  # [1, max_nodes, max_nodes, 1]
                    
                    # 공유 GNN으로 처리
                    agent_gnn_output = models.gnn(
                        agent_node_obs, agent_adj, agent_entity_types, 
                        agent_edge_features, agent_id
                    )  # [1, gnn_output_dim]
                    all_agent_gnn_features.append(agent_gnn_output)
                
                # 2. Actor: 각 에이전트별로 행동 선택
                action_means_list = []
                action_stds_list = []
                for i in range(NUM_AGENTS):
                    agent_local_obs = local_obs_batch[i:i+1]      # [1, local_obs_dim]
                    agent_gnn_features = all_agent_gnn_features[i]  # [1, gnn_output_dim]
                    
                    # 로컬 관측 + GNN 특성 결합
                    combined_input = torch.cat([agent_local_obs, agent_gnn_features], dim=-1)
                    action_mean, action_std = models.actor(combined_input)
                    action_means_list.append(action_mean)
                    action_stds_list.append(action_std)
                
                action_means = torch.cat(action_means_list, dim=0)  # [NUM_AGENTS, action_dim]
                action_stds = torch.cat(action_stds_list, dim=0)    # [NUM_AGENTS, action_dim]
                
                action_dist = torch.distributions.Normal(action_means, action_stds)
                actions_tensor = action_dist.sample()
                actions_tensor = torch.clamp(actions_tensor, -1.0, 1.0)
                log_probs_tensor = action_dist.log_prob(actions_tensor).sum(dim=-1)
                
                # 3. Critic: 모든 에이전트 GNN 특성의 평균으로 가치 추정
                avg_gnn_features = torch.stack(all_agent_gnn_features).mean(dim=0)  # [1, gnn_output_dim]
                values_tensor = models.critic(avg_gnn_features)  # GNN 평균만 사용!
                
                actions = actions_tensor.cpu().numpy().tolist()
                log_probs = log_probs_tensor.cpu().tolist()
                values = [values_tensor.cpu().item()] * NUM_AGENTS
            
            # 환경 스텝
            _, rewards, done, info = env.step(actions)
            
            # 버퍼에 데이터 저장
            buffer.store(
                local_obs=local_obs_list,
                actions=actions,
                rewards=rewards,
                values=values,
                log_probs=log_probs,
                node_obs=node_obs_np,
                adj=adj_np,
                entity_types=entity_types_np,
                edge_features=edge_features_np
            )
            
            # 상태 업데이트 - 그래프는 매번 새로 생성하므로 불필요
            episode_reward += sum(rewards)
            episode_length += 1
            global_step += 1
            
            # 에피소드 종료 처리
            if done or episode_length >= MAX_TIMESTEPS:
                # 마지막 가치 계산 (bootstrap value)
                if not done:
                    # 마지막 상태의 GNN 특성들 계산
                    node_obs_np_last, adj_np_last, entity_types_np_last, edge_features_np_last = build_unimp_graph_observations(
                        env.agents, env.landmarks, env.obstacles, SENSING_RADIUS
                    )
                    
                    node_obs_batch_last = torch.tensor(node_obs_np_last, dtype=torch.float32).to(DEVICE)
                    adj_batch_last = torch.tensor(adj_np_last, dtype=torch.float32).to(DEVICE)
                    entity_types_batch_last = torch.tensor(entity_types_np_last, dtype=torch.long).to(DEVICE)
                    edge_features_batch_last = torch.tensor(edge_features_np_last, dtype=torch.float32).to(DEVICE)
                    
                    with torch.no_grad():
                        # 각 에이전트의 GNN 특성 계산
                        all_agent_gnn_features_last = []
                        for agent_id in range(NUM_AGENTS):
                            agent_node_obs = node_obs_batch_last[agent_id:agent_id+1]
                            agent_adj = adj_batch_last[agent_id:agent_id+1]
                            agent_entity_types = entity_types_batch_last[agent_id:agent_id+1]
                            agent_edge_features = edge_features_batch_last[agent_id:agent_id+1]
                            
                            agent_gnn_output = models.gnn(
                                agent_node_obs, agent_adj, agent_entity_types,
                                agent_edge_features, agent_id
                            )
                            all_agent_gnn_features_last.append(agent_gnn_output)
                        
                        # 평균으로 마지막 가치 계산
                        avg_gnn_features_last = torch.stack(all_agent_gnn_features_last).mean(dim=0)
                        last_value = models.critic(avg_gnn_features_last).cpu().item()
                else:
                    last_value = 0.0
                
                # 경로 종료 및 GAE 계산
                buffer.finish_path(last_value)
                
                # 에피소드 통계 저장
                episode_rewards.append(episode_reward)
                episode_success_rates.append(info['success_rate'])
                episode_lengths.append(episode_length)
                episode_count += 1
                
                # 환경 리셋
                env.reset()
                episode_reward = 0
                episode_length = 0
            
            # 버퍼가 가득 차면 업데이트
            if buffer.is_full():
                break
        
        # PPO 업데이트
        if buffer.is_full():
            update_policy_stepwise(models, optimizer, buffer)
            buffer.reset()
        
        # 평가 및 출력 (300스텝마다 무조건)
        if global_step % EVAL_FREQUENCY == 0:
            if len(episode_rewards) > 0:
                recent_episodes = min(100, len(episode_rewards))
                avg_reward = np.mean(episode_rewards[-recent_episodes:])
                avg_success = np.mean(episode_success_rates[-recent_episodes:])
                avg_length = np.mean(episode_lengths[-recent_episodes:])
                
                print(f"Step {global_step:,}/{TOTAL_TIMESTEPS:,}: "
                      f"Episodes {episode_count}, "
                      f"Avg Reward = {avg_reward:.2f}, "
                      f"Success Rate = {avg_success:.1f}%, "
                      f"Avg Length = {avg_length:.1f} [Step-wise]")
            else:
                print(f"Step {global_step:,}/{TOTAL_TIMESTEPS:,}: "
                      f"Episodes {episode_count} (진행중), "
                      f"현재 보상 = {episode_reward:.2f} [Step-wise]")
    
    print(f"\n=== 스텝 단위 훈련 완료 ===")
    print(f"총 에피소드: {episode_count}")
    if len(episode_rewards) > 0:
        print(f"최종 평균 보상: {np.mean(episode_rewards[-10:]):.2f}")
        print(f"최종 성공률: {np.mean(episode_success_rates[-10:]):.1f}%")
    
    # 데모 실행
    if input("\n스텝 단위 학습 데모를 실행할까요? (y/n): ").lower() == 'y':
        run_stepwise_demo(env, models)


def update_policy_stepwise(models, optimizer, buffer):
    """스텝 단위 PPO 업데이트"""
    data = buffer.get()
    
    batch_size = data['local_obs'].shape[0]
    sampler = MiniBatchSampler(batch_size, MINI_BATCH_SIZE)
    
    # PPO 여러 epoch 업데이트
    for _ in range(PPO_EPOCHS):
        for indices in sampler:
            # 미니배치 데이터 추출
            mini_batch = {key: value[indices] for key, value in data.items()}
            
            # 1. 공유 GNN으로 모든 에이전트의 특성 계산
            batch_size_mini = mini_batch['local_obs'].shape[0]
            all_agent_gnn_features = []
            
            # 각 에이전트의 그래프를 개별적으로 처리
            samples_per_agent = batch_size_mini // NUM_AGENTS
            
            for agent_id in range(NUM_AGENTS):
                agent_start_idx = agent_id * samples_per_agent
                agent_end_idx = (agent_id + 1) * samples_per_agent
                
                agent_node_obs = mini_batch['node_obs'][agent_start_idx:agent_end_idx]
                agent_adj = mini_batch['adj'][agent_start_idx:agent_end_idx]
                agent_entity_types = mini_batch['entity_types'][agent_start_idx:agent_end_idx]
                agent_edge_features = mini_batch['edge_features'][agent_start_idx:agent_end_idx]
                
                # 각 샘플별로 GNN 처리
                agent_gnn_outputs = []
                for sample_idx in range(samples_per_agent):
                    sample_gnn_output = models.gnn(
                        agent_node_obs[sample_idx:sample_idx+1],
                        agent_adj[sample_idx:sample_idx+1],
                        agent_entity_types[sample_idx:sample_idx+1],
                        agent_edge_features[sample_idx:sample_idx+1],
                        agent_id
                    )
                    agent_gnn_outputs.append(sample_gnn_output)
                
                agent_gnn_features = torch.cat(agent_gnn_outputs, dim=0)
                all_agent_gnn_features.append(agent_gnn_features)
            
            # 2. Actor forward pass
            action_means_list = []
            action_stds_list = []
            
            for i in range(NUM_AGENTS):
                # 각 에이전트의 로컬 관측과 GNN 특성 (순서대로 매칭)
                agent_start_idx = i * samples_per_agent
                agent_end_idx = (i + 1) * samples_per_agent
                
                agent_local_obs = mini_batch['local_obs'][agent_start_idx:agent_end_idx]
                agent_gnn_features = all_agent_gnn_features[i]  # 이미 올바른 크기
                
                combined_input = torch.cat([agent_local_obs, agent_gnn_features], dim=-1)
                action_mean, action_std = models.actor(combined_input)
                action_means_list.append(action_mean)
                action_stds_list.append(action_std)
            
            action_means = torch.cat(action_means_list, dim=0)  # [batch_size_mini, action_dim]
            action_stds = torch.cat(action_stds_list, dim=0)    # [batch_size_mini, action_dim]
            
            action_dist = torch.distributions.Normal(action_means, action_stds)
            new_log_probs = action_dist.log_prob(mini_batch['actions']).sum(dim=-1)
            entropy = action_dist.entropy().sum(dim=-1)
            
            # 3. Critic forward pass  
            # 각 샘플에 대해 GNN 특성들의 평균 계산 (올바른 인덱싱)
            avg_gnn_features_list = []
            for sample_idx in range(samples_per_agent):
                sample_gnn_features = []
                for agent_id in range(NUM_AGENTS):
                    sample_gnn_features.append(all_agent_gnn_features[agent_id][sample_idx:sample_idx+1])
                avg_gnn_features = torch.stack(sample_gnn_features).mean(dim=0)  # [1, gnn_output_dim]
                avg_gnn_features_list.append(avg_gnn_features)
            
            # 각 샘플을 NUM_AGENTS번 반복 (각 에이전트마다 같은 value)
            avg_gnn_features_per_sample = torch.cat(avg_gnn_features_list, dim=0)  # [samples_per_agent, gnn_output_dim]
            avg_gnn_features_batch = avg_gnn_features_per_sample.repeat(NUM_AGENTS, 1)  # [batch_size_mini, gnn_output_dim]
            new_values = models.critic(avg_gnn_features_batch).squeeze()  # GNN 평균만 사용!
            
            # PPO 손실 계산
            ratio = torch.exp(new_log_probs - mini_batch['old_log_probs'])
            
            surr1 = ratio * mini_batch['advantages']
            surr2 = torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * mini_batch['advantages']
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss (clipped)
            value_pred_clipped = mini_batch['old_values'] + torch.clamp(
                new_values - mini_batch['old_values'], -CLIP_EPSILON, CLIP_EPSILON
            )
            value_loss_1 = (new_values - mini_batch['returns']).pow(2)
            value_loss_2 = (value_pred_clipped - mini_batch['returns']).pow(2)
            value_loss = VALUE_LOSS_COEF * torch.max(value_loss_1, value_loss_2).mean()
            
            entropy_loss = -ENTROPY_COEF * entropy.mean()
            
            total_loss = policy_loss + value_loss + entropy_loss
            
            # 역전파
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                models.get_all_parameters(),
                MAX_GRAD_NORM
            )
            optimizer.step()


def run_stepwise_demo(env, models):
    """스텝 단위 학습 데모 실행"""
    print("\n=== Step-wise Learning 데모 시작 ===")
    
    for episode in range(3):
        print(f"데모 에피소드 {episode + 1}")
        env.reset()
        episode_reward = 0
        
        for step in range(MAX_TIMESTEPS):
            # 모든 에이전트의 local observation 수집
            local_obs_list = []
            for i in range(NUM_AGENTS):
                local_obs = env.get_local_observation(i)
                local_obs_list.append(local_obs)
            
            # 배치 형태로 변환
            local_obs_batch = torch.tensor(local_obs_list, dtype=torch.float32).to(DEVICE)
            
            # UniMP 그래프 데이터 생성
            node_obs_np, adj_np, entity_types_np, edge_features_np = build_unimp_graph_observations(
                env.agents, env.landmarks, env.obstacles, SENSING_RADIUS
            )
            
            # 텐서 변환
            node_obs_batch = torch.tensor(node_obs_np, dtype=torch.float32).to(DEVICE)
            adj_batch = torch.tensor(adj_np, dtype=torch.float32).to(DEVICE)
            entity_types_batch = torch.tensor(entity_types_np, dtype=torch.long).to(DEVICE)
            edge_features_batch = torch.tensor(edge_features_np, dtype=torch.float32).to(DEVICE)
            
            with torch.no_grad():
                # 평가 모드: 각 에이전트의 그래프를 개별 처리
                all_agent_gnn_features = []
                for agent_id in range(NUM_AGENTS):
                    agent_node_obs = node_obs_batch[agent_id:agent_id+1]
                    agent_adj = adj_batch[agent_id:agent_id+1]
                    agent_entity_types = entity_types_batch[agent_id:agent_id+1]
                    agent_edge_features = edge_features_batch[agent_id:agent_id+1]
                    
                    agent_gnn_output = models.gnn(
                        agent_node_obs, agent_adj, agent_entity_types,
                        agent_edge_features, agent_id
                    )
                    all_agent_gnn_features.append(agent_gnn_output)
                
                # Actor로 행동 선택
                action_means_list = []
                for i in range(NUM_AGENTS):
                    agent_local_obs = local_obs_batch[i:i+1]
                    agent_gnn_features = all_agent_gnn_features[i]
                    
                    combined_input = torch.cat([agent_local_obs, agent_gnn_features], dim=-1)
                    action_means, _ = models.actor(combined_input)
                    action_means_list.append(action_means)
                
                actions_tensor = torch.cat(action_means_list, dim=0)
                actions_tensor = torch.clamp(actions_tensor, -1.0, 1.0)
                actions = actions_tensor.cpu().numpy().tolist()
            
            # 환경 스텝
            _, rewards, done, info = env.step(actions)
            episode_reward += sum(rewards)
            
            # 렌더링
            env.render()
            time.sleep(0.05)
            
            if done:
                print(f"  완료! 스텝: {step + 1}, 보상: {episode_reward:.2f}, "
                      f"성공률: {info['success_rate']:.1f}% [Step-wise]")
                break
        
        time.sleep(1.0)
    
    env.close()


if __name__ == "__main__":
    main()