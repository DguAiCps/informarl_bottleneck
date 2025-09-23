"""
Step-wise Learning UniMP InforMARL
스텝 단위 학습으로 더 안정적이고 효율적인 훈련
"""
import sys
import os
import torch
import numpy as np
import time
import argparse

# Add src and project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from src.env.bottleneck_env import CleanBottleneckEnv
from src.models.policy_unimp import InforMARLModels
from src.env.graph_builder_unimp import build_unimp_graph_observations
from src.utils.rollout_buffer import RolloutBuffer
from utils.model_saver import ModelSaver, get_training_config
import multiprocessing as mp
from multiprocessing import Process, Pipe
from typing import List, Tuple, Dict, Any

# =================== 하드코딩된 설정 ===================
# 환경 설정
NUM_AGENTS = 4
CORRIDOR_WIDTH = 20.0
CORRIDOR_HEIGHT = 10.0  
BOTTLENECK_WIDTH = 6.0
BOTTLENECK_POS = 10.0
AGENT_RADIUS = 0.5
SENSING_RADIUS = 7.0
MAX_SPEED = 1.0
MAX_TIMESTEPS = 300

# 에피소드 기반 훈련 설정
TOTAL_TIMESTEPS = 20000000  # 원본 InforMARL과 동일
EPISODE_LENGTH = 300       # 에피소드 길이 (원본과 동일)
N_ROLLOUT_THREADS = 128   # 병렬 환경 수 (원본 InforMARL과 동일)
USE_PARALLEL_ENVS = True  # 병렬 환경 사용 여부
SAMPLING_METHOD = "random"  # "random" or "stepwise"
LEARNING_RATE = 3e-4
MINI_BATCH_SIZE = 64       # 미니배치 크기
PPO_EPOCHS = 4            # PPO 업데이트 횟수
UPDATE_FREQUENCY = EPISODE_LENGTH  # 에피소드 완료 후 업데이트
EVAL_FREQUENCY = 10    # 평가 빈도 (에피소드 단위)

# 모델 설정 (UniMP 기반)
LOCAL_OBS_DIM = 6
ACTION_DIM = 2              # [dv, dw] - 속도 변화량
CENTRALIZED_OBS_DIM = NUM_AGENTS * LOCAL_OBS_DIM
GNN_OUTPUT_DIM = 64
ACTOR_HIDDEN_DIM = 64
CRITIC_HIDDEN_DIM = 64 #256으로 변경하는것 고려
MAX_NODES = 30              # 그래프 최대 노드 수
EDGE_DIM = 1                # 엣지 특성 차원

# UniMP 하이퍼파라미터
NUM_HEADS = 4
NUM_LAYERS = 2
DROPOUT = 0.1

# PPO 하이퍼파라미터
CLIP_EPSILON = 0.2
VALUE_LOSS_COEF = 0.5
ENTROPY_COEF = 0.2  # 탐험 계수 대폭 증가 (0.05 → 0.2)
GAMMA = 0.99
GAE_LAMBDA = 0.95
MAX_GRAD_NORM = 0.5

# GPU 및 성능 설정
USE_GPU_PHYSICS = True
INCLUDE_OBSTACLES = True
DEVICE = torch.device('cuda')

# 자동 저장 설정 (에피소드 단위)
AUTO_SAVE_FREQUENCY = 500   # 500 에피소드마다 자동 저장
BEST_MODEL_SAVE_FREQUENCY = 100  # 100 에피소드마다 최고 성능 체크


class ParallelEnvironments:
    """병렬 환경 관리자"""

    def __init__(self, n_envs: int, env_config: Dict[str, Any]):
        self.n_envs = n_envs
        self.env_config = env_config
        self.envs = []

        # 각 환경을 다른 설정으로 초기화 (다양성 확보)
        for i in range(n_envs):
            # 환경별로 약간씩 다른 설정 (다양성 확보)
            env_seed = 42 + i
            np.random.seed(env_seed)

            # 병목 위치를 약간씩 다르게 (±10% 범위)
            base_pos = env_config['bottleneck_pos']
            pos_variation = np.random.uniform(-0.1, 0.1) * base_pos
            varied_bottleneck_pos = base_pos + pos_variation

            env = CleanBottleneckEnv(
                num_agents=env_config['num_agents'],
                corridor_width=env_config['corridor_width'],
                corridor_height=env_config['corridor_height'],
                bottleneck_width=env_config['bottleneck_width'],
                bottleneck_pos=varied_bottleneck_pos,  # 다양성 추가
                agent_radius=env_config['agent_radius'],
                sensing_radius=env_config['sensing_radius'],
                max_speed=env_config['max_speed'],
                max_timesteps=env_config['max_timesteps'],
                device=env_config['device'],
                use_gpu_physics=env_config['use_gpu_physics'],
                include_obstacles=env_config['include_obstacles'],
                verbose=False  # 병렬 환경에서는 로그 출력 억제
            )

            self.envs.append(env)

    def reset(self) -> List[Any]:
        """모든 환경 리셋"""
        return [env.reset() for env in self.envs]

    def step(self, actions_list: List[np.ndarray]) -> Tuple[List[Any], List[Any], List[bool], List[Dict]]:
        """모든 환경에서 스텝 실행"""
        obs_list = []
        rewards_list = []
        dones_list = []
        infos_list = []

        for env, actions in zip(self.envs, actions_list):
            obs, rewards, done, info = env.step(actions)
            obs_list.append(obs)
            rewards_list.append(rewards)
            dones_list.append(done)
            infos_list.append(info)

        return obs_list, rewards_list, dones_list, infos_list

    def get_observations(self) -> Tuple[List[List], List[Any], List[Any], List[Any], List[Any]]:
        """모든 환경에서 관측 수집"""
        all_local_obs = []
        all_node_obs = []
        all_adj = []
        all_entity_types = []
        all_edge_features = []

        for env in self.envs:
            # 각 환경의 local observations
            local_obs_list = []
            for i in range(env.num_agents):
                local_obs = env.get_local_observation(i)
                local_obs_list.append(local_obs)
            all_local_obs.append(local_obs_list)

            # 그래프 데이터
            node_obs_np, adj_np, entity_types_np, edge_features_np = build_unimp_graph_observations(
                env.agents, env.landmarks, env.obstacles, env.sensing_radius
            )
            all_node_obs.append(node_obs_np)
            all_adj.append(adj_np)
            all_entity_types.append(entity_types_np)
            all_edge_features.append(edge_features_np)

        return all_local_obs, all_node_obs, all_adj, all_entity_types, all_edge_features


def parse_args():
    """명령행 인수 파싱"""
    parser = argparse.ArgumentParser(description='InforMARL Bottleneck Training')
    parser.add_argument('--resume', type=str, default=None,
                       help='체크포인트 파일 경로 (학습 재개)')
    parser.add_argument('--test-only', action='store_true',
                       help='테스트만 실행 (학습 안함)')
    parser.add_argument('--checkpoint-dir', type=str, default='saved_models',
                       help='체크포인트 저장 디렉토리')
    parser.add_argument('--sampling', type=str, choices=['random', 'stepwise'], default='random',
                       help='샘플링 방식: random (원본) 또는 stepwise (개선)')
    parser.add_argument('--parallel', action='store_true', default=True,
                       help='병렬 환경 사용 (기본값: True)')
    parser.add_argument('--single-env', action='store_true',
                       help='단일 환경 사용 (디버깅용)')
    return parser.parse_args()


def main():
    """메인 실행 함수 - 스텝 단위 학습"""
    args = parse_args()

    # 명령행 인수로 설정 오버라이드
    global USE_PARALLEL_ENVS, SAMPLING_METHOD
    USE_PARALLEL_ENVS = args.parallel and not args.single_env
    SAMPLING_METHOD = args.sampling

    print("=== InforMARL Sampling Comparison 실험 ===")
    print(f"Device: {DEVICE}")
    print(f"에이전트 수: {NUM_AGENTS}")
    print(f"총 스텝: {TOTAL_TIMESTEPS:,}")
    print(f"에피소드 길이: {EPISODE_LENGTH}")
    print(f"병렬 환경: {'예' if USE_PARALLEL_ENVS else '아니오'}")
    print(f"샘플링 방식: {SAMPLING_METHOD}")
    print(f"미니배치 크기: {MINI_BATCH_SIZE}")
    print(f"PPO epochs: {PPO_EPOCHS}")
    if args.resume:
        print(f"체크포인트에서 재개: {args.resume}")
    if args.test_only:
        print("테스트 모드: 학습 없이 실행")
    print()
    
    # 환경 초기화
    if USE_PARALLEL_ENVS:
        print(f"병렬 환경 초기화: {N_ROLLOUT_THREADS}개 환경")
        env_config = {
            'num_agents': NUM_AGENTS,
            'corridor_width': CORRIDOR_WIDTH,
            'corridor_height': CORRIDOR_HEIGHT,
            'bottleneck_width': BOTTLENECK_WIDTH,
            'bottleneck_pos': BOTTLENECK_POS,
            'agent_radius': AGENT_RADIUS,
            'sensing_radius': SENSING_RADIUS,
            'max_speed': MAX_SPEED,
            'max_timesteps': MAX_TIMESTEPS,
            'device': DEVICE,
            'use_gpu_physics': USE_GPU_PHYSICS,
            'include_obstacles': INCLUDE_OBSTACLES
        }
        envs = ParallelEnvironments(N_ROLLOUT_THREADS, env_config)
        env = envs  # 호환성을 위해 동일한 변수명 사용
        print(f"샘플링 방식: {SAMPLING_METHOD}")
    else:
        print("단일 환경 초기화")
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

    # 모델 저장/불러오기 매니저 초기화
    model_saver = ModelSaver(save_dir=args.checkpoint_dir)

    # 현재 실행 중인 실제 설정으로 config 생성
    config = get_training_config(
        NUM_AGENTS=NUM_AGENTS,
        SENSING_RADIUS=SENSING_RADIUS,
        MAX_NODES=MAX_NODES,
        EDGE_DIM=EDGE_DIM,
        LOCAL_OBS_DIM=LOCAL_OBS_DIM,
        ACTION_DIM=ACTION_DIM,
        GNN_OUTPUT_DIM=GNN_OUTPUT_DIM,
        ACTOR_HIDDEN_DIM=ACTOR_HIDDEN_DIM,
        CRITIC_HIDDEN_DIM=CRITIC_HIDDEN_DIM,
        NUM_HEADS=NUM_HEADS,
        NUM_LAYERS=NUM_LAYERS,
        DROPOUT=DROPOUT,
        LEARNING_RATE=LEARNING_RATE,
        CLIP_EPSILON=CLIP_EPSILON,
        VALUE_LOSS_COEF=VALUE_LOSS_COEF,
        ENTROPY_COEF=ENTROPY_COEF,
        GAMMA=GAMMA,
        GAE_LAMBDA=GAE_LAMBDA,
        MAX_GRAD_NORM=MAX_GRAD_NORM,
        MINI_BATCH_SIZE=MINI_BATCH_SIZE,
        PPO_EPOCHS=PPO_EPOCHS,
        CORRIDOR_WIDTH=CORRIDOR_WIDTH,
        CORRIDOR_HEIGHT=CORRIDOR_HEIGHT,
        BOTTLENECK_WIDTH=BOTTLENECK_WIDTH,
        BOTTLENECK_POS=BOTTLENECK_POS,
        AGENT_RADIUS=AGENT_RADIUS,
        MAX_SPEED=MAX_SPEED,
        MAX_TIMESTEPS=MAX_TIMESTEPS,
        USE_GPU_PHYSICS=USE_GPU_PHYSICS,
        INCLUDE_OBSTACLES=INCLUDE_OBSTACLES
    )
    
    # 롤아웃 버퍼 초기화
    if USE_PARALLEL_ENVS:
        # 병렬 환경: 원본 InforMARL과 동일한 크기
        buffer_size = EPISODE_LENGTH * N_ROLLOUT_THREADS
        effective_num_agents = NUM_AGENTS  # 각 환경당 에이전트 수
    else:
        # 단일 환경
        buffer_size = EPISODE_LENGTH
        effective_num_agents = NUM_AGENTS

    buffer = RolloutBuffer(
        buffer_size=buffer_size,
        num_agents=effective_num_agents,
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
    best_success_rate = 0.0

    # 체크포인트에서 재개
    if args.resume:
        try:
            checkpoint_info = model_saver.load_checkpoint(args.resume, models, optimizer, DEVICE)
            global_step = checkpoint_info['global_step']
            episode_count = checkpoint_info['episode_count']
            best_success_rate = checkpoint_info['success_rate']
            print(f"체크포인트에서 재개됨: step {global_step:,}")
        except Exception as e:
            print(f"체크포인트 로드 실패: {e}")
            print("새로운 훈련으로 시작합니다.")

    # 테스트 모드라면 바로 데모 실행
    if args.test_only:
        if not args.resume:
            # 가장 좋은 체크포인트 자동 로드
            best_checkpoint = model_saver.find_best_checkpoint()
            if best_checkpoint:
                print(f"최고 성능 모델 로드: {best_checkpoint['filename']}")
                model_saver.load_checkpoint(best_checkpoint['filepath'], models, device=DEVICE)
            else:
                print("저장된 모델이 없습니다. 랜덤 초기화 모델로 테스트합니다.")

        run_stepwise_demo(env, models)
        return
    
    # 환경 리셋
    env.reset()
    episode_reward = 0
    episode_length = 0
    
    print("에피소드 기반 훈련 시작...")
    
    # 에피소드 기반 훈련 시작
    total_episodes = TOTAL_TIMESTEPS // EPISODE_LENGTH
    episode_count = 0

    while episode_count < total_episodes:
        # 에피소드 시작
        env.reset()
        episode_reward = 0
        episode_length = 0

        # 한 에피소드 동안 데이터 수집
        for step in range(EPISODE_LENGTH):
            if USE_PARALLEL_ENVS:
                # 병렬 환경: 모든 환경에서 동시에 데이터 수집
                all_local_obs, all_node_obs, all_adj, all_entity_types, all_edge_features = env.get_observations()

                # 모든 환경의 데이터를 배치로 합치기
                batch_local_obs = []
                batch_node_obs = []
                batch_adj = []
                batch_entity_types = []
                batch_edge_features = []

                for env_idx in range(N_ROLLOUT_THREADS):
                    for agent_idx in range(NUM_AGENTS):
                        batch_local_obs.append(all_local_obs[env_idx][agent_idx])
                        batch_node_obs.append(all_node_obs[env_idx][agent_idx])
                        batch_adj.append(all_adj[env_idx][agent_idx])
                        batch_entity_types.append(all_entity_types[env_idx][agent_idx])
                        batch_edge_features.append(all_edge_features[env_idx][agent_idx])

                # 배치 텐서 변환
                local_obs_batch = torch.tensor(batch_local_obs, dtype=torch.float32).to(DEVICE)
                node_obs_batch = torch.tensor(batch_node_obs, dtype=torch.float32).to(DEVICE)
                adj_batch = torch.tensor(batch_adj, dtype=torch.float32).to(DEVICE)
                entity_types_batch = torch.tensor(batch_entity_types, dtype=torch.long).to(DEVICE)
                edge_features_batch = torch.tensor(batch_edge_features, dtype=torch.float32).to(DEVICE)

                # 정책 실행 (전체 배치)
                # 1. GNN 특성 추출
                gnn_features_batch = models.gnn(
                    node_obs_batch, adj_batch, entity_types_batch, edge_features_batch,
                    list(range(len(local_obs_batch)))  # agent_ids
                )

                # 2. Actor로 행동 선택
                combined_input = torch.cat([local_obs_batch, gnn_features_batch], dim=-1)
                action_means, action_stds = models.actor(combined_input)
                action_dist = torch.distributions.Normal(action_means, action_stds)
                actions_batch = action_dist.sample()
                actions_batch = torch.clamp(actions_batch, -1.0, 1.0)
                log_probs_batch = action_dist.log_prob(actions_batch).sum(dim=-1)

                # 3. Critic으로 가치 추정 (GNN 평균 사용)
                avg_gnn_features = gnn_features_batch.view(N_ROLLOUT_THREADS, NUM_AGENTS, -1).mean(dim=1)
                values_per_env = models.critic(avg_gnn_features).squeeze(-1)  # [N_ROLLOUT_THREADS]
                # 각 환경의 모든 에이전트에게 동일한 value 할당
                values_batch = values_per_env.unsqueeze(1).repeat(1, NUM_AGENTS).view(-1)  # [N_ROLLOUT_THREADS * NUM_AGENTS]

                # 행동을 환경별로 분할
                actions_by_env = []
                log_probs_by_env = []
                values_by_env = []

                for env_idx in range(N_ROLLOUT_THREADS):
                    start_idx = env_idx * NUM_AGENTS
                    end_idx = start_idx + NUM_AGENTS

                    env_actions = actions_batch[start_idx:end_idx].cpu().numpy().tolist()
                    env_log_probs = log_probs_batch[start_idx:end_idx].cpu().tolist()
                    env_values = values_batch[start_idx:end_idx].cpu().tolist()

                    actions_by_env.append(env_actions)
                    log_probs_by_env.append(env_log_probs)
                    values_by_env.append(env_values)

                # 모든 환경에서 스텝 실행
                _, rewards_list, dones_list, infos_list = env.step(actions_by_env)

                # 버퍼에 데이터 저장 (모든 환경)
                for env_idx in range(N_ROLLOUT_THREADS):
                    raw_rewards = rewards_list[env_idx]
                    rewards = [(r + 18.0) / 28.0 for r in raw_rewards]

                    buffer.store(
                        all_local_obs[env_idx], actions_by_env[env_idx], rewards,
                        values_by_env[env_idx], log_probs_by_env[env_idx],
                        all_node_obs[env_idx], all_adj[env_idx],
                        all_entity_types[env_idx], all_edge_features[env_idx]
                    )

                # 진행 상황 업데이트 (실제 스텝 기준)
                actual_step = step + 1  # 현재 에피소드 내 실제 스텝
                total_env_steps = episode_count * EPISODE_LENGTH + actual_step  # 총 환경 스텝
                global_step += N_ROLLOUT_THREADS  # 데이터 포인트 수 (기존 호환성)
                episode_reward = sum([sum(rewards) for rewards in rewards_list]) / N_ROLLOUT_THREADS

                # 에피소드 완료 체크 (병렬 환경)
                completed_envs = sum(dones_list)
                if completed_envs > 0:
                    # 성공률 계산 (완료된 환경들의 평균)
                    total_success_rate = sum([infos_list[i]['success_rate'] for i in range(N_ROLLOUT_THREADS) if dones_list[i]])
                    avg_success_rate = total_success_rate / completed_envs if completed_envs > 0 else 0
                    episode_count += completed_envs
                    print(f"Episode {episode_count}: {completed_envs} environments completed, "
                          f"Avg Reward = {episode_reward:.2f}, Success Rate = {avg_success_rate:.1f}%")

                    # 병렬 환경 모델 저장 체크
                    if episode_count % AUTO_SAVE_FREQUENCY == 0:
                        model_saver.save_checkpoint(models, optimizer, episode_count, episode_count,
                                                   episode_reward, avg_success_rate, config, f"episode_{episode_count}")
                        print(f"  ✓ 자동 저장 완료 (Episode {episode_count})")

                    if episode_count % BEST_MODEL_SAVE_FREQUENCY == 0:
                        if avg_success_rate > best_success_rate:
                            best_success_rate = avg_success_rate
                            model_saver.save_best_model(models, optimizer, episode_count, episode_reward, avg_success_rate, config)
                            print(f"  ★ 새로운 최고 성능! 모델 저장됨 (성공률: {avg_success_rate:.1f}%)")

            else:
                # 단일 환경: 기존 방식
                local_obs_list = []
                for i in range(NUM_AGENTS):
                    local_obs = env.get_local_observation(i)
                    local_obs_list.append(local_obs)

                # UniMP 그래프 데이터 생성
                node_obs_np, adj_np, entity_types_np, edge_features_np = build_unimp_graph_observations(
                    env.agents, env.landmarks, env.obstacles, SENSING_RADIUS
                )

                # 배치 형태로 변환
                local_obs_batch = torch.tensor(local_obs_list, dtype=torch.float32).to(DEVICE)

                # 텐서 변환
                node_obs_batch = torch.tensor(node_obs_np, dtype=torch.float32).to(DEVICE)
                adj_batch = torch.tensor(adj_np, dtype=torch.float32).to(DEVICE)
                entity_types_batch = torch.tensor(entity_types_np, dtype=torch.long).to(DEVICE)
                edge_features_batch = torch.tensor(edge_features_np, dtype=torch.float32).to(DEVICE)
            
            if not USE_PARALLEL_ENVS:
                # 단일 환경에서만 실행
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

                # 디버깅: 처음 몇 스텝에서 행동 출력 확인
                if global_step < 50 and step == 0:
                    print(f"Step {global_step}:")
                    print(f"  action_means: {action_means.cpu().numpy()}")
                    print(f"  action_stds: {action_stds.cpu().numpy()}")

                action_dist = torch.distributions.Normal(action_means, action_stds)
                actions_tensor = action_dist.sample()
                actions_tensor = torch.clamp(actions_tensor, -1.0, 1.0)
                log_probs_tensor = action_dist.log_prob(actions_tensor).sum(dim=-1)

                # 디버깅: 실제 행동 출력 확인
                if global_step < 50 and step == 0:
                    print(f"  sampled_actions: {actions_tensor.cpu().numpy()}")

                # 3. Critic: 모든 에이전트 GNN 특성의 평균으로 가치 추정
                avg_gnn_features = torch.stack(all_agent_gnn_features).mean(dim=0)  # [1, gnn_output_dim]
                values_tensor = models.critic(avg_gnn_features)  # GNN 평균만 사용!

                actions = actions_tensor.cpu().numpy().tolist()
                log_probs = log_probs_tensor.cpu().tolist()
                values = [values_tensor.cpu().item()] * NUM_AGENTS

            # 환경 스텝은 이미 위에서 처리됨 (병렬 환경의 경우)
            if not USE_PARALLEL_ENVS:
                # 단일 환경: 기존 방식
                _, raw_rewards, done, info = env.step(actions)

                # 보상 정규화: [-18, 10] → [0, 1] (수정된 범위 기준)
                rewards = [(r + 18.0) / 28.0 for r in raw_rewards]

                # 디버깅: 보상값 확인 (처음 몇 스텝만)
                if global_step < 50 and step == 0:
                    print(f"  raw_rewards: {raw_rewards}")
                    print(f"  normalized_rewards: {rewards}")
                    print(f"  success_count: {info['success_count']}")

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

            # 자동 저장 체크 (에피소드 단위)
            if episode_count % AUTO_SAVE_FREQUENCY == 0 and episode_count > 0:
                if len(episode_rewards) > 0:
                    recent_episodes = min(100, len(episode_rewards))
                    avg_reward = np.mean(episode_rewards[-recent_episodes:])
                    avg_success = np.mean(episode_success_rates[-recent_episodes:])
                    model_saver.save_checkpoint(models, optimizer, global_step, episode_count,
                                               avg_reward, avg_success, config, f"auto_{global_step//1000}k")
                else:
                    model_saver.save_checkpoint(models, optimizer, global_step, episode_count,
                                               0.0, 0.0, config, f"auto_{global_step//1000}k")

            # 평가 및 출력 체크 (매 스텝마다) - 단일 환경에서만
            if not USE_PARALLEL_ENVS and global_step % EVAL_FREQUENCY == 0:
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

            # 병렬 환경 전용 로깅 제거 (에피소드 완료 시에만 출력)

            # 최고 성능 모델 저장 체크 (에피소드 단위)
            if episode_count % BEST_MODEL_SAVE_FREQUENCY == 0 and len(episode_rewards) > 0:
                recent_episodes = min(100, len(episode_rewards))
                avg_reward = np.mean(episode_rewards[-recent_episodes:])
                avg_success = np.mean(episode_success_rates[-recent_episodes:])
                if avg_success > best_success_rate:
                    best_success_rate = avg_success
                    model_saver.save_best_model(models, optimizer, global_step, avg_reward, avg_success, config)
                    print(f"  ★ 새로운 최고 성능! 모델 저장됨 (성공률: {avg_success:.1f}%)")

            # 조기 종료 확인 (단일 환경에서만)
            if not USE_PARALLEL_ENVS and done:
                break

        # 에피소드 종료 처리 (단일 환경에서만)
        if not USE_PARALLEL_ENVS:
            # 마지막 가치 계산 (bootstrap value)
            if not done:  # 에피소드가 끝나지 않고 최대 길이에 도달한 경우
                # 마지막 상태의 GNN 특성들 계산 (단일 환경)
                node_obs_np_last, adj_np_last, entity_types_np_last, edge_features_np_last = build_unimp_graph_observations(
                    env.agents, env.landmarks, env.obstacles, SENSING_RADIUS
                )
            else:
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
            last_value = 0.0  # 에피소드가 끝난 경우 다음 가치는 0

        # 에피소드 통계 저장 (단일 환경에서만)
        if not USE_PARALLEL_ENVS:
            # 경로 종료 및 GAE 계산
            buffer.finish_path(last_value)

            # 에피소드 통계 저장
            episode_rewards.append(episode_reward)
            episode_success_rates.append(info['success_rate'])
            episode_lengths.append(step + 1)  # 실제 에피소드 길이

            # 에피소드 완료 후 업데이트
            episode_count += 1

        # PPO 업데이트 (매 에피소드마다)
        if buffer.has_data():
            update_policy_stepwise(models, optimizer, buffer)
            buffer.reset()

        # 에피소드 기반 평가 및 로깅
        if episode_count % EVAL_FREQUENCY == 0:
            recent_episodes = min(10, len(episode_rewards))
            avg_reward = np.mean(episode_rewards[-recent_episodes:])
            avg_success = np.mean(episode_success_rates[-recent_episodes:])
            avg_length = np.mean(episode_lengths[-recent_episodes:])

            print(f"Episode {episode_count}/{total_episodes}: "
                  f"Avg Reward = {avg_reward:.2f}, "
                  f"Success Rate = {avg_success:.1f}%, "
                  f"Avg Length = {avg_length:.1f} [Episode-based]")

        # 최고 성능 모델 저장
        if episode_count % (EVAL_FREQUENCY * 5) == 0 and len(episode_rewards) > 0:
            recent_episodes = min(100, len(episode_rewards))
            avg_reward = np.mean(episode_rewards[-recent_episodes:])
            avg_success = np.mean(episode_success_rates[-recent_episodes:])
            if avg_success > best_success_rate:
                best_success_rate = avg_success
                model_saver.save_best_model(models, optimizer, episode_count, avg_reward, avg_success, config)
                print(f"  ★ 새로운 최고 성능! 모델 저장됨 (성공률: {avg_success:.1f}%)")

        # 자동 저장
        if episode_count % (EVAL_FREQUENCY * 10) == 0 and episode_count > 0:
            if len(episode_rewards) > 0:
                recent_episodes = min(100, len(episode_rewards))
                avg_reward = np.mean(episode_rewards[-recent_episodes:])
                avg_success = np.mean(episode_success_rates[-recent_episodes:])
                model_saver.save_checkpoint(models, optimizer, episode_count * EPISODE_LENGTH, episode_count,
                                           avg_reward, avg_success, config, f"episode_{episode_count}")


    
    print(f"\n=== 에피소드 기반 훈련 완료 ===")
    print(f"총 에피소드: {episode_count}")
    if len(episode_rewards) > 0:
        final_avg_reward = np.mean(episode_rewards[-10:])
        final_success_rate = np.mean(episode_success_rates[-10:])
        print(f"최종 평균 보상: {final_avg_reward:.2f}")
        print(f"최종 성공률: {final_success_rate:.1f}%")

        # 최종 모델 저장
        model_saver.save_checkpoint(models, optimizer, episode_count * EPISODE_LENGTH, episode_count,
                                   final_avg_reward, final_success_rate, config, "final")
        print("최종 모델이 저장되었습니다.")

    # 저장된 체크포인트 목록 출력
    print("\n=== 저장된 체크포인트 ===")
    checkpoints = model_saver.list_checkpoints()
    if checkpoints:
        for cp in checkpoints[-5:]:  # 최근 5개만 표시
            print(f"  {cp['filename']}: step {cp['global_step']:,}, success {cp['success_rate']:.1f}%")
    else:
        print("  저장된 체크포인트가 없습니다.")

    # 데모 실행
    if input("\n스텝 단위 학습 데모를 실행할까요? (y/n): ").lower() == 'y':
        run_stepwise_demo(env, models)


def update_policy_stepwise(models, optimizer, buffer):
    """PPO 업데이트 - 샘플링 방식 선택 가능"""
    data = buffer.get()

    batch_size = data['local_obs'].shape[0]
    num_mini_batch = batch_size // MINI_BATCH_SIZE

    # PPO 여러 epoch 업데이트
    for _ in range(PPO_EPOCHS):
        if SAMPLING_METHOD == "random":
            # 완전 랜덤 샘플링 (원본 InforMARL 방식)
            rand = torch.randperm(batch_size)
            sampler = [
                rand[i * MINI_BATCH_SIZE : (i + 1) * MINI_BATCH_SIZE]
                for i in range(num_mini_batch)
            ]
        elif SAMPLING_METHOD == "stepwise":
            # 스텝 단위 샘플링 (개선된 방식)
            if USE_PARALLEL_ENVS:
                # 병렬 환경: 각 스텝당 (n_envs * num_agents) 샘플
                step_size = N_ROLLOUT_THREADS * NUM_AGENTS
                total_steps = batch_size // step_size
                steps_per_batch = MINI_BATCH_SIZE // step_size

                # 스텝들을 랜덤하게 섞기
                shuffled_steps = torch.randperm(total_steps)
                sampler = []

                for i in range(num_mini_batch):
                    selected_steps = shuffled_steps[i * steps_per_batch : (i + 1) * steps_per_batch]
                    indices = []
                    for step in selected_steps:
                        base_idx = step * step_size
                        for j in range(step_size):
                            indices.append(base_idx + j)
                    sampler.append(torch.tensor(indices, dtype=torch.long))
            else:
                # 단일 환경: 기존 StepWiseSampler 사용
                from src.utils.rollout_buffer import StepWiseSampler
                total_steps = batch_size // NUM_AGENTS
                step_sampler = StepWiseSampler(total_steps, NUM_AGENTS, MINI_BATCH_SIZE)
                sampler = list(step_sampler)
        else:
            raise ValueError(f"Unknown sampling method: {SAMPLING_METHOD}")

        for indices in sampler:
            # 미니배치 데이터 추출
            mini_batch = {key: value[indices] for key, value in data.items()}

            # 1. GNN forward pass - 각 샘플별로 처리 (에이전트 구분 없음)
            batch_node_obs = mini_batch['node_obs']
            batch_adj = mini_batch['adj']
            batch_entity_types = mini_batch['entity_types']
            batch_edge_features = mini_batch['edge_features']

            # 미니배치에서 실제 agent_ids 추출
            batch_agent_ids = mini_batch['agent_ids']

            # 배치로 GNN 처리 (에이전트 ID 정보 유지)
            gnn_features = models.gnn(
                batch_node_obs, batch_adj, batch_entity_types,
                batch_edge_features, batch_agent_ids
            )

            # 2. Actor forward pass
            combined_input = torch.cat([mini_batch['local_obs'], gnn_features], dim=-1)
            action_means, action_stds = models.actor(combined_input)

            action_dist = torch.distributions.Normal(action_means, action_stds)
            new_log_probs = action_dist.log_prob(mini_batch['actions']).sum(dim=-1)
            entropy = action_dist.entropy().sum(dim=-1)

            # 3. Critic forward pass - 기존 저장된 값 사용 (재계산 없음)
            # InforMARL에서는 Critic 값이 이미 수집 시점에서 계산되어 저장됨
            new_values = mini_batch['old_values']  # 저장된 값 사용

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

            # 디버깅: 손실값 출력 (처음 몇 번만)
            if hasattr(update_policy_stepwise, 'update_count'):
                update_policy_stepwise.update_count += 1
            else:
                update_policy_stepwise.update_count = 1

            if update_policy_stepwise.update_count <= 5:
                print(f"Update {update_policy_stepwise.update_count}:")
                print(f"  policy_loss: {policy_loss.item():.6f}")
                print(f"  value_loss: {value_loss.item():.6f}")
                print(f"  entropy_loss: {entropy_loss.item():.6f}")
                print(f"  total_loss: {total_loss.item():.6f}")

            # 역전파
            optimizer.zero_grad()
            total_loss.backward()

            # 그래디언트 노름 확인
            grad_norm = torch.nn.utils.clip_grad_norm_(
                models.get_all_parameters(),
                MAX_GRAD_NORM
            )

            if update_policy_stepwise.update_count <= 5:
                print(f"  grad_norm: {grad_norm:.6f}")

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
            if USE_PARALLEL_ENVS:
                demo_env = env.envs[0]
                node_obs_np, adj_np, entity_types_np, edge_features_np = build_unimp_graph_observations(
                    demo_env.agents, demo_env.landmarks, demo_env.obstacles, SENSING_RADIUS
                )
            else:
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
            
            # 환경 스텝 (데모는 항상 단일 환경 또는 첫 번째 환경 사용)
            if USE_PARALLEL_ENVS:
                # 병렬 환경에서는 첫 번째 환경만 사용
                _, rewards, done, info = env.envs[0].step(actions)
            else:
                _, rewards, done, info = env.step(actions)
            episode_reward += sum(rewards)
            
            # 렌더링 (데모는 항상 단일 환경 또는 첫 번째 환경 사용)
            if USE_PARALLEL_ENVS:
                env.envs[0].render()
            else:
                env.render()
            time.sleep(0.05)
            
            if done:
                success_rate = info.get('success_rate', 0) if 'info' in locals() else 0
                print(f"  완료! 스텝: {step + 1}, 보상: {episode_reward:.2f}, "
                      f"성공률: {success_rate:.1f}% [Step-wise]")
                break
        
        time.sleep(1.0)
    
    env.close()


if __name__ == "__main__":
    main()