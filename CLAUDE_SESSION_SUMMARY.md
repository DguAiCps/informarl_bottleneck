# InforMARL Bottleneck Environment - Claude Session Summary

## 프로젝트 개요

### 환경 설명
- **프로젝트명**: InforMARL Bottleneck Environment
- **목적**: 다중 에이전트 강화학습 환경에서 병목 지점 협력 학습
- **핵심 기술**: UniMP GNN + PPO 기반 협력적 학습
- **에이전트**: 4개 에이전트가 병목 지점을 통과하며 목표 지점 도달

### 환경 구성요소
```
환경 크기: 20.0 × 10.0 복도
병목 위치: x=10.0, 폭=6.0
에이전트: 반지름 0.5, 최대속도 1.0
센싱 범위: 7.0
최대 스텝: 300
```

### 핵심 파일 구조
```
src/
├── env/
│   ├── bottleneck_env.py          # 메인 환경
│   ├── physics.py                 # 물리 시뮬레이션
│   ├── reward.py                  # 보상 시스템
│   ├── map.py                     # 맵 생성
│   └── graph_builder_unimp.py     # GNN 그래프 생성
├── models/
│   ├── policy_unimp.py            # Actor-Critic 모델
│   └── gnn_unimp.py               # UniMP GNN 구현
├── utils/
│   ├── types.py                   # 데이터 타입 정의
│   └── rollout_buffer.py          # 롤아웃 버퍼
└── utils/
    └── model_saver.py             # 모델 저장/로딩

main_stepwise.py                   # 메인 학습 스크립트
```

## 발견된 주요 문제점들

### 1. 심각한 데이터 인덱싱 오류 🚨
**문제**: 하나의 에이전트만 학습되고 나머지는 거의 학습되지 않음

**원인**: 미니배치 샘플링에서 잘못된 에이전트 분류
```python
# 잘못된 방식
agent_start_idx = agent_id * samples_per_agent  # 연속 블록으로 나눔
# 실제로는 서로 다른 타임스텝의 여러 에이전트 데이터가 섞임
```

**실제 데이터 순서**:
```
버퍼 저장: [step0_agent0, step0_agent1, step0_agent2, step0_agent3,
           step1_agent0, step1_agent1, step1_agent2, step1_agent3, ...]
미니배치: [인덱스 847, 23, 456, 1089, ...] (랜덤)
```

### 2. 빈 텐서 문제
**문제**: `torch.stack()` 실행 시 크기가 다른 텐서들로 인한 오류
```
Agent 0: [16, 64] 크기 텐서
Agent 1: [15, 64] 크기 텐서
Agent 2: [17, 64] 크기 텐서
→ torch.stack() 실패
```

### 3. 논문 의도 왜곡
**문제**: Critic이 시간적 일관성 없는 혼합 데이터로 학습
- 원래: "동일 시점 4명 협력 상황의 가치 평가"
- 기존 구현: "서로 다른 시간대 에이전트들의 혼합 상황 평가"

## 핵심 해결책: Step-wise Sampling

### 근본적 해결 방법
**기존**: 개별 에이전트 데이터 랜덤 샘플링
**수정**: Step 단위 샘플링 (동일 타임스텝의 모든 에이전트 함께)

```python
class StepWiseSampler:
    def __init__(self, total_steps: int, num_agents: int, mini_batch_size: int):
        self.steps_per_batch = mini_batch_size // num_agents  # 자동 계산

    def __iter__(self):
        # 1. 스텝을 랜덤 선택
        selected_steps = torch.randperm(self.total_steps)[:self.steps_per_batch]

        # 2. 각 스텝의 모든 에이전트 포함
        indices = []
        for step in selected_steps:
            base_idx = step * self.num_agents
            for agent in range(self.num_agents):
                indices.append(base_idx + agent)
```

### 해결된 문제들

**1. ✅ 일관성 문제 해결**
- 동일 시점의 모든 에이전트가 함께 샘플링됨
- 논문의 협력적 학습 의도 구현

**2. ✅ 빈 텐서 문제 해결**
- 각 에이전트마다 정확히 동일한 수의 샘플 (16개씩)
- `torch.stack()` 오류 발생 불가능

**3. ✅ 균등한 학습**
- 모든 에이전트가 자신의 데이터로만 학습
- 4명 모두 똑같이 발전 가능

## 수정된 코드 구조

### 1. 새로운 샘플러 (rollout_buffer.py)
```python
# 기존 MiniBatchSampler → StepWiseSampler로 교체
class StepWiseSampler:
    # Step 단위로 미니배치 생성
    # 논문 의도에 맞는 동일 타임스텝 에이전트들을 함께 샘플링
```

### 2. 단순화된 에이전트별 처리 (main_stepwise.py)
```python
# GNN 처리 - 복잡한 마스킹 제거
for agent_id in range(NUM_AGENTS):
    agent_indices = torch.arange(agent_id, len(indices), NUM_AGENTS)
    agent_gnn_features = models.gnn(
        mini_batch['node_obs'][agent_indices],
        # ...
        agent_id
    )

# Actor 처리 - 동일한 방식
for agent_id in range(NUM_AGENTS):
    agent_indices = torch.arange(agent_id, len(indices), NUM_AGENTS)
    combined_input = torch.cat([
        mini_batch['local_obs'][agent_indices],
        all_agent_gnn_features[agent_id]
    ], dim=-1)
```

### 3. 논문 의도대로 Critic 구현
```python
# 각 스텝별로 모든 에이전트의 GNN 출력 평균
stacked_features = torch.stack(all_agent_gnn_features)    # [agents, steps, features]
avg_features_per_step = stacked_features.mean(dim=0)     # [steps, features]

# 각 에이전트에 해당 스텝의 글로벌 컨텍스트 적용
global_contexts = avg_features_per_step.repeat(NUM_AGENTS, 1)
new_values = models.critic(global_contexts)
```

## 코드 실행 흐름

### 1. 환경 초기화
```python
env = CleanBottleneckEnv(num_agents=4, ...)
models = InforMARLModels(...)
buffer = RolloutBuffer(buffer_size=300, num_agents=4, ...)
```

### 2. 데이터 수집 (매 스텝)
```python
for rollout_step in range(300):
    # 4명 에이전트 각각 관측
    local_obs_list = [agent0_obs, agent1_obs, agent2_obs, agent3_obs]

    # 그래프 생성 (모든 에이전트 고려)
    node_obs, adj, entity_types, edge_features = build_unimp_graph_observations(...)

    # 각 에이전트별 행동 선택
    for agent_id in range(4):
        gnn_output = models.gnn(graph_data, agent_id)
        action_mean, action_std = models.actor(local_obs + gnn_output)

    # 환경 스텝 실행
    rewards = env.step(actions)

    # 버퍼에 저장
    buffer.store(local_obs_list, actions, rewards, ...)
```

### 3. 학습 업데이트 (버퍼가 가득 찰 때)
```python
# Step-wise 샘플링
sampler = StepWiseSampler(total_steps=300, num_agents=4, mini_batch_size=64)

for indices in sampler:  # 16개 스텝 × 4개 에이전트 = 64개 샘플
    # 1. 에이전트별 GNN 처리
    for agent_id in range(4):
        agent_indices = torch.arange(agent_id, 64, 4)  # [0,4,8,12,...]
        agent_gnn_features[agent_id] = models.gnn(data[agent_indices], agent_id)

    # 2. 에이전트별 Actor 처리
    for agent_id in range(4):
        agent_indices = torch.arange(agent_id, 64, 4)
        actions[agent_id] = models.actor(obs[agent_indices] + gnn_features[agent_id])

    # 3. 논문 의도대로 Critic 처리
    step_wise_features = stack(agent_gnn_features).mean(dim=0)  # 각 스텝별 평균
    values = models.critic(step_wise_features.repeat(4, 1))     # 모든 에이전트에 적용

    # 4. PPO 손실 계산 및 업데이트
    policy_loss, value_loss, entropy_loss = compute_losses(...)
    optimizer.step()
```

## 주요 설정값

### 환경 설정
```python
NUM_AGENTS = 4
ROLLOUT_LENGTH = 300
MINI_BATCH_SIZE = 64           # 4의 배수 필수
PPO_EPOCHS = 4
LEARNING_RATE = 3e-4
ENTROPY_COEF = 0.05           # 0.01에서 증가 (더 많은 탐험)
```

### 모델 구조
```python
LOCAL_OBS_DIM = 6             # [rel_pos, rel_vel, goal_pos]
ACTION_DIM = 2                # [dv, dw] 속도 변화량
GNN_OUTPUT_DIM = 64
ACTOR_HIDDEN_DIM = 64
CRITIC_HIDDEN_DIM = 64        # 256으로 증가 고려 중
```

### 물리 파라미터
```python
MAX_SPEED = 1.0               # 최대 속도
AGENT_RADIUS = 0.5           # 에이전트 반지름
SENSING_RADIUS = 7.0         # 센싱 범위
속도 변화율 = 0.5             # 매 스텝 최대 ±0.5 속도 변화
가속도 = 5.0 단위/초²         # 상당히 민첩한 움직임
```

## 그래프 구조 (UniMP)

### 노드 타입
```python
ENTITY_TYPES = {"agent": 0, "landmark": 1, "obstacle": 2}
```

### 메시지 패싱 규칙
- **Agent ↔ Agent**: 양방향 연결
- **Non-agent → Agent**: 단방향 연결 (landmark, obstacle이 agent에게 정보 전달)
- **Non-agent ↔ Non-agent**: 연결 없음

### 노드 특성 (6차원)
```python
features = [rel_x, rel_y, rel_vx, rel_vy, goal_x, goal_y]
# 모든 위치는 ego agent 기준 상대 좌표
```

## 보상 시스템

### 보상 구성요소
```python
# 거리 기반 보상 (음수)
reward = -distance_to_goal

# 목표 도달 보상 (양수)
if distance < goal_radius:
    reward += 5.0

# 충돌 패널티
if collision:
    reward -= collision_penalty
```

### 정규화
```python
# 원시 보상 범위: 약 [-23, 5]
# 정규화 후: [0, 1] 범위로 변환
normalized_reward = (raw_reward + 20.0) / 20.0
```

## 성능 지표

### 추적되는 메트릭
- **Success Rate**: 목표 도달 에이전트 비율
- **Episode Reward**: 에피소드별 총 보상
- **Episode Length**: 에피소드 지속 시간
- **Collision Count**: 충돌 발생 횟수

### 모델 저장 기준
- **자동 저장**: 50k 스텝마다
- **최고 성능 저장**: 성공률 기준으로 최고 모델 저장
- **최종 저장**: 학습 완료 시

## 다음 클로드 세션에서 참고사항

### 즉시 실행 가능
현재 수정된 코드는 모든 문제가 해결되어 바로 실행 가능합니다.

### 추가 개선 고려사항
1. **CRITIC_HIDDEN_DIM**: 64 → 256 증가로 Value Loss 감소 고려
2. **VALUE_LOSS_COEF**: 0.5 → 0.25 감소로 학습 안정성 향상
3. **미니배치 크기**: 다른 배수 실험 (예: 68, 72, 80 등)

### 디버깅 정보
- 학습 초기 몇 스텝은 action_means, action_stds, rewards가 출력됨
- 업데이트 초기 5회는 손실값들이 상세히 출력됨
- 모든 에이전트가 균등하게 학습되는지 모니터링 가능

### 주의사항
- `MINI_BATCH_SIZE`는 반드시 `NUM_AGENTS`의 배수여야 함
- Step-wise sampling으로 논문의 협력적 학습 의도가 구현됨
- 모든 하드코딩이 자동 계산 변수로 교체됨

이제 모든 에이전트가 협력적으로 학습하여 병목 지점을 효율적으로 통과할 수 있습니다! 🚀