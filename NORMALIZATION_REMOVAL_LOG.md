# 정규화 제거 로그

## 개요
InforMARL 프로젝트에서 모든 좌표/속도 정규화를 제거하여 절대값 사용으로 변경

## 수정된 파일과 내용

### 1. `src/env/bottleneck_env.py`
**위치**: `get_local_observation()` 함수 (147-153줄)

**변경 전**:
```python
obs = [
    agent.x / self.corridor_width,      # 정규화된 위치
    agent.y / self.corridor_height,
    agent.vx / agent.max_speed,         # 정규화된 속도
    agent.vy / agent.max_speed,
    (target.x - agent.x) / self.corridor_width,  # 정규화된 목표 상대 위치
    (target.y - agent.y) / self.corridor_height
]
```

**변경 후**:
```python
obs = [
    agent.x,                            # 절대 위치
    agent.y,
    agent.vx,                           # 절대 속도
    agent.vy,
    (target.x - agent.x),               # 목표 상대 위치
    (target.y - agent.y)
]
```

### 2. `src/env/graph_builder_unimp.py`
**위치**: `build_unimp_graph_observations()` 함수

#### 2-1. 에이전트 상대 위치/속도 (42-50줄)
**변경 전**:
```python
# 상대적 위치/속도 (정규화 적용 - 성능 개선을 위해)
rel_x = (agent.x - ego_agent.x) / sensing_radius
rel_y = (agent.y - ego_agent.y) / sensing_radius
rel_vx = (agent.vx - ego_agent.vx) / agent.max_speed  # 상대속도 + 정규화
rel_vy = (agent.vy - ego_agent.vy) / agent.max_speed

# 해당 에이전트의 목표 위치 (ego 에이전트 기준) - 논문 정의
target = landmarks[agent.target_id] 
goal_x = (target.x - ego_agent.x) / sensing_radius  # p^{goal,j}_i
goal_y = (target.y - ego_agent.y) / sensing_radius
```

**변경 후**:
```python
# 상대적 위치/속도 (절대값 사용)
rel_x = agent.x - ego_agent.x
rel_y = agent.y - ego_agent.y
rel_vx = agent.vx - ego_agent.vx                    # 상대속도
rel_vy = agent.vy - ego_agent.vy

# 해당 에이전트의 목표 위치 (ego 에이전트 기준)
target = landmarks[agent.target_id] 
goal_x = target.x - ego_agent.x
goal_y = target.y - ego_agent.y
```

#### 2-2. 랜드마크 위치 (59-60줄)
**변경 전**:
```python
rel_x = (landmark.x - ego_agent.x) / sensing_radius
rel_y = (landmark.y - ego_agent.y) / sensing_radius
```

**변경 후**:
```python
rel_x = landmark.x - ego_agent.x
rel_y = landmark.y - ego_agent.y
```

#### 2-3. 장애물 위치 (70-71줄)
**변경 전**:
```python
rel_x = (obstacle.x - ego_agent.x) / sensing_radius
rel_y = (obstacle.y - ego_agent.y) / sensing_radius
```

**변경 후**:
```python
rel_x = obstacle.x - ego_agent.x
rel_y = obstacle.y - ego_agent.y
```

#### 2-4. 엣지 거리 특성 (117, 125, 132줄)
**변경 전**:
```python
# 엣지 특성: 정규화된 거리
normalized_dist = min(edge_distance / sensing_radius, 1.0)
edge_features[ego_idx, i, j, 0] = normalized_dist
```

**변경 후**:
```python
# 엣지 특성: 절대 거리
edge_features[ego_idx, i, j, 0] = edge_distance
```

## 영향을 받는 부분
1. **환경 관측**: 에이전트의 로컬 관측값이 절대 좌표/속도로 변경
2. **그래프 노드 특성**: 모든 노드의 위치/속도 특성이 절대값으로 변경
3. **엣지 특성**: 엣지 거리가 절대 거리로 변경

## 변경 이유
- 정규화로 인한 정보 손실 방지
- 원본 InforMARL과의 성능 비교를 위한 동일 조건 구현
- 네트워크가 절대값에서 직접 학습하도록 변경

## 수정 완료 일시
2025-01-15