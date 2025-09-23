# Clean InforMARL Implementation

순수 InforMARL로 병목구간 통과 학습 - 불필요한 기능 제거

## 핵심 특징

- **순수 InforMARL**: waypoint 제거, 강화학습만으로 학습
- **공유 네트워크**: 모든 에이전트가 GNN + Actor/Critic 공유  
- **연속 행동공간**: [dv, dw] 속도 변화량 출력
- **동적 그래프**: 센싱 범위 기반 그래프 구성
- **GPU 배치 처리**: 효율적인 학습
- **PPO 알고리즘**: 안정적인 정책 최적화

## 프로젝트 구조

```
clean_informarl/
├── main.py                    # 메인 실행 파일 (모든 설정 하드코딩)
├── src/
│   ├── env/
│   │   └── bottleneck_env.py  # 병목 환경 (waypoint 제거)
│   ├── models/
│   │   ├── gnn.py            # Graph Neural Network
│   │   └── policy.py         # Actor-Critic Networks
│   └── utils/
│       └── renderer.py       # 간단한 시각화
└── requirements.txt
```

## 실행 방법

### 기본 학습
```bash
# 의존성 설치
pip install -r requirements.txt

# 새로운 학습 시작
python main_stepwise.py

# 데모 모드 (훈련 후 y 입력)
```

### 모델 저장/불러오기 및 학습 재개

#### 1. 새로운 학습 시작
```bash
python main_stepwise.py
```
- 처음부터 새로 학습 시작
- `saved_models/` 디렉토리에 자동 저장됨

#### 2. 자동 저장 시점
- **50,000 스텝마다**: `checkpoint_날짜시간_auto_50k.pth`
- **10,000 스텝마다**: 최고 성능 달성 시 `checkpoint_날짜시간_best.pth`
- **훈련 완료 시**: `checkpoint_날짜시간_final.pth`

#### 3. 학습 재개
```bash
# 특정 체크포인트에서 재개
python main_stepwise.py --resume saved_models/checkpoint_20231216_143022_auto_100k.pth

# 다른 디렉토리 사용
python main_stepwise.py --resume /path/to/checkpoint.pth --checkpoint-dir my_models
```

#### 4. 테스트만 실행
```bash
# 최고 성능 모델 자동 선택하여 테스트
python main_stepwise.py --test-only

# 특정 모델로 테스트
python main_stepwise.py --test-only --resume saved_models/checkpoint_20231216_143022_best.pth
```

#### 5. 저장된 파일명 예시
- `checkpoint_20231216_143022_auto_50k.pth` - 50k 스텝 자동 저장
- `checkpoint_20231216_143022_best.pth` - 최고 성능 모델
- `checkpoint_20231216_143022_final.pth` - 최종 완료 모델
- `checkpoint_20231216_143022_step1500000.pth` - 일반 체크포인트

#### 6. 주의사항
- **Ctrl+C로 중단하면 저장 안됨** (다음 자동 저장까지 기다려야 함)
- 최대 2M 스텝 중 50k마다 저장되므로 최대 49,999 스텝만 손실 가능
- 최고 성능 모델은 따로 `best` 파일로 보관됨

## 하드코딩된 설정

모든 설정이 main.py에 하드코딩되어 있어 쉽게 수정 가능:

```python
# 환경 설정
NUM_AGENTS = 4
CORRIDOR_WIDTH = 20.0
BOTTLENECK_WIDTH = 1.2
SENSING_RADIUS = 3.0

# 훈련 설정  
NUM_EPISODES = 1000
LEARNING_RATE = 0.003
BATCH_SIZE = 64
```

## InforMARL 핵심 플로우

1. **동적 그래프 생성**: 각 에이전트마다 센싱 범위 내 이웃 정보를 그래프로 구성
2. **공유 GNN**: 모든 에이전트가 같은 GNN으로 정보 집계  
3. **정책 결정**: 로컬 관측 + GNN 임베딩으로 연속 행동 출력
4. **배치 학습**: 모든 에이전트 경험을 모아 PPO로 네트워크 업데이트

## 기존 프로젝트와의 차이점

- ❌ waypoint 기반 경로 계획 제거
- ❌ YAML 설정 파일 제거  
- ❌ 프로세스 병렬처리 제거
- ❌ 복잡한 폴백 메커니즘 제거
- ✅ 순수 강화학습만으로 병목 통과 학습
- ✅ GPU 배치 처리 유지
- ✅ 동적 그래프 + GPU 가속 유지