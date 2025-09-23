"""
환경 체크 스크립트 - 오류 확인 및 디버깅
"""
import sys
import os
import torch
import numpy as np

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
if not os.path.exists(src_path):
    # 상위 디렉토리에서 찾기
    parent_dir = os.path.dirname(current_dir)
    for subdir in os.listdir(parent_dir):
        potential_path = os.path.join(parent_dir, subdir, 'src')
        if os.path.exists(potential_path):
            src_path = potential_path
            break
    else:
        print("❌ src 디렉토리를 찾을 수 없습니다")
        sys.exit(1)

sys.path.insert(0, os.path.dirname(src_path))

from src.env.bottleneck_env import CleanBottleneckEnv
from src.env.graph_builder_unimp import build_unimp_graph_observations

def test_environment():
    """환경 테스트"""
    print("=== 환경 초기화 테스트 ===")
    
    try:
        # 환경 생성
        env = CleanBottleneckEnv(
            num_agents=4,
            corridor_width=20.0,
            corridor_height=10.0,
            bottleneck_width=2.0,
            bottleneck_pos=10.0,
            agent_radius=0.5,
            sensing_radius=7.0,
            max_speed=1.0,
            max_timesteps=300,
            device=torch.device('cpu'),  # CPU로 안전하게 테스트
            use_gpu_physics=False,
            include_obstacles=True
        )
        print("✓ 환경 초기화 성공")
        
        # 리셋 테스트
        print("\n=== 환경 리셋 테스트 ===")
        env.reset()
        print("✓ 환경 리셋 성공")
        print(f"  - 에이전트 수: {len(env.agents)}")
        print(f"  - 랜드마크 수: {len(env.landmarks)}")
        print(f"  - 장애물 수: {len(env.obstacles)}")
        
        # 에이전트 정보 확인
        print("\n=== 에이전트 상태 확인 ===")
        for i, agent in enumerate(env.agents):
            print(f"Agent {i}: pos=({agent.x:.2f}, {agent.y:.2f}), vel=({agent.vx:.2f}, {agent.vy:.2f}), target={agent.target_id}")
        
        # 로컬 관측 테스트
        print("\n=== 로컬 관측 테스트 ===")
        for i in range(len(env.agents)):
            obs = env.get_local_observation(i)
            print(f"Agent {i} obs: {obs}")
            if len(obs) != 6:
                print(f"❌ 관측 차원 오류: 예상 6, 실제 {len(obs)}")
                return False
        print("✓ 로컬 관측 성공")
        
        # 그래프 생성 테스트
        print("\n=== 그래프 생성 테스트 ===")
        try:
            node_obs, adj, entity_types, edge_features = build_unimp_graph_observations(
                env.agents, env.landmarks, env.obstacles, env.sensing_radius, max_nodes=30
            )
            print(f"✓ 그래프 생성 성공")
            print(f"  - node_obs shape: {node_obs.shape}")
            print(f"  - adj shape: {adj.shape}")
            print(f"  - entity_types shape: {entity_types.shape}")
            print(f"  - edge_features shape: {edge_features.shape}")
            
            # 차원 검증
            expected_shape = (len(env.agents), 30, 6)
            if node_obs.shape != expected_shape:
                print(f"❌ node_obs 차원 오류: 예상 {expected_shape}, 실제 {node_obs.shape}")
                return False
                
        except Exception as e:
            print(f"❌ 그래프 생성 오류: {e}")
            return False
        
        # 액션 테스트
        print("\n=== 액션 실행 테스트 ===")
        try:
            # 랜덤 액션 생성
            actions = [[np.random.uniform(-1, 1), np.random.uniform(-1, 1)] for _ in range(len(env.agents))]
            print(f"Actions: {actions}")
            
            # 스텝 실행
            obs, rewards, done, info = env.step(actions)
            print(f"✓ 스텝 실행 성공")
            print(f"  - Rewards: {rewards}")
            print(f"  - Done: {done}")
            print(f"  - Info: {info}")
            
            if len(rewards) != len(env.agents):
                print(f"❌ 보상 차원 오류: 예상 {len(env.agents)}, 실제 {len(rewards)}")
                return False
                
        except Exception as e:
            print(f"❌ 액션 실행 오류: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 위치 범위 확인
        print("\n=== 위치 범위 확인 ===")
        for i, agent in enumerate(env.agents):
            if not (0 <= agent.x <= env.corridor_width):
                print(f"❌ Agent {i} x 위치 범위 오류: {agent.x} (범위: 0-{env.corridor_width})")
                return False
            if not (0 <= agent.y <= env.corridor_height):
                print(f"❌ Agent {i} y 위치 범위 오류: {agent.y} (범위: 0-{env.corridor_height})")
                return False
        print("✓ 위치 범위 정상")
        
        env.close()
        print("\n=== 모든 테스트 통과 ===")
        return True
        
    except Exception as e:
        print(f"❌ 환경 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_value_ranges():
    """정규화 제거 후 값 범위 확인"""
    print("\n=== 값 범위 테스트 ===")
    
    env = CleanBottleneckEnv(
        num_agents=4,
        corridor_width=20.0,
        corridor_height=10.0,
        sensing_radius=7.0,
        device=torch.device('cpu'),
        use_gpu_physics=False
    )
    env.reset()
    
    # 로컬 관측값 범위
    print("로컬 관측값 범위:")
    for i in range(len(env.agents)):
        obs = env.get_local_observation(i)
        print(f"Agent {i}: x={obs[0]:.2f}, y={obs[1]:.2f}, vx={obs[2]:.2f}, vy={obs[3]:.2f}, goal_dx={obs[4]:.2f}, goal_dy={obs[5]:.2f}")
    
    # 그래프 노드 특성 범위
    print("\n그래프 노드 특성 범위:")
    node_obs, _, _, edge_features = build_unimp_graph_observations(
        env.agents, env.landmarks, env.obstacles, env.sensing_radius
    )
    
    for agent_idx in range(len(env.agents)):
        print(f"Agent {agent_idx} 그래프:")
        for node_idx in range(min(5, node_obs.shape[1])):  # 처음 5개 노드만
            features = node_obs[agent_idx, node_idx]
            if np.any(features != 0):  # 0이 아닌 노드만
                print(f"  Node {node_idx}: {features}")
    
    print(f"\nEdge features 범위: min={edge_features.min():.2f}, max={edge_features.max():.2f}")

if __name__ == "__main__":
    success = test_environment()
    if success:
        test_value_ranges()
        print("\n🎉 환경 체크 완료: 모든 테스트 통과!")
    else:
        print("\n💥 환경 체크 실패: 오류 수정 필요")