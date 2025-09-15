"""
Physics simulation for agent movement and collision handling
기존 physics.py에서 waypoint 의존성 제거, 순수 연속 행동만
"""
import math
import torch
from typing import List
from ..utils.types import Agent2D, Obstacle2D


def execute_continuous_action(agent: Agent2D, action: List[float]):
    """
    순수 연속 행동 실행 - waypoint 제거
    
    Args:
        agent: 에이전트 객체
        action: [dv, dw] - 속도 변화량 (범위: [-1, 1])
    """
    dv, dw = action
    
    # 현재 속도에 변화량 적용
    agent.vx += dv * 0.1  # 속도 변화율 조정
    agent.vy += dw * 0.1
    
    # 최대 속도 제한
    speed = math.sqrt(agent.vx**2 + agent.vy**2)
    if speed > agent.max_speed:
        agent.vx = (agent.vx / speed) * agent.max_speed
        agent.vy = (agent.vy / speed) * agent.max_speed


def update_positions(agents: List[Agent2D], obstacles: List[Obstacle2D],
                    corridor_width: float, corridor_height: float,
                    bottleneck_position: float, bottleneck_width: float) -> int:
    """
    위치 업데이트 및 충돌 처리
    기존 코드에서 가져온 검증된 물리 시뮬레이션
    """
    collision_count = 0
    dt = 0.1  # 시간 간격
    
    # 임시 새 위치 계산
    new_positions = []
    for agent in agents:
        new_x = agent.x + agent.vx * dt
        new_y = agent.y + agent.vy * dt
        new_positions.append((new_x, new_y))
    
    # 각 에이전트 위치 업데이트 및 충돌 처리
    for i, agent in enumerate(agents):
        new_x, new_y = new_positions[i]
        
        # 경계 충돌 검사 및 처리
        if new_x <= agent.radius:
            agent.x = agent.radius
            collision_count += 1
        elif new_x >= corridor_width - agent.radius:
            agent.x = corridor_width - agent.radius
            collision_count += 1
        else:
            agent.x = new_x
        
        if new_y <= agent.radius:
            agent.y = agent.radius
            collision_count += 1
        elif new_y >= corridor_height - agent.radius:
            agent.y = corridor_height - agent.radius
            collision_count += 1
        else:
            agent.y = new_y
        
        # 병목 벽 충돌 처리 - 회색 직사각형 벽들 완전 차단
        center_y = corridor_height / 2
        upper_wall_y = center_y + bottleneck_width / 2  # 5.6
        lower_wall_y = center_y - bottleneck_width / 2  # 4.4
        wall_left = bottleneck_position - 0.5
        wall_right = bottleneck_position + 0.5
        margin = agent.radius
        
        # 회색 벽 영역 완전 차단 - 진입 자체를 막음
        if wall_left - margin <= agent.x <= wall_right + margin:
            # 위쪽 회색 벽 영역: x ∈ [9.5, 10.5], y ∈ [5.6, 10.0]
            if agent.y >= upper_wall_y - margin:
                # 벽 영역에서 완전히 밀어냄
                if agent.x < bottleneck_position:  # 왼쪽으로
                    agent.x = wall_left - margin
                    agent.vx = min(agent.vx, 0.0)  # 오른쪽 속도 차단
                else:  # 오른쪽으로
                    agent.x = wall_right + margin
                    agent.vx = max(agent.vx, 0.0)  # 왼쪽 속도 차단
                collision_count += 1
                
            # 아래쪽 회색 벽 영역: x ∈ [9.5, 10.5], y ∈ [0.0, 4.4] 
            elif agent.y <= lower_wall_y + margin:
                # 벽 영역에서 완전히 밀어냄
                if agent.x < bottleneck_position:  # 왼쪽으로
                    agent.x = wall_left - margin
                    agent.vx = min(agent.vx, 0.0)  # 오른쪽 속도 차단
                else:  # 오른쪽으로
                    agent.x = wall_right + margin
                    agent.vx = max(agent.vx, 0.0)  # 왼쪽 속도 차단
                collision_count += 1
                
            # 병목 통로 내에서 위아래 경계 충돌
            else:  # lower_wall_y < agent.y < upper_wall_y (통로 내부)
                if agent.y + margin > upper_wall_y:
                    agent.y = upper_wall_y - margin
                    collision_count += 1
                elif agent.y - margin < lower_wall_y:
                    agent.y = lower_wall_y + margin  
                    collision_count += 1
        
        # 장애물 충돌 검사
        for obstacle in obstacles:
            dist = agent.get_distance_to(obstacle.x, obstacle.y)
            min_dist = agent.radius + obstacle.radius
            
            if dist < min_dist and dist > 0:
                # 겹침 해결
                dx = agent.x - obstacle.x
                dy = agent.y - obstacle.y
                if dist > 0:
                    dx /= dist
                    dy /= dist
                
                overlap = min_dist - dist
                agent.x += dx * overlap
                agent.y += dy * overlap
                
                # 속도 반발
                dot_product = agent.vx * dx + agent.vy * dy
                agent.vx -= 2 * dot_product * dx * 0.5
                agent.vy -= 2 * dot_product * dy * 0.5
                
                collision_count += 1
    
    # 에이전트 간 충돌 처리
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            agent1, agent2 = agents[i], agents[j]
            
            dist = agent1.get_distance_to_agent(agent2)
            min_dist = agent1.radius + agent2.radius
            
            if dist < min_dist and dist > 0:
                # 겹침 해결
                dx = agent1.x - agent2.x
                dy = agent1.y - agent2.y
                dx /= dist
                dy /= dist
                
                overlap = min_dist - dist
                agent1.x += dx * overlap * 0.5
                agent1.y += dy * overlap * 0.5
                agent2.x -= dx * overlap * 0.5
                agent2.y -= dy * overlap * 0.5
                
                # 속도 교환 (탄성 충돌)
                v1_parallel = agent1.vx * dx + agent1.vy * dy
                v2_parallel = agent2.vx * dx + agent2.vy * dy
                
                agent1.vx += (v2_parallel - v1_parallel) * dx * 0.5
                agent1.vy += (v2_parallel - v1_parallel) * dy * 0.5
                agent2.vx += (v1_parallel - v2_parallel) * dx * 0.5
                agent2.vy += (v1_parallel - v2_parallel) * dy * 0.5
                
                collision_count += 1
    
    return collision_count


# GPU 배치 처리 함수들 (기존 코드에서 가져온 최적화 버전)
def batch_execute_actions_gpu(agents: List[Agent2D], actions: List[List[float]], device) -> torch.Tensor:
    """GPU에서 배치로 행동 실행"""
    if len(agents) == 0 or len(actions) == 0:
        return torch.empty((0, 2), dtype=torch.float32, device=device)
    
    velocities = torch.tensor([[agent.vx, agent.vy] for agent in agents], dtype=torch.float32, device=device)
    max_speeds = torch.tensor([agent.max_speed for agent in agents], dtype=torch.float32, device=device)
    actions_tensor = torch.tensor(actions, dtype=torch.float32, device=device)
    
    # 속도 변화량 적용
    new_velocities = velocities + actions_tensor * 0.1
    
    # 최대 속도 제한
    speeds = torch.norm(new_velocities, dim=1, keepdim=True)
    max_speeds_expanded = max_speeds.unsqueeze(1)
    
    # 속도 제한이 필요한 에이전트만 처리
    speed_mask = speeds.squeeze() > max_speeds
    if speed_mask.any():
        # 정규화된 방향 벡터 * 최대속도
        normalized = new_velocities[speed_mask] / speeds[speed_mask]
        new_velocities[speed_mask] = normalized * max_speeds_expanded[speed_mask]
    
    return new_velocities


def batch_update_positions_gpu(agents: List[Agent2D], new_velocities: torch.Tensor, obstacles: List[Obstacle2D],
                              corridor_width: float, corridor_height: float, 
                              bottleneck_position: float, bottleneck_width: float, device) -> int:
    """GPU에서 배치로 위치 업데이트"""
    collision_count = 0
    dt = 0.1
    
    # GPU 텐서를 CPU로 변환하여 개별 처리
    velocities_cpu = new_velocities.cpu().numpy()
    
    for i, agent in enumerate(agents):
        agent.vx, agent.vy = velocities_cpu[i]
        
        # 위치 업데이트
        new_x = agent.x + agent.vx * dt
        new_y = agent.y + agent.vy * dt
        
        # 경계 충돌 검사 (기존 로직과 동일)
        if new_x <= agent.radius:
            agent.x = agent.radius
            collision_count += 1
        elif new_x >= corridor_width - agent.radius:
            agent.x = corridor_width - agent.radius
            collision_count += 1
        else:
            agent.x = new_x
        
        if new_y <= agent.radius:
            agent.y = agent.radius
            collision_count += 1
        elif new_y >= corridor_height - agent.radius:
            agent.y = corridor_height - agent.radius
            collision_count += 1
        else:
            agent.y = new_y
        
        # 병목 벽 충돌 처리 - GPU 버전에도 추가!
        center_y = corridor_height / 2
        upper_wall_y = center_y + bottleneck_width / 2  # 5.6
        lower_wall_y = center_y - bottleneck_width / 2  # 4.4
        wall_left = bottleneck_position - 0.5
        wall_right = bottleneck_position + 0.5
        margin = agent.radius
        
        # 회색 벽 영역 완전 차단 - 진입 자체를 막음
        if wall_left - margin <= agent.x <= wall_right + margin:
            # 위쪽 회색 벽 영역: x ∈ [9.5, 10.5], y ∈ [5.6, 10.0]
            if agent.y >= upper_wall_y - margin:
                # 벽 영역에서 완전히 밀어냄
                if agent.x < bottleneck_position:  # 왼쪽으로
                    agent.x = wall_left - margin
                    agent.vx = min(agent.vx, 0.0)  # 오른쪽 속도 차단
                else:  # 오른쪽으로
                    agent.x = wall_right + margin
                    agent.vx = max(agent.vx, 0.0)  # 왼쪽 속도 차단
                collision_count += 1
                
            # 아래쪽 회색 벽 영역: x ∈ [9.5, 10.5], y ∈ [0.0, 4.4] 
            elif agent.y <= lower_wall_y + margin:
                # 벽 영역에서 완전히 밀어냄
                if agent.x < bottleneck_position:  # 왼쪽으로
                    agent.x = wall_left - margin
                    agent.vx = min(agent.vx, 0.0)  # 오른쪽 속도 차단
                else:  # 오른쪽으로
                    agent.x = wall_right + margin
                    agent.vx = max(agent.vx, 0.0)  # 왼쪽 속도 차단
                collision_count += 1
                
            # 병목 통로 내에서 위아래 경계 충돌
            else:  # lower_wall_y < agent.y < upper_wall_y (통로 내부)
                if agent.y + margin > upper_wall_y:
                    agent.y = upper_wall_y - margin
                    collision_count += 1
                elif agent.y - margin < lower_wall_y:
                    agent.y = lower_wall_y + margin  
                    collision_count += 1
    
    # 에이전트 간 충돌은 기존 함수 재사용
    additional_collisions = 0
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            agent1, agent2 = agents[i], agents[j]
            
            dist = agent1.get_distance_to_agent(agent2)
            min_dist = agent1.radius + agent2.radius
            
            if dist < min_dist and dist > 0:
                # 겹침 해결 (기존 로직과 동일)
                dx = agent1.x - agent2.x
                dy = agent1.y - agent2.y
                dx /= dist
                dy /= dist
                
                overlap = min_dist - dist
                agent1.x += dx * overlap * 0.5
                agent1.y += dy * overlap * 0.5
                agent2.x -= dx * overlap * 0.5
                agent2.y -= dy * overlap * 0.5
                
                additional_collisions += 1
    
    return collision_count + additional_collisions