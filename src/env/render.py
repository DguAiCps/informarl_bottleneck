"""
Rendering and visualization for bottleneck environment
기존 render.py에서 waypoint 표시 제거
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
from typing import List
from ..utils.types import Agent2D, Landmark2D, Obstacle2D


class BottleneckRenderer:
    """병목 환경 렌더러 - waypoint 표시 제거"""
    
    def __init__(self):
        self.fig = None
        self.ax = None
    
    def render(self, agents: List[Agent2D], landmarks: List[Landmark2D], obstacles: List[Obstacle2D],
               corridor_width: float, corridor_height: float, bottleneck_position: float, 
               bottleneck_width: float, timestep: int, success_count: int, collision_count: int,
               show_obstacles: bool = True):
        """환경 렌더링"""
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(14, 8))
            plt.ion()  # 인터랙티브 모드
        
        self.ax.clear()
        
        # 환경 전체 배경
        self.ax.fill_between([0, corridor_width], 0, corridor_height, 
                            color='lightblue', alpha=0.2, label='복도')
        
        # 병목 구역 표시 (회색 벽들)
        center_y = corridor_height / 2
        bottleneck_x = bottleneck_position
        
        # 위쪽 벽
        upper_wall = patches.Rectangle(
            (bottleneck_x - 0.5, center_y + bottleneck_width/2), 
            1.0, corridor_height - (center_y + bottleneck_width/2),
            facecolor='darkgray', edgecolor='black', linewidth=2
        )
        self.ax.add_patch(upper_wall)
        
        # 아래쪽 벽  
        lower_wall = patches.Rectangle(
            (bottleneck_x - 0.5, 0), 
            1.0, center_y - bottleneck_width/2,
            facecolor='darkgray', edgecolor='black', linewidth=2
        )
        self.ax.add_patch(lower_wall)
        
        # 병목 통로 표시 (노란색으로 강조)
        bottleneck_passage = patches.Rectangle(
            (bottleneck_x - 0.5, center_y - bottleneck_width/2),
            1.0, bottleneck_width,
            facecolor='yellow', alpha=0.3, edgecolor='orange', linewidth=2
        )
        self.ax.add_patch(bottleneck_passage)
        
        # 환경 경계 테두리
        boundary = patches.Rectangle(
            (0, 0), corridor_width, corridor_height,
            linewidth=3, edgecolor='black', facecolor='none'
        )
        self.ax.add_patch(boundary)
        
        # 장애물 그리기
        if show_obstacles and obstacles:
            for obstacle in obstacles:
                obs_circle = patches.Circle(
                    (obstacle.x, obstacle.y), obstacle.radius,
                    color='red', alpha=0.9, edgecolor='darkred', linewidth=2
                )
                self.ax.add_patch(obs_circle)
        
        # 목표 지점 그리기
        for i, landmark in enumerate(landmarks):
            goal_circle = patches.Circle(
                (landmark.x, landmark.y), landmark.radius,
                color='green', alpha=0.7, edgecolor='darkgreen', linewidth=2
            )
            self.ax.add_patch(goal_circle)
            self.ax.text(landmark.x, landmark.y, f'G{i}', 
                        ha='center', va='center', fontweight='bold', fontsize=10)
        
        # 에이전트 그리기
        colors = ['blue', 'orange', 'purple', 'brown', 'pink', 'cyan']
        for i, agent in enumerate(agents):
            color = colors[i % len(colors)]
            
            # 에이전트 원
            agent_circle = patches.Circle(
                (agent.x, agent.y), agent.radius,
                color=color, alpha=0.8, edgecolor='black', linewidth=1
            )
            self.ax.add_patch(agent_circle)
            
            # 에이전트 ID
            self.ax.text(agent.x, agent.y, str(i), 
                        ha='center', va='center', fontweight='bold', 
                        fontsize=10, color='white')
            
            # 속도 벡터 표시
            if abs(agent.vx) > 0.1 or abs(agent.vy) > 0.1:
                self.ax.arrow(agent.x, agent.y, agent.vx, agent.vy,
                             head_width=0.2, head_length=0.2, 
                             fc=color, ec=color, alpha=0.7, linewidth=2)
            
            # 목표까지의 방향 선 (얇게)
            target = landmarks[agent.target_id]
            self.ax.plot([agent.x, target.x], [agent.y, target.y], 
                        color=color, linestyle='--', alpha=0.3, linewidth=1)
        
        # 상태 정보 표시
        info_text = f"Step: {timestep}\n"
        info_text += f"Success: {success_count}/{len(agents)}\n"
        info_text += f"Collisions: {collision_count}"
        
        self.ax.text(0.02, 0.98, info_text, transform=self.ax.transAxes,
                    verticalalignment='top', fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
        
        # 축 설정
        self.ax.set_xlim(-0.5, corridor_width + 0.5)
        self.ax.set_ylim(-0.5, corridor_height + 0.5)
        self.ax.set_aspect('equal')
        self.ax.set_title('InforMARL Bottleneck Navigation', fontsize=16)
        self.ax.grid(True, alpha=0.3)
        
        plt.draw()
        plt.pause(0.001)
    
    def close(self):
        """렌더러 종료"""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None