"""
모델 저장 및 불러오기 유틸리티
"""
import os
import torch
import json
from datetime import datetime


class ModelSaver:
    def __init__(self, save_dir='saved_models'):
        """
        Args:
            save_dir: 모델 저장 디렉토리
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def save_checkpoint(self, models, optimizer, global_step, episode_count,
                       avg_reward, success_rate, config=None, suffix=""):
        """
        체크포인트 저장

        Args:
            models: InforMARLModels 객체
            optimizer: 옵티마이저
            global_step: 현재 스텝 수
            episode_count: 에피소드 수
            avg_reward: 평균 보상
            success_rate: 성공률
            config: 학습 설정 (딕셔너리)
            suffix: 파일명 접미사
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if suffix:
            filename = f"checkpoint_{timestamp}_{suffix}.pth"
        else:
            filename = f"checkpoint_{timestamp}_step{global_step}.pth"

        filepath = os.path.join(self.save_dir, filename)

        checkpoint = {
            'global_step': global_step,
            'episode_count': episode_count,
            'avg_reward': avg_reward,
            'success_rate': success_rate,
            'timestamp': timestamp,
            'models_state_dict': {
                'gnn': models.gnn.state_dict(),
                'actor': models.actor.state_dict(),
                'critic': models.critic.state_dict()
            },
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config
        }

        torch.save(checkpoint, filepath)
        print(f"체크포인트 저장: {filepath}")
        print(f"  - 스텝: {global_step:,}")
        print(f"  - 에피소드: {episode_count}")
        print(f"  - 평균 보상: {avg_reward:.2f}")
        print(f"  - 성공률: {success_rate:.1f}%")

        return filepath

    def save_best_model(self, models, optimizer, global_step, avg_reward, success_rate, config=None):
        """
        최고 성능 모델 저장
        """
        return self.save_checkpoint(
            models, optimizer, global_step, 0, avg_reward, success_rate, config, "best"
        )

    def load_checkpoint(self, filepath, models, optimizer=None, device='cuda'):
        """
        체크포인트 불러오기

        Args:
            filepath: 체크포인트 파일 경로
            models: InforMARLModels 객체
            optimizer: 옵티마이저 (선택사항)
            device: 디바이스

        Returns:
            dict: 로드된 정보 (global_step, episode_count 등)
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"체크포인트 파일을 찾을 수 없습니다: {filepath}")

        print(f"체크포인트 로딩: {filepath}")
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)

        # 모델 가중치 로드
        models.gnn.load_state_dict(checkpoint['models_state_dict']['gnn'])
        models.actor.load_state_dict(checkpoint['models_state_dict']['actor'])
        models.critic.load_state_dict(checkpoint['models_state_dict']['critic'])

        # 옵티마이저 상태 로드 (있는 경우)
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        info = {
            'global_step': checkpoint.get('global_step', 0),
            'episode_count': checkpoint.get('episode_count', 0),
            'avg_reward': checkpoint.get('avg_reward', 0.0),
            'success_rate': checkpoint.get('success_rate', 0.0),
            'timestamp': checkpoint.get('timestamp', 'unknown'),
            'config': checkpoint.get('config', {})
        }

        print(f"체크포인트 로드 완료:")
        print(f"  - 스텝: {info['global_step']:,}")
        print(f"  - 에피소드: {info['episode_count']}")
        print(f"  - 평균 보상: {info['avg_reward']:.2f}")
        print(f"  - 성공률: {info['success_rate']:.1f}%")
        print(f"  - 저장 시간: {info['timestamp']}")

        return info

    def list_checkpoints(self):
        """
        저장된 체크포인트 목록 반환
        """
        checkpoints = []
        if not os.path.exists(self.save_dir):
            return checkpoints

        for filename in os.listdir(self.save_dir):
            if filename.endswith('.pth'):
                filepath = os.path.join(self.save_dir, filename)
                try:
                    checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
                    info = {
                        'filepath': filepath,
                        'filename': filename,
                        'global_step': checkpoint.get('global_step', 0),
                        'episode_count': checkpoint.get('episode_count', 0),
                        'avg_reward': checkpoint.get('avg_reward', 0.0),
                        'success_rate': checkpoint.get('success_rate', 0.0),
                        'timestamp': checkpoint.get('timestamp', 'unknown')
                    }
                    checkpoints.append(info)
                except Exception as e:
                    print(f"체크포인트 로드 실패: {filename} - {e}")

        # 스텝 수로 정렬
        checkpoints.sort(key=lambda x: x['global_step'])
        return checkpoints

    def find_latest_checkpoint(self):
        """
        가장 최근 체크포인트 찾기
        """
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            return None

        return max(checkpoints, key=lambda x: x['global_step'])

    def find_best_checkpoint(self):
        """
        가장 좋은 성능의 체크포인트 찾기 (success_rate 기준)
        """
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            return None

        return max(checkpoints, key=lambda x: x['success_rate'])


def get_model_config(**kwargs):
    """
    모델 구조 및 환경 설정만 반환 (훈련 목표/실험 설정 제외)

    Args:
        **kwargs: 동적으로 전달받을 설정 값들

    Returns:
        dict: 모델/환경 설정 (재현 가능한 설정만)
    """
    # 모델 재현에 필요한 핵심 설정만
    config = {
        # === 환경 설정 (모델 입력 크기 결정) ===
        'NUM_AGENTS': 4,
        'SENSING_RADIUS': 7.0,
        'MAX_NODES': 30,
        'EDGE_DIM': 1,

        # === 모델 구조 설정 ===
        'LOCAL_OBS_DIM': 6,
        'ACTION_DIM': 2,
        'GNN_OUTPUT_DIM': 64,
        'ACTOR_HIDDEN_DIM': 64,
        'CRITIC_HIDDEN_DIM': 64,
        'NUM_HEADS': 4,
        'NUM_LAYERS': 2,
        'DROPOUT': 0.1,

        # === 학습 하이퍼파라미터 ===
        'LEARNING_RATE': 3e-4,
        'CLIP_EPSILON': 0.2,
        'VALUE_LOSS_COEF': 0.5,
        'ENTROPY_COEF': 0.01,
        'GAMMA': 0.99,
        'GAE_LAMBDA': 0.95,
        'MAX_GRAD_NORM': 0.5,
        'MINI_BATCH_SIZE': 64,
        'PPO_EPOCHS': 4,

        # === 물리 환경 설정 (재현성) ===
        'CORRIDOR_WIDTH': 20.0,
        'CORRIDOR_HEIGHT': 10.0,
        'BOTTLENECK_WIDTH': 6.0,
        'BOTTLENECK_POS': 10.0,
        'AGENT_RADIUS': 0.5,
        'MAX_SPEED': 1.0,
        'MAX_TIMESTEPS': 300,
        'USE_GPU_PHYSICS': True,
        'INCLUDE_OBSTACLES': True
    }

    # 동적으로 전달된 설정으로 업데이트
    config.update(kwargs)
    return config


def get_training_config(**kwargs):
    """하위 호환성을 위한 래퍼 함수"""
    return get_model_config(**kwargs)