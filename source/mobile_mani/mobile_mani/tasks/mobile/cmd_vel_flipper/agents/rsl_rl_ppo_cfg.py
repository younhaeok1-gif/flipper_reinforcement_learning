from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg

@configclass
class PPOActorCriticCfg(RslRlPpoActorCriticCfg):
    """정책 네트워크 설정"""
    class_name: str = "ActorCritic"
    init_noise_std: float = 1.0
    actor_hidden_dims: list[int] = [512, 256, 128] 
    critic_hidden_dims: list[int] = [512, 256, 128]
    activation: str = "elu"


@configclass
class PPOAlgorithmCfg:
    """PPO 알고리즘 하이퍼파라미터"""
    class_name: str = "PPO"
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    entropy_coef: float = 0.0
    num_learning_epochs: int = 5
    num_mini_batches: int = 8
    learning_rate: float = 3.0e-4
    schedule: str = "adaptive"
    gamma: float = 0.99
    lam: float = 0.95
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0

@configclass
class PPORunnerCfg:
    """최종 실행 설정 (Runner)"""
    class_name: str = "OnPolicyRunner"

    seed: int = 42
    device: str = "cuda:0"
    num_steps_per_env: int = 24
    max_iterations: int = 1500
    save_interval: int = 50
    experiment_name: str = "mobile_cmd_vel_flipper_rsl_rl"
    empirical_normalization: bool = False
    
    # [수정됨] None -> 딕셔너리로 변경
    # 의미: "policy" 그룹의 관측값을 환경의 "policy" 키에서 가져온다.
    obs_groups: dict = {"policy": ["policy"]}

    clip_actions: float = 1.0

    # 필수 속성들
    run_name: str = ""
    logger: str = "tensorboard"
    resume: bool = False
    load_run: str = ".*"
    load_checkpoint: str = "model_.*.pt"
    
    # WandB 설정
    wandb_project: str = "mobile_cmd_vel_flipper"
    wandb_entity: str = None
    wandb_log_model: bool = False

    # 연결
    policy: PPOActorCriticCfg = PPOActorCriticCfg()
    algorithm: PPOAlgorithmCfg = PPOAlgorithmCfg()
