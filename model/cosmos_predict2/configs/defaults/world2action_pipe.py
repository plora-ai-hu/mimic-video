from hydra.core.config_store import ConfigStore

from cosmos_predict2.configs.config_world2action import SchedulerConfig, World2ActionPipelineConfig
from cosmos_predict2.configs.defaults.ema import EMAConfig
from cosmos_predict2.models.text2image_dit import SACConfig
from cosmos_predict2.models.world2action_dit import World2ActionDIT
from imaginaire.lazy_config import LazyCall as L

ACTION_DECODER_NETS = {
    "libero": L(World2ActionDIT)(
        max_horizon=61,
        in_channels=10,
        out_channels=10,
        model_channels=1024,
        num_blocks=24,
        num_heads=8,
        mlp_ratio=4.0,
        atten_backend="flash_attn_no_cp",
        crossattn_emb_channels=2048,
        use_adaln_lora=True,
        adaln_lora_dim=128,
        pair_timestep_feature_rank=1024,
        sac_config=SACConfig(mode="none", every_n_blocks=1),
    ),
    "bridge": L(World2ActionDIT)(
        max_horizon=16,
        in_channels=10,
        out_channels=10,
        model_channels=1024,
        num_blocks=24,
        num_heads=8,
        mlp_ratio=4.0,
        atten_backend="flash_attn_no_cp",
        crossattn_emb_channels=2048,
        use_adaln_lora=True,
        adaln_lora_dim=128,
        pair_timestep_feature_rank=1024,
        sac_config=SACConfig(mode="none", every_n_blocks=1),
    ),
    # SO100: joint-space control. in_channels = proprio dim (6 joints),
    # out_channels = action dim (6 joints). max_horizon >= action horizon (15) + 1.
    "so100": L(World2ActionDIT)(
        max_horizon=16,
        in_channels=6,
        out_channels=6,
        model_channels=1024,
        num_blocks=24,
        num_heads=8,
        mlp_ratio=4.0,
        # torch SDPA backend: flash-attn 2.6.3 has no sm_120 (Blackwell/RTX 5090) kernels.
        # Numerically identical here and the decoder is tiny, so no perf loss.
        atten_backend="torch",
        crossattn_emb_channels=2048,
        use_adaln_lora=True,
        adaln_lora_dim=128,
        pair_timestep_feature_rank=1024,
        sac_config=SACConfig(mode="none", every_n_blocks=1),
    ),
    # One SO-101 arm (6 joints): absolute joint-space state and actions.
    # Selected by data configs whose name starts with "so101_1arm".
    "so101_1arm": L(World2ActionDIT)(
        max_horizon=16,
        in_channels=6,
        out_channels=6,
        model_channels=1024,
        num_blocks=24,
        num_heads=8,
        mlp_ratio=4.0,
        atten_backend="torch",
        crossattn_emb_channels=2048,
        use_adaln_lora=True,
        adaln_lora_dim=128,
        pair_timestep_feature_rank=1024,
        sac_config=SACConfig(mode="none", every_n_blocks=1),
    ),
    # Two SO-101 arms (2 x 6 joints, left arm first): absolute joint-space state and actions.
    # Selected by data configs whose name starts with "so101_2arm".
    "so101_2arm": L(World2ActionDIT)(
        max_horizon=16,
        in_channels=12,
        out_channels=12,
        model_channels=1024,
        num_blocks=24,
        num_heads=8,
        mlp_ratio=4.0,
        atten_backend="torch",
        crossattn_emb_channels=2048,
        use_adaln_lora=True,
        adaln_lora_dim=128,
        pair_timestep_feature_rank=1024,
        sac_config=SACConfig(mode="none", every_n_blocks=1),
    ),
}


def register_pipe() -> None:
    cs = ConfigStore.instance()

    for name, net in ACTION_DECODER_NETS.items():
        cs.store(
            group="action_pipe",
            package="action_pipe",
            name=f"w2a_{name}",
            node=L(World2ActionPipelineConfig)(
                precision="bfloat16",
                scheduler=SchedulerConfig(alpha=1.0, beta=1.0, num_denoising_steps=10),
                net=net,
                ema=EMAConfig(enabled=False),
            ),
        )
