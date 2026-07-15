"""MaskablePPO policy with a public actor and privileged training critic."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from env.jass_aec_env import OBS_SIZE

try:
    from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
    from stable_baselines3.common.torch_layers import FlattenExtractor
except ImportError as exc:  # pragma: no cover - guarded by the ``rl`` extra
    raise ImportError("sb3-contrib is required for the CTDE policy") from exc


def _network(input_size: int, widths: list[int], activation: type[nn.Module]) -> nn.Module:
    layers: list[nn.Module] = []
    previous = input_size
    for width in widths:
        layers.extend((nn.Linear(previous, width), activation()))
        previous = width
    return nn.Sequential(*layers) if layers else nn.Identity()


class PublicActorPrivilegedCriticExtractor(nn.Module):
    """Route only the canonical public prefix into the policy network."""

    def __init__(
        self,
        feature_dim: int,
        net_arch: list[int] | dict[str, list[int]],
        activation_fn: type[nn.Module],
        public_observation_size: int = OBS_SIZE,
    ) -> None:
        super().__init__()
        if not 0 < public_observation_size <= feature_dim:
            raise ValueError("public_observation_size must fit inside the feature vector")
        self.public_observation_size = public_observation_size
        if isinstance(net_arch, dict):
            actor_widths = list(net_arch.get("pi", ()))
            critic_widths = list(net_arch.get("vf", ()))
        else:
            actor_widths = list(net_arch)
            critic_widths = list(net_arch)
        if any(width <= 0 for width in (*actor_widths, *critic_widths)):
            raise ValueError("network widths must be positive")

        self.policy_net = _network(public_observation_size, actor_widths, activation_fn)
        self.value_net = _network(feature_dim, critic_widths, activation_fn)
        self.latent_dim_pi = actor_widths[-1] if actor_widths else public_observation_size
        self.latent_dim_vf = critic_widths[-1] if critic_widths else feature_dim

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(features[..., : self.public_observation_size])

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)


class CTDEMaskableActorCriticPolicy(MaskableActorCriticPolicy):
    """A decentralized actor whose critic may use the appended private suffix."""

    def __init__(
        self,
        *args: Any,
        public_observation_size: int = OBS_SIZE,
        **kwargs: Any,
    ) -> None:
        if public_observation_size != OBS_SIZE:
            raise ValueError(
                f"CTDE actor input must be exactly the canonical public prefix ({OBS_SIZE})"
            )
        extractor = kwargs.get("features_extractor_class", FlattenExtractor)
        if extractor is not FlattenExtractor:
            raise ValueError(
                "CTDE requires FlattenExtractor so private features cannot mix into the actor"
            )
        if kwargs.get("share_features_extractor", True) is not True:
            raise ValueError("CTDE requires the audited shared identity feature extractor")
        self.public_observation_size = public_observation_size
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        from rl.privileged_observation import CTDE_OBS_SIZE

        if self.features_dim != CTDE_OBS_SIZE:
            raise ValueError(
                f"CTDE policy requires {CTDE_OBS_SIZE} flat features, got {self.features_dim}"
            )
        self.mlp_extractor = PublicActorPrivilegedCriticExtractor(
            self.features_dim,
            self.net_arch,
            self.activation_fn,
            self.public_observation_size,
        ).to(self.device)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data["public_observation_size"] = self.public_observation_size
        return data


__all__ = [
    "CTDEMaskableActorCriticPolicy",
    "PublicActorPrivilegedCriticExtractor",
]
