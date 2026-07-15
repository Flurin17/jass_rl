from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from types import MappingProxyType

from .cards import MODE_OBEABE, MODE_TRUMP, MODE_UNEUFE, SUITS
from .legal_moves import RuleSet

STANDARD_PROFILE_VERSION = "schieber-pagat-v1"
ALL_CONTRACTS_X1_PROFILE_VERSION = "schieber-all-contracts-x1-v1"
LEGACY_UNMULTIPLIED_PROFILE_VERSION = "schieber-legacy-unmultiplied-v1"

# Trump contracts are keyed by their Swiss suit name.  The no-trump
# contracts are keyed by mode.  A copy is stored in each profile so callers
# can supply ordinary dictionaries without sharing mutable configuration.
STANDARD_CONTRACT_FACTORS: Mapping[str, int] = MappingProxyType(
    {
        "eicheln": 1,
        "rosen": 1,
        "schilten": 2,
        "schellen": 2,
        MODE_OBEABE: 3,
        MODE_UNEUFE: 4,
    }
)
ALL_CONTRACTS_X1_FACTORS: Mapping[str, int] = MappingProxyType(
    {key: 1 for key in STANDARD_CONTRACT_FACTORS}
)


def _standard_contract_factors() -> dict[str, int]:
    return dict(STANDARD_CONTRACT_FACTORS)


@dataclass(frozen=True)
class RulesetConfig:
    """Versioned rules profile for one Schieber round.

    ``contract_factors`` uses suit names for trump contracts and mode names
    for Obeabe and Uneufe.  Match is added to the raw trick total before the
    factor is applied.
    """

    # Keep the original three fields first for positional compatibility.
    allow_stock: bool = True
    allow_weis: bool = True
    legal_moves: RuleSet = field(default_factory=RuleSet)
    version: str = STANDARD_PROFILE_VERSION
    contract_factors: Mapping[str, int] = field(
        default_factory=_standard_contract_factors
    )
    match_bonus: int = 100

    def __post_init__(self) -> None:
        if not isinstance(self.allow_stock, bool) or not isinstance(self.allow_weis, bool):
            raise TypeError("allow_stock and allow_weis must be booleans")
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("rules profile version must be a non-empty string")
        if not isinstance(self.legal_moves, RuleSet):
            raise TypeError("legal_moves must be a RuleSet")
        if (
            not isinstance(self.match_bonus, int)
            or isinstance(self.match_bonus, bool)
            or self.match_bonus < 0
        ):
            raise ValueError("match_bonus must be a non-negative integer")

        expected_keys = set(SUITS) | {MODE_OBEABE, MODE_UNEUFE}
        overrides = dict(self.contract_factors)
        extra = sorted(set(overrides) - expected_keys)
        if extra:
            raise ValueError(f"invalid contract_factors: unknown {extra}")
        factors = dict(STANDARD_CONTRACT_FACTORS)
        factors.update(overrides)
        if any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
            for value in factors.values()
        ):
            raise ValueError("contract factors must be positive integers")

        object.__setattr__(self, "contract_factors", MappingProxyType(factors))

    def contract_factor(self, mode: str, trump_suit: str | None = None) -> int:
        """Return the configured multiplier and validate the contract."""

        if mode == MODE_TRUMP:
            if trump_suit not in SUITS:
                raise ValueError(
                    "trump_suit is required and must be a valid suit for trump mode"
                )
            return self.contract_factors[trump_suit]
        if mode in (MODE_OBEABE, MODE_UNEUFE):
            if trump_suit is not None:
                raise ValueError("trump_suit must be None for non-trump modes")
            return self.contract_factors[mode]
        raise ValueError(f"unknown mode: {mode}")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe representation for replays and run manifests."""

        legal_moves = asdict(self.legal_moves)
        # Preserve the serialized shape of existing standard-profile manifests.
        # The stricter edge case is written only by profiles that opt into it.
        if not legal_moves["must_overtrump_when_only_trumps"]:
            del legal_moves["must_overtrump_when_only_trumps"]
        return {
            "version": self.version,
            "allow_stock": self.allow_stock,
            "allow_weis": self.allow_weis,
            "legal_moves": legal_moves,
            "contract_factors": dict(self.contract_factors),
            "match_bonus": self.match_bonus,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> RulesetConfig:
        """Reconstruct a profile previously produced by :meth:`to_dict`."""

        legal_payload = payload.get("legal_moves", {})
        if not isinstance(legal_payload, Mapping):
            raise ValueError("rules profile legal_moves must be a mapping")
        factor_payload = payload.get("contract_factors", {})
        if not isinstance(factor_payload, Mapping):
            raise ValueError("rules profile contract_factors must be a mapping")
        allow_stock = payload.get("allow_stock", True)
        allow_weis = payload.get("allow_weis", True)
        if not isinstance(allow_stock, bool) or not isinstance(allow_weis, bool):
            raise ValueError("rules profile allow_stock and allow_weis must be booleans")
        return cls(
            version=str(payload.get("version", STANDARD_PROFILE_VERSION)),
            allow_stock=allow_stock,
            allow_weis=allow_weis,
            legal_moves=RuleSet(**dict(legal_payload)),
            contract_factors={str(key): value for key, value in factor_payload.items()},
            match_bonus=payload.get("match_bonus", 100),
        )


def all_contracts_x1_profile(*, match_bonus: int = 100) -> RulesetConfig:
    """Return a profile with every contract scored at face value."""

    return RulesetConfig(
        version=ALL_CONTRACTS_X1_PROFILE_VERSION,
        contract_factors=ALL_CONTRACTS_X1_FACTORS,
        match_bonus=match_bonus,
    )


STANDARD_RULES_PROFILE = RulesetConfig()
ALL_CONTRACTS_X1_PROFILE = all_contracts_x1_profile()
# Exact pre-profile trick scoring for reproducible legacy experiments.
LEGACY_UNMULTIPLIED_PROFILE = RulesetConfig(
    version=LEGACY_UNMULTIPLIED_PROFILE_VERSION,
    contract_factors=ALL_CONTRACTS_X1_FACTORS,
    match_bonus=0,
)
