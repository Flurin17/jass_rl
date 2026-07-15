# Schieber rules profile

The default `schieber-pagat-v1` profile follows the standard four-player
Schieber rules described by
[Pagat](https://www.pagat.com/jass/schieber.html) (consulted 2026-07-15).

## Legal play

- In Obeabe and Uneufe, a player follows the led suit when possible and may
  discard any card when void.
- In a trump contract, a player may either follow a non-trump lead or play a
  trump. When void in the led suit, the player may trump or discard.
- After a non-trump lead has already been ruffed, a lower trump is illegal
  unless the player's hand contains only trumps. Following the led suit or
  playing a higher trump remains legal; a void player may instead discard.
- When trump is led, a player follows trump but does not have to overtrump.
- A player whose only trump is the Puur (trump Jack/Under) may retain it and
  discard another card when trump is led.

`core.legal_moves.RuleSet` exposes house-rule switches while using these rules
as its defaults.

The matched external `verardo-v1` profile deliberately pins one stricter engine
edge case: when a non-trump lead has been ruffed and a player's remaining hand
contains only trumps, an available overtrump is mandatory. Undertrumping is
allowed only when none of those trumps can beat the current winning trump. This
does not change the default `schieber-pagat-v1` profile.

## Scoring

Card and last-trick points remain raw in `card_points` and `trick_points`.
`RulesetConfig.contract_factor()` supplies the hand multiplier:

| Contract | Factor |
| --- | ---: |
| Eicheln | 1 |
| Rosen | 1 |
| Schilten | 2 |
| Schellen | 2 |
| Obeabe | 3 |
| Uneufe | 4 |

Winning all nine tricks adds 100 Match points before multiplication, so a Match
scores `257 * factor`. `all_contracts_x1_profile()` retains Match with factor
one for every contract. `LEGACY_UNMULTIPLIED_PROFILE` additionally disables
Match for exact pre-profile experiments.
