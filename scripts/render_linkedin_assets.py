"""Render deterministic LinkedIn images and a GIF from verified Jass reports.

This script intentionally reads the same JSON artifacts used by the advisor;
it never invents intermediate win rates or training measurements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

FELT = "#071d18"
FELT_LIGHT = "#0d3027"
INK = "#f3ecd9"
MUTED = "#aeb9ad"
MINT = "#76c6a2"
CORAL = "#eb5a46"
GOLD = "#e5b84a"
PAPER = "#f4ecd8"
PAPER_INK = "#17241e"
LINE = "#315047"

TARGET_GIF_BYTES = 5 * 1024 * 1024
MAX_GIF_FRAMES = 500
MAX_GIF_ANIMATED_PIXELS = 36_152_320

FONT_SANS = Path("/System/Library/Fonts/SFNS.ttf")
FONT_MONO = Path("/System/Library/Fonts/SFNSMono.ttf")
FONT_SERIF = Path("/System/Library/Fonts/NewYork.ttf")


def font(path: Path, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(str(path), size=size)
    except OSError:
        return ImageFont.load_default(size=size)


def _implementation_digest(files: list[str]) -> str:
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for relative in files:
        path = root / relative
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def report_metrics(path: Path, *, expected_opponent: str) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("report_version") != 2:
        raise ValueError(f"{path}: expected report_version 2")
    provenance = payload.get("evaluation_provenance", {}).get("git", {})
    if provenance.get("dirty") is not False:
        raise ValueError(f"{path}: report must come from a clean Git tree")
    if payload.get("qualification", {}).get("all_requested") is not True:
        raise ValueError(f"{path}: merged qualification did not pass")
    opponents = payload.get("opponents")
    if not isinstance(opponents, dict) or set(opponents) != {expected_opponent}:
        raise ValueError(f"{path}: expected only opponent {expected_opponent!r}")
    report = opponents[expected_opponent]
    if report.get("qualification", {}).get("qualified") is not True:
        raise ValueError(f"{path}: opponent qualification did not pass")
    candidate = payload.get("candidate_policy", {})
    if candidate.get("type") != "NeuralGuidedPIMCPolicy":
        raise ValueError(f"{path}: unexpected candidate policy")
    implementation = candidate.get("implementation")
    if not isinstance(implementation, dict):
        raise ValueError(f"{path}: policy implementation identity is missing")
    files = implementation.get("files")
    if not isinstance(files, list) or not all(isinstance(item, str) for item in files):
        raise ValueError(f"{path}: policy implementation files are invalid")
    if implementation.get("sha256") != _implementation_digest(files):
        raise ValueError(f"{path}: report policy no longer matches this checkout")
    overall = report["metrics"]["overall"]
    return {
        "opponent": expected_opponent,
        "win_rate": float(overall["win_rate"]),
        "win_low": float(overall["win_rate_ci95"]["low"]),
        "paired_win_rate": float(overall["paired_win_rate"]),
        "point_difference": float(overall["average_point_difference"]),
        "games": int(overall["episodes"]),
        "model_sha256": payload["model"]["sha256"],
        "model_timesteps": int(payload["model"]["num_timesteps"]),
        "manifest_sha256": payload["run"]["manifest_sha256"],
        "implementation": implementation,
    }


def load_inputs(args: argparse.Namespace) -> dict[str, Any]:
    model_bytes = args.model_manifest.read_bytes()
    model = json.loads(model_bytes)
    advice = json.loads(args.advice.read_text())
    full_random = report_metrics(args.full_random, expected_opponent="random")
    full_strategic = report_metrics(args.full_strategic, expected_opponent="strategic")
    if full_random["model_sha256"] != full_strategic["model_sha256"]:
        raise ValueError("full-game reports do not evaluate the same model")
    if full_random["implementation"] != full_strategic["implementation"]:
        raise ValueError("full-game reports do not evaluate the same policy implementation")
    model_sha256 = model.get("final_model_sha256")
    if model_sha256 != full_random["model_sha256"]:
        raise ValueError("model manifest does not match the qualified reports")
    model_steps = int(model["actual_timesteps"])
    if {full_random["model_timesteps"], full_strategic["model_timesteps"]} != {
        model_steps
    }:
        raise ValueError("training-step evidence does not match the qualified reports")
    manifest_sha256 = hashlib.sha256(model_bytes).hexdigest()
    if {
        full_random["manifest_sha256"],
        full_strategic["manifest_sha256"],
        advice.get("manifest_sha256"),
    } != {manifest_sha256}:
        raise ValueError("advisor and reports do not share the supplied model manifest")
    if advice.get("model_sha256") != model_sha256:
        raise ValueError("advisor example does not use the qualified model")
    if advice.get("policy_implementation") != full_random["implementation"]:
        raise ValueError("advisor example does not use the qualified policy implementation")
    actions = advice.get("actions")
    if not isinstance(actions, list) or not actions:
        raise ValueError("advisor example has no ranked actions")
    selected = [row for row in actions if row.get("selected") is True]
    if len(selected) != 1 or selected[0].get("card") != advice.get("selected_card"):
        raise ValueError("advisor example has an inconsistent selected card")
    if advice.get("selected_card") != "schilten:9":
        raise ValueError("social example expects the qualified Schilten 9 decision")
    best_search = max(actions, key=lambda row: float(row["expected_margin"]))
    best_instinct = max(actions, key=lambda row: float(row["neural_probability"]))
    if best_search.get("card") != "schilten:9" or best_instinct.get("card") != "schilten:J":
        raise ValueError("social example no longer supports the 9-versus-J explanation")
    if advice.get("successful_determinizations") != 24 or advice.get("used_fallback"):
        raise ValueError("advisor example must complete all 24 hidden-hand samples")
    if any(row.get("determinizations") != 24 for row in actions):
        raise ValueError("advisor example must score every card on all 24 hidden hands")
    test_games = full_random["games"] + full_strategic["games"]
    return {
        "full_random": full_random,
        "full_strategic": full_strategic,
        "model_steps": model_steps,
        "test_games": test_games,
        "advice": advice,
    }


def background(size: tuple[int, int], *, seed: int = 17) -> Image.Image:
    width, height = size
    image = Image.new("RGB", size, FELT)
    draw = ImageDraw.Draw(image, "RGBA")
    for y in range(height):
        blend = y / max(1, height - 1)
        draw.line((0, y, width, y), fill=(7, 29 + int(14 * blend), 24 + int(10 * blend), 255))
    for offset in range(-height, width, 18):
        draw.line((offset, 0, offset + height, height), fill=(255, 255, 255, 5), width=2)
    rng = random.Random(seed)
    for _ in range(180):
        x = rng.randrange(width)
        y = rng.randrange(height)
        radius = rng.choice((1, 1, 1, 2))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(118, 198, 162, 18))
    return image


def text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    value: str,
    *,
    typeface: ImageFont.ImageFont,
    fill: str = INK,
    anchor: str | None = None,
    spacing: int = 8,
    align: str = "left",
) -> None:
    draw.multiline_text(
        xy,
        value,
        font=typeface,
        fill=fill,
        anchor=anchor,
        spacing=spacing,
        align=align,
    )


def eyebrow(draw: ImageDraw.ImageDraw, value: str, x: int, y: int) -> None:
    text(draw, (x, y), value.upper(), typeface=font(FONT_SANS, 25), fill=MINT)


def card(
    canvas: Image.Image,
    center: tuple[int, int],
    suit: str,
    rank: str,
    *,
    angle: float = 0.0,
    selected: bool = False,
    scale: float = 1.0,
) -> None:
    width, height = int(210 * scale), int(292 * scale)
    source = Image.new("RGBA", (width + 30, height + 30), (0, 0, 0, 0))
    draw = ImageDraw.Draw(source, "RGBA")
    shadow = (15, 22, 18, 100)
    draw.rounded_rectangle((17, 19, width + 17, height + 19), radius=18, fill=shadow)
    draw.rounded_rectangle(
        (8, 8, width + 8, height + 8),
        radius=18,
        fill=PAPER,
        outline=CORAL if selected else "#d4cbb5",
        width=7 if selected else 2,
    )
    red = suit in {"◉", "✿"}
    color = CORAL if red else PAPER_INK
    text(draw, (30, 24), rank, typeface=font(FONT_SERIF, int(52 * scale)), fill=color)
    text(
        draw,
        ((width + 16) // 2, (height + 16) // 2),
        suit,
        typeface=font(FONT_SANS, int(94 * scale)),
        fill=color,
        anchor="mm",
    )
    if selected:
        draw.ellipse((width - 43, 23, width + 3, 69), fill=MINT)
        text(draw, (width - 20, 47), "✓", typeface=font(FONT_SANS, 26), fill=FELT, anchor="mm")
    rotated = source.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
    canvas.alpha_composite(
        rotated,
        (center[0] - rotated.width // 2, center[1] - rotated.height // 2),
    )


def metric_strip(draw: ImageDraw.ImageDraw, metrics: list[tuple[str, float]], y: int) -> None:
    column_width = 1056 / len(metrics)
    for index, (label, value) in enumerate(metrics):
        x = int(72 + index * column_width)
        text(draw, (x, y), label.upper(), typeface=font(FONT_SANS, 20), fill=MUTED)
        text(draw, (x, y + 35), f"{value * 100:.2f}%", typeface=font(FONT_MONO, 40), fill=INK)


def render_hero(data: dict[str, Any], out: Path) -> None:
    image = background((1200, 1200)).convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    eyebrow(draw, "AI meets Swiss Jass", 72, 62)
    text(
        draw,
        (68, 118),
        "I taught an AI\nto play Jass.",
        typeface=font(FONT_SERIF, 78),
        spacing=-2,
    )
    text(
        draw,
        (72, 360),
        f"Then I made it prove itself in {data['test_games']:,} full games.",
        typeface=font(FONT_SANS, 26),
        fill=GOLD,
    )

    card(image, (867, 255), "◉", "A", angle=-13, scale=.76)
    card(image, (976, 285), "✿", "K", angle=5, scale=.76)
    card(image, (1060, 342), "▰", "9", angle=17, selected=True, scale=.76)

    draw.line((72, 455, 1128, 455), fill=LINE, width=2)
    metric_strip(
        draw,
        [
            ("Wins vs random play", data["full_random"]["win_rate"]),
            ("Wins vs strategy", data["full_strategic"]["win_rate"]),
        ],
        505,
    )

    draw.rounded_rectangle((72, 675, 1128, 1015), radius=28, fill=(244, 236, 216, 255))
    text(draw, (112, 720), "FROM PRACTICE TO ADVICE", typeface=font(FONT_SANS, 22), fill="#53635a")
    steps = [
        ("1", "LEARN", f"{data['model_steps'] / 1_000_000:.0f} million\npractice decisions"),
        ("2", "THINK", "24 possible\nhidden hands"),
        ("3", "ADVISE", "Suggest the\nnext card"),
    ]
    for index, (number, label, copy) in enumerate(steps):
        x = 112 + index * 340
        draw.ellipse((x, 785, x + 52, 837), fill=MINT)
        text(draw, (x + 26, 811), number, typeface=font(FONT_MONO, 25), fill=FELT, anchor="mm")
        text(draw, (x, 862), label, typeface=font(FONT_SANS, 20), fill="#53635a")
        text(draw, (x, 900), copy, typeface=font(FONT_SERIF, 29), fill=PAPER_INK, spacing=6)
    text(draw, (72, 1080), "jass_rl", typeface=font(FONT_MONO, 25), fill=MINT)
    text(
        draw,
        (1128, 1080),
        "FAIR TEST · BOTH SIDES PLAYED THE SAME CARDS",
        typeface=font(FONT_SANS, 18),
        fill=MUTED,
        anchor="ra",
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(out, quality=95)


def render_decision(_data: dict[str, Any], out: Path) -> None:
    image = background((1200, 1200), seed=23).convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    eyebrow(draw, "One example decision", 72, 62)
    text(
        draw,
        (68, 118),
        "Its first instinct: J.\nIts final choice: 9.",
        typeface=font(FONT_SERIF, 72),
        spacing=1,
    )
    card(image, (330, 590), "▰", "9", selected=True, scale=1.2)
    card(image, (640, 590), "▰", "J", scale=1.2)
    text(draw, (242, 785), "FINAL CHOICE", typeface=font(FONT_SANS, 22), fill=MINT)
    text(draw, (242, 825), "9 of Schilten", typeface=font(FONT_SERIF, 32))
    text(draw, (552, 785), "FIRST INSTINCT", typeface=font(FONT_SANS, 22), fill=GOLD)
    text(draw, (552, 825), "Jack of Schilten", typeface=font(FONT_SERIF, 32))

    draw.rounded_rectangle((820, 390, 1128, 865), radius=28, fill=(244, 236, 216, 255))
    text(draw, (860, 430), "24 POSSIBLE HANDS", typeface=font(FONT_SANS, 20), fill="#53635a")
    for index in range(24):
        row, column = divmod(index, 6)
        x, y = 875 + column * 38, 490 + row * 45
        draw.ellipse((x, y, x + 17, y + 17), fill=MINT)
    text(draw, (860, 705), "WHY THE 9?", typeface=font(FONT_SANS, 19), fill="#53635a")
    text(draw, (860, 744), "It edged ahead", typeface=font(FONT_SERIF, 30), fill=PAPER_INK)
    text(draw, (860, 784), "in this check.", typeface=font(FONT_SERIF, 30), fill=PAPER_INK)

    draw.line((72, 930, 1128, 930), fill=LINE, width=2)
    text(
        draw,
        (72, 975),
        "Most of the cards are hidden in Jass. So before suggesting a play,\n"
        "the advisor checks how each legal card could work across many likely hands.",
        typeface=font(FONT_SANS, 27),
        fill=INK,
        spacing=10,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(out, quality=95)


def progress_bar(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    fraction: float,
    *,
    color: str = MINT,
) -> None:
    draw.rounded_rectangle(box, radius=12, fill=LINE)
    x0, y0, x1, y1 = box
    fill_right = x0 + max(14, int((x1 - x0) * max(0.0, min(1.0, fraction))))
    draw.rounded_rectangle((x0, y0, fill_right, y1), radius=12, fill=color)


def gif_frame(data: dict[str, Any], index: int) -> Image.Image:
    image = background((900, 900), seed=31).convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    eyebrow(draw, "Teaching an AI to play Jass", 50, 44)

    if index <= 9:
        local = index / 9
        text(
            draw,
            (48, 108),
            "First, it learned\nthe game.",
            typeface=font(FONT_SERIF, 70),
            spacing=0,
        )
        trained = int(data["model_steps"] * min(1.0, local))
        text(draw, (50, 345), f"{trained:,}", typeface=font(FONT_MONO, 49), fill=INK)
        text(draw, (50, 408), "PRACTICE DECISIONS", typeface=font(FONT_SANS, 21), fill=MUTED)
        progress_bar(draw, (50, 475, 850, 505), local)
        text(
            draw,
            (50, 575),
            "LEGAL CARDS  ·  SCORING  ·  TEAMWORK",
            typeface=font(FONT_MONO, 20),
            fill=GOLD,
        )
        text(
            draw,
            (50, 675),
            "It practised choosing a card,\nseeing what happened, and trying again.",
            typeface=font(FONT_SANS, 28),
            fill=MUTED,
            spacing=12,
        )
    elif index <= 18:
        local = (index - 10) / 8
        worlds = max(1, min(24, math.ceil(local * 24)))
        text(
            draw,
            (48, 108),
            "Then it learned to\nthink past one hand.",
            typeface=font(FONT_SERIF, 62),
        )
        for world in range(24):
            angle = 2 * math.pi * world / 24
            radius = 220 + 23 * math.sin(world * 1.7)
            x = 450 + math.cos(angle) * radius
            y = 515 + math.sin(angle) * radius
            active = world < worlds
            draw.ellipse((x - 12, y - 12, x + 12, y + 12), fill=MINT if active else LINE)
            if active:
                draw.line((450, 515, x, y), fill=(118, 198, 162, 38), width=2)
        card(image, (450, 515), "▰", "9", selected=True, scale=.9)
        text(
            draw,
            (450, 805),
            f"{worlds}/24 POSSIBLE HIDDEN HANDS CHECKED",
            typeface=font(FONT_MONO, 22),
            fill=GOLD,
            anchor="mm",
        )
    elif index <= 28:
        local = (index - 19) / 9
        text(draw, (48, 108), "Finally, I made it\nprove itself.", typeface=font(FONT_SERIF, 65))
        rows = [
            ("WINS VS RANDOM PLAY", data["full_random"]["win_rate"]),
            ("WINS VS STRATEGY", data["full_strategic"]["win_rate"]),
        ]
        for row, (label, value) in enumerate(rows):
            y = 390 + row * 155
            shown = value * min(1.0, max(0.0, local * 1.35 - row * .18))
            text(draw, (50, y), label, typeface=font(FONT_SANS, 23), fill=MUTED)
            text(draw, (850, y), f"{shown * 100:.2f}%", typeface=font(FONT_MONO, 29), anchor="ra")
            progress_bar(draw, (50, y + 48, 850, y + 74), shown)
        text(
            draw,
            (50, 735),
            f"{data['test_games']:,} FULL GAMES\nFAIR TEST · BOTH SIDES GOT THE SAME CARDS",
            typeface=font(FONT_MONO, 22),
            fill=GOLD,
            spacing=12,
        )
    else:
        text(draw, (48, 108), "Tested.\nReady to advise.", typeface=font(FONT_SERIF, 68))
        card(image, (260, 495), "▰", "9", selected=True, scale=1.02)
        draw.rounded_rectangle((490, 315, 850, 685), radius=25, fill=(244, 236, 216, 255))
        text(
            draw,
            (530, 360),
            f"{data['test_games']:,} TEST GAMES",
            typeface=font(FONT_SANS, 21),
            fill="#53635a",
        )
        text(
            draw,
            (530, 420),
            f"{data['full_strategic']['win_rate'] * 100:.2f}%",
            typeface=font(FONT_MONO, 50),
            fill=PAPER_INK,
        )
        text(draw, (530, 485), "wins vs strategy", typeface=font(FONT_SERIF, 31), fill=PAPER_INK)
        text(
            draw,
            (530, 570),
            f"{data['full_random']['win_rate'] * 100:.2f}% wins\nvs random play",
            typeface=font(FONT_SANS, 25),
            fill="#53635a",
            spacing=10,
        )
        text(draw, (50, 790), "THE NEXT MOVE, EXPLAINED", typeface=font(FONT_MONO, 23), fill=GOLD)
    return image.convert("RGB")


def render_gif(data: dict[str, Any], out: Path) -> None:
    frame_count = 36
    frames = [gif_frame(data, index) for index in range(frame_count)]
    palette_frames = [
        frame.convert("P", palette=Image.Palette.ADAPTIVE, colors=128)
        for frame in frames
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    palette_frames[0].save(
        out,
        save_all=True,
        append_images=palette_frames[1:],
        duration=100,
        loop=0,
        optimize=True,
        disposal=2,
        comment=b"Illustrated Jass AI training and verified evaluation story",
    )
    with Image.open(out) as rendered:
        animated_pixels = rendered.width * rendered.height * rendered.n_frames
        if rendered.n_frames > MAX_GIF_FRAMES:
            raise ValueError("rendered GIF exceeds LinkedIn's frame limit")
        if animated_pixels > MAX_GIF_ANIMATED_PIXELS:
            raise ValueError("rendered GIF exceeds LinkedIn's animated-pixel limit")
    if out.stat().st_size > TARGET_GIF_BYTES:
        raise ValueError("rendered GIF exceeds this project's compact 5 MiB target")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-random", type=Path, required=True)
    parser.add_argument("--full-strategic", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--advice", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/assets/linkedin"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = load_inputs(args)
    render_hero(data, args.output_dir / "jass-rl-linkedin-hero.png")
    render_decision(data, args.output_dir / "jass-rl-linkedin-decision.png")
    render_gif(data, args.output_dir / "jass-rl-training-timelapse.gif")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
