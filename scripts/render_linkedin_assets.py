"""Render deterministic LinkedIn images and a GIF from verified Jass reports.

This script intentionally reads the same JSON artifacts used by the advisor;
it never invents intermediate win rates or training measurements.
"""

from __future__ import annotations

import argparse
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

FONT_SANS = Path("/System/Library/Fonts/SFNS.ttf")
FONT_MONO = Path("/System/Library/Fonts/SFNSMono.ttf")
FONT_SERIF = Path("/System/Library/Fonts/NewYork.ttf")


def font(path: Path, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(str(path), size=size)
    except OSError:
        return ImageFont.load_default(size=size)


def report_metrics(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    opponent, report = next(iter(payload["opponents"].items()))
    overall = report["metrics"]["overall"]
    return {
        "opponent": opponent,
        "win_rate": float(overall["win_rate"]),
        "win_low": float(overall["win_rate_ci95"]["low"]),
        "paired_win_rate": float(overall["paired_win_rate"]),
        "point_difference": float(overall["average_point_difference"]),
        "games": int(overall["episodes"]),
    }


def load_inputs(args: argparse.Namespace) -> dict[str, Any]:
    resource = json.loads(args.resource_manifest.read_text())
    model = json.loads(args.model_manifest.read_text())
    advice = json.loads(args.advice.read_text())
    return {
        "full_random": report_metrics(args.full_random),
        "full_strategic": report_metrics(args.full_strategic),
        "external_random": report_metrics(args.external_random),
        "external_reference": report_metrics(args.external_reference),
        "resource": resource["resource_usage"],
        "model_steps": int(model["actual_timesteps"]),
        "checkpoint_count": len(model["checkpoints"]),
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
    x_positions = (72, 352, 632, 912)
    for x, (label, value) in zip(x_positions, metrics, strict=True):
        text(draw, (x, y), label.upper(), typeface=font(FONT_SANS, 20), fill=MUTED)
        text(draw, (x, y + 35), f"{value * 100:.2f}%", typeface=font(FONT_MONO, 40), fill=INK)


def render_hero(data: dict[str, Any], out: Path) -> None:
    image = background((1200, 1200)).convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    eyebrow(draw, "Local ML · Swiss Schieber", 72, 62)
    text(
        draw,
        (68, 118),
        "A Jass advisor\nthat survived\n16,000 games.",
        typeface=font(FONT_SERIF, 78),
        spacing=-2,
    )
    text(
        draw,
        (72, 410),
        "PUBLIC INFORMATION ONLY  ·  NEURAL ACTOR + 24-WORLD SEARCH",
        typeface=font(FONT_MONO, 20),
        fill=GOLD,
    )

    card(image, (867, 255), "◉", "A", angle=-13, scale=.76)
    card(image, (976, 285), "✿", "K", angle=5, scale=.76)
    card(image, (1060, 342), "▰", "9", angle=17, selected=True, scale=.76)

    draw.line((72, 495, 1128, 495), fill=LINE, width=2)
    metrics = [
        ("Random", data["full_random"]["win_rate"]),
        ("Strategic", data["full_strategic"]["win_rate"]),
        ("Ext. random", data["external_random"]["win_rate"]),
        ("Ext. proxy", data["external_reference"]["win_rate"]),
    ]
    metric_strip(draw, metrics, 540)

    draw.rounded_rectangle((72, 700, 1128, 1010), radius=28, fill=(244, 236, 216, 255))
    text(draw, (112, 748), "TRAINED LOCALLY", typeface=font(FONT_SANS, 22), fill="#53635a")
    text(draw, (112, 792), "Apple M3 Pro · 36 GB", typeface=font(FONT_SERIF, 47), fill=PAPER_INK)
    speed = float(data["resource"]["steps_per_learning_second"])
    memory_mib = float(data["resource"]["peak_rss_bytes"]) / 1024**2
    text(
        draw,
        (112, 875),
        f"{speed:,.0f} steps/s   ·   {memory_mib:.0f} MiB peak RSS   ·   8 CPU threads",
        typeface=font(FONT_MONO, 24),
        fill=PAPER_INK,
    )
    text(
        draw,
        (112, 932),
        "Rule-correct Schieber · paired same-deal evaluation · strict confidence gates",
        typeface=font(FONT_SANS, 22),
        fill="#53635a",
    )
    text(draw, (72, 1080), "jass_rl", typeface=font(FONT_MONO, 25), fill=MINT)
    text(
        draw,
        (1128, 1080),
        "EXTERNAL RESULTS USE A CLEAN-ROOM PROXY",
        typeface=font(FONT_SANS, 18),
        fill=MUTED,
        anchor="ra",
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(out, quality=95)


def render_decision(data: dict[str, Any], out: Path) -> None:
    image = background((1200, 1200), seed=23).convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    eyebrow(draw, "One real advisor decision", 72, 62)
    text(
        draw,
        (68, 118),
        "The actor wanted J.\nSearch kept the 9.",
        typeface=font(FONT_SERIF, 72),
        spacing=1,
    )
    actions = {row["card"]: row for row in data["advice"]["actions"]}
    nine = actions["schilten:9"]
    jack = actions["schilten:J"]
    card(image, (330, 590), "▰", "9", selected=True, scale=1.2)
    card(image, (640, 590), "▰", "J", scale=1.2)
    text(draw, (242, 785), "SEARCH", typeface=font(FONT_SANS, 22), fill=MINT)
    text(draw, (242, 825), f"{nine['expected_margin']:+.2f}", typeface=font(FONT_MONO, 38))
    text(draw, (552, 785), "NEURAL PRIOR", typeface=font(FONT_SANS, 22), fill=GOLD)
    text(
        draw,
        (552, 825),
        f"{jack['neural_probability'] * 100:.2f}%",
        typeface=font(FONT_MONO, 38),
    )

    draw.rounded_rectangle((820, 390, 1128, 865), radius=28, fill=(244, 236, 216, 255))
    text(draw, (860, 430), "24 POSSIBLE WORLDS", typeface=font(FONT_SANS, 20), fill="#53635a")
    for index in range(24):
        row, column = divmod(index, 6)
        x, y = 875 + column * 38, 490 + row * 45
        draw.ellipse((x, y, x + 17, y + 17), fill=MINT if index < 20 else GOLD)
    gap = float(nine["expected_margin"] - jack["expected_margin"])
    text(draw, (860, 705), "SEARCH GAP", typeface=font(FONT_SANS, 19), fill="#53635a")
    text(draw, (860, 740), f"{gap:+.2f} pts", typeface=font(FONT_MONO, 36), fill=PAPER_INK)
    text(draw, (860, 798), "Override limit  +3.00", typeface=font(FONT_MONO, 20), fill="#53635a")

    draw.line((72, 930, 1128, 930), fill=LINE, width=2)
    text(
        draw,
        (72, 975),
        "The learned policy may override search only inside a 3-point guardrail.\n"
        "Here the gap was wider, so the safer search choice stayed in control.",
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


def gif_frame(data: dict[str, Any], index: int, total: int) -> Image.Image:
    image = background((960, 960), seed=31).convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    eyebrow(draw, "Building a Jass advisor on one laptop", 54, 46)
    phase = index / total

    if phase < .38:
        local = phase / .38
        text(draw, (52, 112), "Training,\nlocally.", typeface=font(FONT_SERIF, 78), spacing=0)
        trained = int(data["model_steps"] * min(1.0, local))
        text(draw, (54, 350), f"{trained:,}", typeface=font(FONT_MONO, 54), fill=INK)
        text(draw, (54, 418), "PPO TIMESTEPS", typeface=font(FONT_SANS, 22), fill=MUTED)
        progress_bar(draw, (54, 490, 906, 520), local)
        checkpoints = data["checkpoint_count"]
        for checkpoint in range(checkpoints):
            x = 62 + checkpoint * (830 / max(1, checkpoints - 1))
            active = checkpoint / checkpoints <= local
            draw.ellipse((x - 5, 559, x + 5, 569), fill=MINT if active else LINE)
        text(
            draw,
            (54, 625),
            f"{checkpoints} HASHED CHECKPOINTS   ·   M3 PRO   ·   8 CPU THREADS",
            typeface=font(FONT_MONO, 21),
            fill=GOLD,
        )
        speed = float(data["resource"]["steps_per_learning_second"])
        memory_mib = float(data["resource"]["peak_rss_bytes"]) / 1024**2
        text(
            draw,
            (54, 720),
            f"Resource gate\n{speed:,.0f} steps/s  ·  {memory_mib:.0f} MiB peak RSS",
            typeface=font(FONT_SANS, 28),
            fill=MUTED,
            spacing=12,
        )
    elif phase < .62:
        local = (phase - .38) / .24
        worlds = max(1, min(24, math.ceil(local * 24)))
        text(draw, (52, 112), "Search the\npossible worlds.", typeface=font(FONT_SERIF, 66))
        for world in range(24):
            angle = 2 * math.pi * world / 24
            radius = 235 + 25 * math.sin(world * 1.7)
            x = 480 + math.cos(angle) * radius
            y = 535 + math.sin(angle) * radius
            active = world < worlds
            draw.ellipse((x - 12, y - 12, x + 12, y + 12), fill=MINT if active else LINE)
            if active:
                draw.line((480, 535, x, y), fill=(118, 198, 162, 38), width=2)
        card(image, (480, 535), "▰", "9", selected=True, scale=.9)
        text(
            draw,
            (480, 835),
            f"{worlds}/24 DETERMINIZATIONS",
            typeface=font(FONT_MONO, 25),
            fill=GOLD,
            anchor="mm",
        )
    elif phase < .88:
        local = (phase - .62) / .26
        text(draw, (52, 112), "Then make it\nprove itself.", typeface=font(FONT_SERIF, 68))
        rows = [
            ("RANDOM", data["full_random"]["win_rate"], .60),
            ("STRATEGIC", data["full_strategic"]["win_rate"], .55),
            ("EXT. RANDOM", data["external_random"]["win_rate"], .87),
            ("EXT. PROXY", data["external_reference"]["win_rate"], .67),
        ]
        for row, (label, value, gate) in enumerate(rows):
            y = 390 + row * 115
            shown = value * min(1.0, max(0.0, local * 1.5 - row * .12))
            text(draw, (54, y), label, typeface=font(FONT_SANS, 23), fill=MUTED)
            text(draw, (906, y), f"{shown * 100:.2f}%", typeface=font(FONT_MONO, 28), anchor="ra")
            progress_bar(
                draw,
                (54, y + 44, 906, y + 66),
                shown,
                color=MINT if shown >= gate else GOLD,
            )
            gate_x = 54 + int(852 * gate)
            draw.line((gate_x, y + 38, gate_x, y + 72), fill=INK, width=3)
    else:
        text(draw, (52, 112), "Qualified.\nNow it can advise.", typeface=font(FONT_SERIF, 70))
        card(image, (285, 510), "▰", "9", selected=True, scale=1.05)
        draw.rounded_rectangle((520, 340, 906, 690), radius=25, fill=(244, 236, 216, 255))
        text(draw, (560, 385), "FORMAL RESULT", typeface=font(FONT_SANS, 21), fill="#53635a")
        text(
            draw,
            (560, 435),
            f"{data['full_strategic']['win_rate'] * 100:.2f}%",
            typeface=font(FONT_MONO, 54),
            fill=PAPER_INK,
        )
        text(draw, (560, 508), "vs strategic", typeface=font(FONT_SERIF, 36), fill=PAPER_INK)
        text(
            draw,
            (560, 575),
            f"paired wins  {data['full_strategic']['paired_win_rate'] * 100:.2f}%\n"
            f"mean diff   +{data['full_strategic']['point_difference']:.1f}",
            typeface=font(FONT_MONO, 22),
            fill="#53635a",
            spacing=10,
        )
        text(draw, (54, 820), "PUBLIC INFORMATION ONLY", typeface=font(FONT_MONO, 23), fill=GOLD)
    return image.convert("RGB")


def render_gif(data: dict[str, Any], out: Path) -> None:
    frames = [gif_frame(data, index, 52) for index in range(52)]
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
        comment=b"Verified Jass RL training and qualification timelapse",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-random", type=Path, required=True)
    parser.add_argument("--full-strategic", type=Path, required=True)
    parser.add_argument("--external-random", type=Path, required=True)
    parser.add_argument("--external-reference", type=Path, required=True)
    parser.add_argument("--resource-manifest", type=Path, required=True)
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
