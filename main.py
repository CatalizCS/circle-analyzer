import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Tuple, List
from pathlib import Path
from osrparse import Mod, Replay
from dataclasses import dataclass
from enum import Flag, auto
import os
import inquirer
import json
import sys
import time
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich import print as rprint
from urllib.parse import unquote
import re

# Constants and configurations
VERSION = "1.0.0"
APP_NAME = "Circle Analyzer"
AUTHOR = "catalizcs"
GITHUB = "https://github.com/catalizcs/circle-analyzer"

MENU_CONFIG = {
    "Analysis Tools": {
        "1": {
            "label": "Full Analysis",
            "action": "all",
            "description": "Complete replay analysis including all metrics",
        },
        "2": {
            "label": "Hold Time Analysis",
            "action": "holdtime",
            "description": "Analyze key press duration patterns",
        },
        "3": {
            "label": "Frame Time Analysis",
            "action": "frametime",
            "description": "Analyze frame timing distribution",
        },
        "4": {
            "label": "UR Analysis",
            "action": "ur",
            "description": "Analyze hit timing consistency",
        },
    },
    "Detection Tools": {
        "5": {
            "label": "Cheat Detection [BETA]",
            "action": "cheat",
            "description": "Advanced pattern analysis for suspicious behavior",
        },
    },
    "Utilities": {
        "6": {
            "label": "Drop Mode",
            "action": "drop",
            "description": "Batch process multiple replay files",
        },
        "7": {
            "label": "Settings",
            "action": "settings",
            "description": "Configure application settings",
        },
    },
    "System": {
        "8": {
            "label": "Exit",
            "action": "exit",
            "description": "Close the application",
        },
    },
}


@dataclass
class Settings:
    """Settings class for managing application configuration"""

    output_dir: str = "output"
    output_format: str = "png"

    @classmethod
    def load(cls):
        try:
            with open("settings.json", "r") as f:
                data = json.load(f)
                return cls(**data)
        except FileNotFoundError:
            return cls()

    def save(self):
        with open("settings.json", "w") as f:
            json.dump(self.__dict__, f, indent=2)


class StandardKeys(Flag):
    """OSU! standard key flags for parsing replay data"""

    K1 = auto()
    K2 = auto()


def analyze_replay_holdtimes(
    replay_path: str, settings: Settings
) -> Tuple[Dict[int, int], Dict[int, int], Replay, dict]:
    """
    Analyzes holdtimes from replay file
    Returns:
        - dist1: K1 holdtime distribution
        - dist2: K2 holdtime distribution
        - replay: Replay object
        - stats: Statistics including counts and averages
    """
    replay = Replay.from_path(replay_path)
    offset = 0
    key1 = []
    key2 = []

    for frame in replay.replay_data:
        offset += frame.time_delta
        keys = frame.keys
        key1.append((offset, bool(keys & 5)))
        key2.append((offset, bool(keys & 10)))

    # Process holdtimes
    holdtimes1 = []
    holdtimes2 = []
    pressed_frame = None

    # Process K1
    for frame in key1:
        if pressed_frame is None:
            if frame[1]:  # pressed
                pressed_frame = frame
        elif not frame[1]:  # released
            holdtimes1.append(frame[0] - pressed_frame[0])
            pressed_frame = None

    # Process K2
    pressed_frame = None
    for frame in key2:
        if pressed_frame is None:
            if frame[1]:
                pressed_frame = frame
        elif not frame[1]:
            holdtimes2.append(frame[0] - pressed_frame[0])
            pressed_frame = None

    # Calculate distributions and stats with smarter max_holdtime
    if holdtimes1 or holdtimes2:
        all_holdtimes = holdtimes1 + holdtimes2
        # Calculate 95th percentile
        p95 = int(np.percentile(all_holdtimes, 95))

        # Set max_holdtime between 100-300ms based on data
        if p95 < 100:
            max_holdtime = 100  # minimum threshold
        elif p95 > 300:
            max_holdtime = 300  # maximum threshold
        else:
            max_holdtime = p95  # use 95th percentile if within range
    else:
        max_holdtime = 100  # default if no holdtimes

    dist1 = {}
    dist2 = {}

    for time in holdtimes1:
        if time < max_holdtime:
            dist1[time] = dist1.get(time, 0) + 1

    for time in holdtimes2:
        if time < max_holdtime:
            dist2[time] = dist2.get(time, 0) + 1

    stats = {
        "k1_count": len(holdtimes1),
        "k2_count": len(holdtimes2),
        "k1_avg": round(sum(holdtimes1) / len(holdtimes1), 1) if holdtimes1 else 0,
        "k2_avg": round(sum(holdtimes2) / len(holdtimes2), 1) if holdtimes2 else 0,
        "max_holdtime": max_holdtime,
    }

    return dist1, dist2, replay, stats


def plot_holdtime_distribution(
    dist1: Dict[int, int],
    dist2: Dict[int, int],
    replay,
    stats: dict,
    beatmap_name: str = "Unknown",
):
    """
    Creates visualization of holdtime distributions
    Key features:
    - Automatic region detection and splitting
    - Density-based visualization
    - Metadata display including player, map, and mod info
    """
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 6))

    # Sort distributions and combine for full range
    all_times = set(dist1.keys()) | set(dist2.keys())
    min_time = min(all_times) if all_times else 0
    max_time = max(all_times) if all_times else 160

    # Create evenly spaced bins
    bins = np.arange(min_time - 3, max_time + 6, 3)

    # Convert dictionary data to arrays with zeros for missing values
    times1 = np.zeros(len(bins) - 1)
    times2 = np.zeros(len(bins) - 1)

    for i in range(len(bins) - 1):
        for t in range(int(bins[i]), int(bins[i + 1])):
            times1[i] += dist1.get(t, 0)
            times2[i] += dist2.get(t, 0)

    # Add automatic splitting logic
    max_x_values_per_subplot = 30  # Maximum number of x ticks per subplot
    total_range = max_time - min_time
    num_splits = max(1, int(np.ceil(total_range / max_x_values_per_subplot)))

    # Create split regions and filter empty ones
    split_regions = []
    split_size = total_range / num_splits
    for i in range(num_splits):
        start = min_time + i * split_size
        end = start + split_size

        # Check if region has any data
        region_has_data = False
        for k in dist1.keys():
            if start <= k < end and dist1[k] > 0:
                region_has_data = True
                break
        for k in dist2.keys():
            if start <= k < end and dist2[k] > 0:
                region_has_data = True
                break

        if region_has_data:
            split_regions.append([start, end])

    # Skip plotting if no regions have data
    if not split_regions:
        split_regions = [[min_time, max_time]]

    # Create subplots only for regions with data
    fig, axes = plt.subplots(
        len(split_regions), 1, figsize=(12, 3 * len(split_regions))
    )
    if len(split_regions) == 1:
        axes = [axes]

    for ax, (start, end) in zip(axes, split_regions):
        # Filter data for this region
        region_dist1 = {k: v for k, v in dist1.items() if start <= k < end}
        region_dist2 = {k: v for k, v in dist2.items() if start <= k < end}

        # Skip empty regions
        if not region_dist1 and not region_dist2:
            continue

        # Create region bins
        region_bins = np.arange(start - 3, end + 6, 3)

        # Convert to arrays
        region_times1 = np.zeros(len(region_bins) - 1)
        region_times2 = np.zeros(len(region_bins) - 1)

        for i in range(len(region_bins) - 1):
            for t in range(int(region_bins[i]), int(region_bins[i + 1])):
                region_times1[i] += region_dist1.get(t, 0)
                region_times2[i] += region_dist2.get(t, 0)

        # Plot bars for this region
        ax.bar(
            region_bins[:-1],
            region_times1,
            width=2,
            align="edge",
            color="#9999FF",
            alpha=0.8,
        )

        # Configure subplot
        ax.set_xlim(region_bins[0] - 1, region_bins[-1] + 1)

        # Fix y-axis limit warning by ensuring different min/max values
        if len(region_times1) > 0:
            max_y = max(region_times1)
            if max_y == 0:
                max_y = 1  # Set minimum height if all values are zero
            ax.set_ylim(
                0, max_y * 1.1 + 0.1
            )  # Add small offset to prevent identical limits
        else:
            ax.set_ylim(0, 1)  # Default range when no data

        # Add region label
        ax.text(
            0.02,
            0.95,
            f"Range: {int(start)}ms - {int(end)}ms",
            transform=ax.transAxes,
            fontsize=8,
            color="gray",
            alpha=0.8,
        )

        # Smarter y-axis ticks
        if max_y > 20:
            step = max(1, int(max_y / 8))  # Show around 8 ticks
            ax.yaxis.set_major_locator(plt.MultipleLocator(base=step))
        else:
            ax.yaxis.set_major_locator(plt.MultipleLocator(base=1.0))

        # Format y-axis to show only integers
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: str(int(x)) if x.is_integer() else "")
        )

        # X-axis ticks with reduced density
        x_tick_step = max(3, int((end - start) / 20))  # Ensure readable x-axis
        x_ticks = np.arange(int(region_bins[0]), int(region_bins[-1] + 1), x_tick_step)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([str(int(x)) for x in x_ticks], rotation=0)
        ax.tick_params(axis="both", which="major", labelsize=8)

        # Styling
        ax.grid(True, axis="both", linestyle="--", alpha=0.2, color="gray")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.set_facecolor("black")

    # Get mods string
    mods = []
    for mod in Mod:
        if replay.mods & mod.value:
            mod_name = mod.name
            if mod_name == "NC" and "DT" in mods:
                mods.remove("DT")
            mods.append(mod_name)
    mods_str = "+" + ",".join(mods) if mods else "NoMod"

    # Format timestamp
    play_timestamp = replay.timestamp.strftime("%Y-%m-%d %H:%M:%S")

    # Update title with split information if applicable
    title_text = (
        f"Circle | Holdtime distribution {'(Split View)' if len(split_regions) > 1 else ''}\n"
        f"Player: {replay.username}\n"
        f"Beatmap ID: {replay.beatmap_hash[:8]}... | Score ID: {replay.replay_hash[:8]}...\n"
        f"Mods: {mods_str} | Played on: {play_timestamp}\n"
        f"K1: {stats['k1_count']} ({stats['k1_avg']}ms avg) "
        f"K2: {stats['k2_count']} ({stats['k2_avg']}ms avg)"
    )
    axes[0].set_title(title_text, pad=20, loc="left", fontsize=10, fontweight="normal")

    # Layout adjustments
    fig.set_figheight(3 * len(split_regions) + 1)
    fig.patch.set_facecolor("black")
    plt.tight_layout()

    return fig


def find_replay_files(directory: str) -> List[str]:
    """Find all .osr files in the given directory"""
    replay_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".osr"):
                replay_files.append(os.path.join(root, file))
    return replay_files


def select_replay() -> str:
    """Show interactive prompt to select replay file"""
    current_dir = os.getcwd()
    replay_files = find_replay_files(current_dir)

    if not replay_files:
        raise ValueError(f"No replay files found in {current_dir}")

    questions = [
        inquirer.List("replay", message="Select a replay file", choices=replay_files)
    ]
    answers = inquirer.prompt(questions)
    return answers["replay"]


def configure_settings() -> Settings:
    """Configure analysis settings through interactive prompt"""
    questions = [
        inquirer.Text("output_dir", message="Output directory", default="output"),
        inquirer.List(
            "output_format",
            message="Output format",
            choices=["png", "jpg", "pdf"],
            default="png",
        ),
    ]

    answers = inquirer.prompt(questions)
    return Settings(
        output_dir=answers["output_dir"],
        output_format=answers["output_format"],
    )


def show_welcome():
    console = Console()
    console.print(
        Panel.fit(
            f"[bold]Welcome to {APP_NAME} v{VERSION}[/bold]\n"
            f"Author: {AUTHOR}\n"
            f"GitHub: {GITHUB}\n"
        )
    )


def clean_file_path(file_path: str) -> str:
    """Clean and normalize file path"""
    path = file_path.strip()
    path = re.sub(r"^&\s*", "", path)

    if (path.startswith('"') and path.endswith('"')) or (
        path.startswith("'") and path.endswith("'")
    ):
        path = path[1:-1]

    path = path.replace("''", "'")
    path = unquote(path)

    if os.name == "nt":
        path = path.replace("/", "\\")

    path = re.sub(r"^file:[/\\]*", "", path)
    path = re.sub(r"[\<\>\"\|\?\*]", "", path)

    return os.path.normpath(path.strip())


def analyze_frame_times(replay_data) -> Dict[str, any]:
    """Analyzes frame timing distribution from replay data"""
    frame_times = []
    last_time = 0

    # Calculate real frame times between frames
    for frame in replay_data:
        current_time = frame.time_delta
        if current_time > 0:  # Skip zero deltas
            frame_time = current_time - last_time
            if frame_time > 0:  # Only include positive frame times
                frame_times.append(frame_time)
        last_time = current_time

    if not frame_times:
        return {"avg": 0, "min": 0, "max": 0, "distribution": {}, "unstable_rate": 0}

    # Calculate statistics with lower precision for better timewarp detection
    avg_frame = np.mean(frame_times)
    min_frame = min(frame_times)
    max_frame = max(frame_times)
    unstable_rate = np.std(frame_times) * 10

    # Create distribution with 0.1ms precision
    frame_dist = {}
    for time in frame_times:
        rounded = round(time, 1)  # Round to 0.1ms for more precise analysis
        frame_dist[rounded] = frame_dist.get(rounded, 0) + 1

    return {
        "avg": round(avg_frame, 3),  # More precise average
        "min": round(min_frame, 3),
        "max": round(max_frame, 3),
        "distribution": frame_dist,
        "unstable_rate": round(unstable_rate, 3),
        "frame_count": len(frame_times),
    }


def plot_frame_distribution(frame_stats: Dict[str, any], replay) -> plt.Figure:
    """Creates visualization of frame time distribution"""
    plt.style.use("dark_background")

    # Calculate data range and binning
    times = sorted(frame_stats["distribution"].keys())
    time_range = max(times) - min(times)

    # Create evenly spaced bins centered around typical frame time
    typical_frame_time = 16.66
    min_time = max(0, min(min(times), typical_frame_time - 10))
    max_time = max(max(times), typical_frame_time + 10)
    bins = np.arange(
        min_time - 3, max_time + 6, 1
    )  # Smaller 1ms bins for better precision

    # Convert dictionary data to arrays
    values = np.zeros(len(bins) - 1)
    for i in range(len(bins) - 1):
        for t in frame_stats["distribution"].keys():
            if bins[i] <= t < bins[i + 1]:
                values[i] += frame_stats["distribution"][t]

    # Focus on the most relevant region
    active_regions = []
    main_region_start = max(0, int((typical_frame_time - 10) / 1))
    main_region_end = min(len(bins) - 1, int((typical_frame_time + 10) / 1))
    active_regions.append([main_region_start, main_region_end])

    # Add additional regions if there are significant values outside main region
    threshold = max(values) * 0.1  # 10% of max value
    for i in range(len(bins) - 1):
        if values[i] > threshold and (
            i < main_region_start - 5 or i > main_region_end + 5
        ):
            region_start = max(0, i - 5)
            region_end = min(len(bins) - 1, i + 5)
            active_regions.append([region_start, region_end])

    # Merge overlapping regions
    active_regions.sort(key=lambda x: x[0])
    merged_regions = []
    if active_regions:
        current = active_regions[0]
        for region in active_regions[1:]:
            if region[0] <= current[1] + 3:  # Allow small gaps
                current[1] = max(current[1], region[1])
            else:
                merged_regions.append(current)
                current = region
        merged_regions.append(current)

    # Plot data in regions
    fig, axes = plt.subplots(
        len(merged_regions), 1, figsize=(12, 3 * len(merged_regions))
    )
    if len(merged_regions) == 1:
        axes = [axes]

    for ax, (start, end) in zip(axes, merged_regions):
        region_bins = bins[start : end + 2]
        region_values = values[start : end + 1]

        # Plot reference lines with larger font and more visible labels
        ax.axvline(
            x=typical_frame_time,
            color="#888888",  # Brighter color
            linestyle="--",
            alpha=0.7,  # More visible
            linewidth=1.5,  # Thicker line
            label="60 FPS (16.66ms)",
        )
        ax.axvline(
            x=typical_frame_time / 2,
            color="#666666",  # Brighter color
            linestyle=":",
            alpha=0.5,  # More visible
            linewidth=1.5,  # Thicker line
            label="120 FPS (8.33ms)",
        )

        # Plot bars
        ax.bar(
            region_bins[:-1],
            region_values,
            width=0.8,  # Thinner bars for better visibility
            align="edge",
            color="#FF99FF",
            alpha=0.8,
        )

        # Configure subplot
        ax.set_xlim(region_bins[0] - 1, region_bins[-1] + 1)
        ax.set_ylim(0, max(values) * 1.1)

        # Improved axis formatting with larger text
        ax.yaxis.set_major_locator(plt.MaxNLocator(6))  # Fewer ticks for clarity
        ax.xaxis.set_major_locator(plt.MultipleLocator(2))
        ax.tick_params(axis="both", which="major", labelsize=8)  # Larger tick labels
        ax.grid(True, axis="both", linestyle="--", alpha=0.2)  # More visible grid

        # Add legend with improved visibility
        if max(region_values) > 0:
            legend_loc = (
                "upper left" if region_bins[0] > typical_frame_time else "upper right"
            )
            ax.legend(fontsize=8, loc=legend_loc, framealpha=0.8)  # More visible legend

        # Styling
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.set_facecolor("black")

    # Update title and metadata
    mods_str = (
        "+" + ",".join([mod.name for mod in Mod if replay.mods & mod.value])
        if replay.mods
        else "NoMod"
    )
    play_timestamp = replay.timestamp.strftime("%Y-%m-%d %H:%M:%S")

    title_text = (
        f"Circle | Frame Time Analysis\n"
        f"Player: {replay.username} | {mods_str}\n"
        f"Beatmap ID: {replay.beatmap_hash[:8]}... | Score ID: {replay.replay_hash[:8]}...\n"
        f"Average: {frame_stats['avg']}ms | Min: {frame_stats['min']}ms | "
        f"Max: {frame_stats['max']}ms | UR: {frame_stats['unstable_rate']}\n"
        f"Frame Time Distribution (frames: {frame_stats['frame_count']})\n"
        f"Played on: {play_timestamp}"
    )
    axes[0].set_title(title_text, pad=20, loc="left", fontsize=11, fontweight="normal")

    # Add detailed explanation with improved visibility
    fig.text(
        0.01,
        0.01,
        "Frame Time Analysis:\n"
        "• Typical frame time is 16.66ms (60 FPS)\n"
        "• Lower times may indicate timewarp (but aren't definitive proof)\n"
        "• Higher times indicate frame drops/lag",
        color="#AAAAAA",  # Brighter color for better visibility
        fontsize=9,  # Larger text
        alpha=0.9,  # More visible
        bbox=dict(
            facecolor="black", edgecolor="none", alpha=0.7, pad=5
        ),  # Add background
    )

    # Adjust spacing and layout
    fig.set_figheight(3 * len(merged_regions) + 1.7)  # More space for text
    plt.tight_layout()
    # Add extra bottom margin for explanation text
    plt.subplots_adjust(bottom=0.15)

    return fig


def detect_cheats(
    replay_data,
    holdtimes1: List[float],
    holdtimes2: List[float],
    frame_stats: Dict[str, any],
) -> Dict[str, any]:
    """Analyze replay for potential cheating patterns with enhanced detection (BETA)"""
    suspicious = {}
    detailed_stats = {}

    if isinstance(replay_data, str):
        replay = Replay.from_path(replay_data)
        replay_frames = replay.replay_data
    else:
        replay_frames = replay_data
        replay = None

    # Check for DT/NC mod
    is_speed_up = bool(
        replay and replay.mods & (Mod.DoubleTime.value | Mod.Nightcore.value)
    )
    mod_text = (
        "[NC]"
        if replay and bool(replay.mods & Mod.Nightcore.value)
        else "[DT]" if replay and bool(replay.mods & Mod.DoubleTime.value) else ""
    )

    # osu! specific timing limits (in milliseconds), adjusted for speed-up mods
    OSU_LIMITS = {
        "click_speed": 6 if is_speed_up else 8,
        "suspect_speed": 3 if is_speed_up else 4,
        "inhuman_speed": 1.5 if is_speed_up else 2,
        "sus_percentage": 0.10 if is_speed_up else 0.05,
        "speed_multiplier": 1.5 if is_speed_up else 1.0,
        "adjusted_frame_time": (16.66 / 1.5 if is_speed_up else 16.66),
        "frame_sus_threshold": 0.75,
        "frame_sus_percentage": 0.40,
    }

    # Key press analysis with speed-up mod consideration
    key_intervals = []
    key_press_times = []
    last_press_time = 0

    for frame in replay_frames:
        if frame.keys > 0:  # Key pressed
            key_press_times.append(frame.time_delta)
            if last_press_time > 0:
                interval = frame.time_delta - last_press_time
                if interval > 0 and interval < 1000:  # Filter out unrealistic intervals
                    key_intervals.append(interval)
            last_press_time = frame.time_delta

    if key_intervals:
        # Adjust intervals if DT/NC is enabled
        adjusted_intervals = (
            [i / OSU_LIMITS["speed_multiplier"] for i in key_intervals]
            if is_speed_up
            else key_intervals
        )

        avg_interval = np.mean(adjusted_intervals)
        std_interval = np.std(adjusted_intervals)
        cv_interval = std_interval / avg_interval if avg_interval > 0 else 0

        # Calculate different categories of fast clicks with DT adjustment
        inhuman_clicks = sum(
            1 for i in adjusted_intervals if i < OSU_LIMITS["inhuman_speed"]
        )
        suspect_clicks = sum(
            1 for i in adjusted_intervals if i < OSU_LIMITS["suspect_speed"]
        )
        fast_clicks = sum(
            1 for i in adjusted_intervals if i < OSU_LIMITS["click_speed"]
        )
        total_clicks = len(adjusted_intervals)

        # Calculate percentages
        inhuman_percentage = inhuman_clicks / total_clicks if total_clicks else 0
        suspect_percentage = suspect_clicks / total_clicks if total_clicks else 0
        fast_percentage = fast_clicks / total_clicks if total_clicks else 0

        detailed_stats["key_press"] = {
            "avg_interval": round(avg_interval, 2),
            "std_interval": round(std_interval, 2),
            "cv_interval": round(cv_interval, 4),
            "min_interval": round(min(adjusted_intervals), 2),
            "max_interval": round(max(adjusted_intervals), 2),
            "total_presses": total_clicks,
            "inhuman_clicks_percentage": round(inhuman_percentage * 100, 2),
            "suspect_clicks_percentage": round(suspect_percentage * 100, 2),
            "fast_clicks_percentage": round(fast_percentage * 100, 2),
        }

        # Enhanced inhuman speed detection with different confidence levels
        if inhuman_percentage > 0.02:  # Tăng ngưỡng từ 0.01 lên 0.02
            suspicious["inhuman_speed"] = {
                "confidence": "High [BETA]",
                "reason": f"Physically impossible click speed detected ({inhuman_percentage:.1%} < {OSU_LIMITS['inhuman_speed']}ms) {mod_text}\nWarning: This detection is in beta and may need manual verification",
            }
        elif (
            suspect_percentage > OSU_LIMITS["sus_percentage"]
            and inhuman_percentage > 0.01
        ):
            suspicious["inhuman_speed"] = {
                "confidence": "Medium [BETA]",
                "reason": f"Suspiciously fast clicks ({suspect_percentage:.1%} < {OSU_LIMITS['suspect_speed']}ms) with some inhuman clicks ({inhuman_percentage:.1%}) {mod_text}",
            }
        elif fast_percentage > 0.3:  # More than 30% very fast (but possible) clicks
            suspicious["inhuman_speed"] = {
                "confidence": "Low",
                "reason": f"Unusually high number of fast clicks ({fast_percentage:.1%} < {OSU_LIMITS['click_speed']}ms) {mod_text}",
            }

    # Enhanced hold time analysis
    all_holdtimes = holdtimes1 + holdtimes2
    if all_holdtimes:
        hold_avg = np.mean(all_holdtimes)
        hold_std = np.std(all_holdtimes)
        hold_cv = hold_std / hold_avg if hold_avg > 0 else 0

        detailed_stats["hold_time"] = {
            "avg": round(hold_avg, 2),
            "std": round(hold_std, 2),
            "cv": round(hold_cv, 4),
            "min": round(min(all_holdtimes), 2),
            "max": round(max(all_holdtimes), 2),
            "total_holds": len(all_holdtimes),
        }

        if hold_cv < 0.15:
            suspicious["hold_bot"] = {
                "confidence": "High" if hold_cv < 0.1 else "Medium",
                "reason": f"Suspicious hold time consistency (CV: {hold_cv:.4f})",
            }

    # Enhanced frame time analysis with more accurate timewarp detection
    frame_times = []
    last_time = 0
    consistent_low_frames = 0
    total_frames = 0

    # Use adjusted typical frame time
    typical_frame_time = OSU_LIMITS["adjusted_frame_time"]

    for frame in replay_frames:
        if frame.time_delta > last_time:
            frame_time = frame.time_delta - last_time
            if frame_time > 0:
                # Adjust frame time for DT/NC
                adjusted_frame_time = (
                    frame_time / OSU_LIMITS["speed_multiplier"]
                    if is_speed_up
                    else frame_time
                )
                frame_times.append(adjusted_frame_time)
                total_frames += 1
                if (
                    adjusted_frame_time < typical_frame_time * 0.9
                ):  # 90% of expected frame time
                    consistent_low_frames += 1
        last_time = frame.time_delta

    if frame_times:
        frame_avg = np.mean(frame_times)
        frame_std = np.std(frame_times)
        frame_cv = frame_std / frame_avg if frame_avg > 0 else 0
        frame_ratio = frame_avg / typical_frame_time
        low_frame_percentage = (
            consistent_low_frames / total_frames if total_frames > 0 else 0
        )

        detailed_stats["frame_time"] = {
            "avg": round(frame_avg, 2),
            "std": round(frame_std, 2),
            "cv": round(frame_cv, 4),
            "min": round(min(frame_times), 2),
            "max": round(max(frame_times), 2),
            "total_frames": len(frame_times),
            "typical_ratio": round(frame_ratio, 3),
            "low_frame_percentage": round(low_frame_percentage * 100, 2),
        }

        # Updated timewarp detection logic
        if (
            frame_ratio < OSU_LIMITS["frame_sus_threshold"]
            and low_frame_percentage > OSU_LIMITS["frame_sus_percentage"]
        ):  # More than 30% frames too fast
            suspicious["low_frametime"] = {
                "confidence": "High [BETA]",
                "reason": f"Frame times consistently too low ({low_frame_percentage:.1%} < {typical_frame_time * 0.9:.2f}ms, avg: {frame_avg:.2f}ms) {mod_text}\nWarning: This detection is in beta and may need manual verification",
            }
        elif frame_ratio < 0.8 and low_frame_percentage > 0.25:  # Thêm điều kiện phụ
            suspicious["low_frametime"] = {
                "confidence": "Medium [BETA]",
                "reason": f"Some frame times below typical ({frame_avg:.2f}ms vs {typical_frame_time:.2f}ms) with {low_frame_percentage:.1%} suspicious frames",
            }

    return suspicious, detailed_stats


def analyze_all_replay_data(replay_path: str, settings: Settings) -> Dict[str, any]:
    """Analyze all aspects of a replay file and return combined results"""
    replay = Replay.from_path(replay_path)

    # Get holdtime analysis
    dist1, dist2, _, stats = analyze_replay_holdtimes(replay_path, settings)

    # Get frame analysis
    frame_stats = analyze_frame_times(replay.replay_data)

    # Pass replay.replay_data instead of replay_path
    suspicious, detailed_stats = detect_cheats(
        replay.replay_data,  # Changed from replay_path
        [t for t in dist1.keys()],
        [t for t in dist2.keys()],
        frame_stats,
    )

    # Extract hit timing data
    timing_data = []
    last_time = 0
    for frame in replay.replay_data:
        if frame.keys > 0:  # Key pressed
            if last_time > 0:
                timing = frame.time_delta - last_time
                if 0 < timing < 1000:  # Filter out unrealistic timings
                    timing_data.append(timing)
            last_time = frame.time_delta

    return {
        "replay": replay,
        "holdtime": {"dist1": dist1, "dist2": dist2, "stats": stats},
        "frame_stats": frame_stats,
        "suspicious": suspicious,
        "detailed_stats": detailed_stats,
        "timing_data": timing_data,
    }


def process_dropped_file(
    file_path: str, settings: Settings, analysis_type: str = "holdtime"
):
    """Process replay file with specified analysis type"""
    try:
        clean_path = clean_file_path(file_path)
        console = Console()

        console.print(f"[blue]Original path: {file_path}[/blue]")
        console.print(f"[blue]Cleaned path: {clean_path}[/blue]")

        # Validate file existence and type
        if not os.path.exists(clean_path):
            console.print(f"[red]Error: File not found: {clean_path}[/red]")
            return

        if not clean_path.endswith(".osr"):
            console.print("[red]Error: Not a valid replay file[/red]")
            return

        # Create output directory if it doesn't exist
        os.makedirs(settings.output_dir, exist_ok=True)

        # Validate analysis type
        if analysis_type not in ["holdtime", "frametime", "all", "cheat", "ur"]:
            console.print(f"[red]Error: Invalid analysis type: {analysis_type}[/red]")
            return

        with console.status(
            f"[bold blue]Analyzing replay: {os.path.basename(clean_path)}..."
        ):
            # Get all analysis data
            results = analyze_all_replay_data(clean_path, settings)
            base_path = os.path.join(settings.output_dir, Path(clean_path).stem)

            # Generate requested outputs
            if analysis_type in ["holdtime", "all"]:
                hold_fig = plot_holdtime_distribution(
                    results["holdtime"]["dist1"],
                    results["holdtime"]["dist2"],
                    results["replay"],
                    results["holdtime"]["stats"],
                )
                hold_fig.savefig(f"{base_path}_holdtime.{settings.output_format}")
                plt.close(hold_fig)

            if analysis_type in ["frametime", "all"]:
                frame_fig = plot_frame_distribution(
                    results["frame_stats"], results["replay"]
                )
                frame_fig.savefig(f"{base_path}_frametime.{settings.output_format}")
                plt.close(frame_fig)

            # Add UR analysis
            if analysis_type in ["ur", "all"]:
                ur_fig = plot_ur_distribution(results["timing_data"], results["replay"])
                ur_fig.savefig(f"{base_path}_ur.{settings.output_format}")
                plt.close(ur_fig)

            report_path = os.path.join(
                settings.output_dir, f"{Path(clean_path).stem}_analysis.txt"
            )
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(f"Analysis Report for {results['replay'].username}\n")
                f.write("=" * 50 + "\n\n")

                if results["suspicious"]:
                    f.write("(!!) SUSPICIOUS PATTERNS DETECTED:\n")
                    for cheat_type, details in results["suspicious"].items():
                        f.write(f"- {cheat_type.upper()}: {details['reason']}\n")
                        f.write(f"  Confidence: {details['confidence']}\n")
                else:
                    f.write("[OK] No suspicious patterns detected\n")

                f.write("\nDetailed Statistics:\n")
                f.write(
                    f"Frame timing (UR): {results['frame_stats']['unstable_rate']}\n"
                )
                f.write(
                    f"Hold time consistency: K1={results['holdtime']['stats']['k1_avg']}ms, "
                    f"K2={results['holdtime']['stats']['k2_avg']}ms\n"
                )

            if results["suspicious"]:
                console.print(
                    "[red](!!) Suspicious patterns detected! See analysis report.[/red]"
                )
            console.print(f"[green]Analysis saved in {settings.output_dir}[/green]")

    except Exception as e:
        console = Console()
        console.print(f"[red]Error processing file: {e}[/red]")
        raise  # Re-raise for debugging


def show_menu(settings: Settings):
    """Professional menu system with categories and descriptions"""
    while True:
        console = Console()
        console.clear()

        # Show header
        console.print(
            Panel(
                f"{APP_NAME} v{VERSION}\n" f"Created by {AUTHOR}\n" f"{GITHUB}",
                title="Welcome",
                border_style="blue",
                padding=(1, 2),
            )
        )

        # Show menu with categories
        for category, options in MENU_CONFIG.items():
            console.print(f"\n[bold blue]{category}[/bold blue]")
            console.print("─" * len(category))
            for key, item in options.items():
                console.print(
                    f"{key}. [cyan]{item['label']}[/cyan]"
                    f"\n   [dim]{item['description']}[/dim]"
                )

        # Get user input
        try:
            option = Prompt.ask(
                "\nSelect option",
                choices=[str(i) for i in range(1, len(MENU_CONFIG) + 5)],
                show_choices=False,
            )

            # Handle menu actions
            action = None
            for category in MENU_CONFIG.values():
                if option in category:
                    action = category[option]["action"]
                    break

            if action == "exit":
                console.print("[yellow]Goodbye![/yellow]")
                return

            elif action == "settings":
                settings = configure_settings()
                settings.save()
                console.print("[green]Settings saved successfully![/green]")
                time.sleep(1)

            elif action == "drop":
                handle_drop_mode(settings)

            else:
                # Handle analysis actions
                try:
                    replay_path = select_replay()
                    if replay_path:
                        process_dropped_file(replay_path, settings, action)
                        input("\nPress Enter to continue...")
                except Exception as e:
                    console.print(f"[red]Error processing replay: {e}[/red]")
                    time.sleep(2)

        except KeyboardInterrupt:
            console.print("\n[yellow]Operation cancelled.[/yellow]")
            time.sleep(1)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            time.sleep(2)


def handle_drop_mode(settings: Settings):
    """Handle drop mode with improved UI"""
    console = Console()
    console.print(
        Panel.fit(
            "Drop Mode - Batch Processing\n\n"
            "• Drag and drop .osr files here\n"
            "• Press Ctrl+C to exit",
            title="Instructions",
            border_style="yellow",
        )
    )

    analysis_map = {
        "h": "holdtime",
        "f": "frametime",
        "a": "all",
        "c": "cheat",
        "u": "ur",
    }

    analysis_descriptions = {
        "h": "Hold Time Analysis",
        "f": "Frame Time Analysis",
        "a": "Full Analysis",
        "c": "Cheat Detection",
        "u": "UR Analysis",
    }

    # Show analysis options
    for key, desc in analysis_descriptions.items():
        console.print(f"[cyan]{key}[/cyan]: {desc}")

    analysis_type = Prompt.ask(
        "Select analysis type", choices=list(analysis_map.keys()), default="a"
    )

    console.print(f"\n[green]Selected: {analysis_descriptions[analysis_type]}[/green]")
    console.print("[yellow]Waiting for files...[/yellow]")

    try:
        while True:
            file_path = input().strip()
            if file_path:
                process_dropped_file(file_path, settings, analysis_map[analysis_type])
    except KeyboardInterrupt:
        console.print("\n[yellow]Exiting drop mode...[/yellow]")
        time.sleep(1)


def main():
    """Main entry point of the application"""
    try:
        # Initialize settings
        settings = Settings.load()

        # Show menu and handle user interaction
        show_menu(settings)

    except KeyboardInterrupt:
        print("\nProgram terminated by user")
        sys.exit(0)
    except Exception as e:
        console = Console()
        console.print(f"[red]Fatal Error: {e}[/red]")
        input("\nPress Enter to exit...")
        sys.exit(1)


def save_analysis_report(output_path: str, results: Dict[str, any]):
    """Save enhanced analysis results to a text file"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"Analysis Report for {results['replay'].username}\n")
        f.write("=" * 50 + "\n\n")

        # Write suspicious patterns
        if results["suspicious"]:
            f.write("(!!) SUSPICIOUS PATTERNS DETECTED:\n")
            for cheat_type, details in results["suspicious"].items():
                f.write(f"- {cheat_type.upper()}: {details['reason']}\n")
                f.write(f"  Confidence: {details['confidence']}\n")
            f.write("\n")
        else:
            f.write("[OK] No suspicious patterns detected\n\n")

        # Write detailed statistics
        f.write("=== Detailed Statistics ===\n\n")

        # Key Press Statistics
        f.write("Key Press Analysis:\n")
        if "key_press" in results.get("detailed_stats", {}):
            kp = results["detailed_stats"]["key_press"]
            f.write(f"- Average interval: {kp['avg_interval']}ms\n")
            f.write(f"- Std deviation: {kp['std_interval']}ms\n")
            f.write(f"- Consistency (CV): {kp['cv_interval']}\n")
            f.write(f"- Fastest press: {kp['min_interval']}ms\n")
            f.write(f"- Slowest press: {kp['max_interval']}ms\n")
            f.write(f"- Total key presses: {kp['total_presses']}\n")
        f.write("\n")

        # Hold Time Statistics
        f.write("Hold Time Analysis:\n")
        if "hold_time" in results.get("detailed_stats", {}):
            ht = results["detailed_stats"]["hold_time"]
            f.write(f"- Average hold: {ht['avg']}ms\n")
            f.write(f"- Std deviation: {ht['std']}ms\n")
            f.write(f"- Consistency (CV): {ht['cv']}\n")
            f.write(f"- Shortest hold: {ht['min']}ms\n")
            f.write(f"- Longest hold: {ht['max']}ms\n")
            f.write(f"- Total holds: {ht['total_holds']}\n")
        f.write("\n")

        # Frame Time Analysis
        f.write("Frame Time Analysis:\n")
        if "frame_time" in results.get("detailed_stats", {}):
            ft = results["detailed_stats"]["frame_time"]
            f.write(f"- Average frame: {ft['avg']}ms (Typical: 16.66ms)\n")
            f.write(f"- Ratio to typical: {ft['typical_ratio']} (1.0 is typical)\n")
            f.write(f"- Std deviation: {ft['std']}ms\n")
            f.write(f"- Consistency (CV): {ft['cv']}\n")
            f.write(f"- Fastest frame: {ft['min']}ms\n")
            f.write(f"- Slowest frame: {ft['max']}ms\n")
            f.write(f"- Total frames: {ft['total_frames']}\n")
            f.write(
                "Note: Frame times consistently below 16.66ms may indicate timewarp\n"
            )
        f.write(f"- Unstable Rate: {results['frame_stats']['unstable_rate']}\n")


def handle_analysis(replay_path: str, settings: Settings, analysis_type: str):
    """Handle analysis of a replay file"""
    try:
        # Get all analysis data
        results = analyze_all_replay_data(replay_path, settings)
        base_path = os.path.join(settings.output_dir, Path(replay_path).stem)

        # Generate requested outputs
        if analysis_type in ["holdtime", "all"]:
            hold_fig = plot_holdtime_distribution(
                results["holdtime"]["dist1"],
                results["holdtime"]["dist2"],
                results["replay"],
                results["holdtime"]["stats"],
            )
            hold_fig.savefig(f"{base_path}_holdtime.{settings.output_format}")
            plt.close(hold_fig)

        if analysis_type in ["frametime", "all"]:
            frame_fig = plot_frame_distribution(
                results["frame_stats"], results["replay"]
            )
            frame_fig.savefig(f"{base_path}_frametime.{settings.output_format}")
            plt.close(frame_fig)

        # Save analysis report
        report_path = os.path.join(
            settings.output_dir, f"{Path(replay_path).stem}_analysis.txt"
        )
        save_analysis_report(report_path, results)

        return results

    except Exception as e:
        console = Console()
        console.print(f"[red]Error processing file: {e}[/red]")
        raise


def plot_ur_distribution(timing_data: List[float], replay) -> plt.Figure:
    """Creates visualization of hit timing distribution (UR)"""
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 4))

    # Calculate UR
    std_dev = np.std(timing_data) if timing_data else 0
    ur = std_dev * 10
    mean = np.mean(timing_data) if timing_data else 0

    # Create histogram
    if timing_data:
        # Calculate optimal bin width using Freedman-Diaconis rule
        iqr = np.percentile(timing_data, 75) - np.percentile(timing_data, 25)
        bin_width = 2 * iqr / (len(timing_data) ** (1 / 3)) if iqr > 0 else 1
        num_bins = int((max(timing_data) - min(timing_data)) / bin_width)
        num_bins = min(max(20, num_bins), 50)  # Keep bins between 20 and 50

        n, bins, patches = ax.hist(
            timing_data, bins=num_bins, color="#9999FF", alpha=0.8
        )

        # Add vertical lines for mean and standard deviations
        ax.axvline(x=mean, color="#FF99FF", linestyle="--", alpha=0.8, label="Mean")
        ax.axvline(
            x=mean + std_dev, color="#FF9999", linestyle=":", alpha=0.6, label="+1σ"
        )
        ax.axvline(
            x=mean - std_dev, color="#FF9999", linestyle=":", alpha=0.6, label="-1σ"
        )

    # Styling
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_facecolor("black")
    ax.grid(True, axis="both", linestyle="--", alpha=0.2)
    ax.legend(fontsize=8)

    # Get mods string
    mods_str = (
        "+" + ",".join([mod.name for mod in Mod if replay.mods & mod.value])
        if replay.mods
        else "NoMod"
    )
    play_timestamp = replay.timestamp.strftime("%Y-%m-%d %H:%M:%S")

    # Update title
    title_text = (
        f"Circle | Hit Timing Distribution\n"
        f"Player: {replay.username} | {mods_str}\n"
        f"Beatmap ID: {replay.beatmap_hash[:8]}... | Score ID: {replay.replay_hash[:8]}...\n"
        f"UR: {ur:.2f} | Mean offset: {mean:.2f}ms | σ: {std_dev:.2f}ms\n"
        f"Played on: {play_timestamp}"
    )
    ax.set_title(title_text, pad=20, loc="left", fontsize=10)

    # Add axis labels
    ax.set_xlabel("Hit Timing Offset (ms)", fontsize=10)
    ax.set_ylabel("Number of Hits", fontsize=10)

    # Add explanation text with more details
    fig.text(
        0.01,
        0.01,
        "Unstable Rate (UR) Analysis:\n"
        "• X-axis: Time difference between consecutive hits (ms)\n"
        "• Y-axis: Number of hits at each timing\n"
        "• Lower UR = more consistent timing\n"
        "• Typical UR range: 70-120\n"
        "• <50 may indicate relax hack",
        color="#AAAAAA",
        fontsize=8,
        alpha=0.9,
        bbox=dict(facecolor="black", edgecolor="none", alpha=0.7, pad=5),
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)

    return fig


if __name__ == "__main__":
    main()
