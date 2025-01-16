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
import shlex
from urllib.parse import unquote
import re

# Program configuration
PROGRAM_VERSION = "1.0.0"
PROGRAM_INFO = """
Circle Hold Distribution Analyzer
Version: {version}
Created by: tamaisme
GitHub: https://github.com/tamaisme/circle-hold-contribution
"""


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
                # # Clean up any old settings format
                # if "max_holdtime" in data:
                #     del data["max_holdtime"]
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

    # Find regions with data and analyze density
    active_regions = []
    current_region = None
    current_density = []
    gap_threshold = 3  # Base threshold for gaps
    density_threshold = 0.4  # Minimum density to keep region together

    for i in range(len(bins) - 1):
        has_data = times1[i] > 0 or times2[i] > 0

        if has_data:
            if current_region is None:
                current_region = [i, i]
                current_density = [has_data]
            else:
                current_region[1] = i
                current_density.append(has_data)
        elif current_region is not None:
            # Check density of current region
            region_length = current_region[1] - current_region[0] + 1
            density = sum(current_density) / region_length

            # Split region if it's too sparse
            if region_length > 10 and density < density_threshold:
                # Find dense sub-regions
                sub_regions = []
                sub_start = current_region[0]
                last_data = 0

                for j, has_point in enumerate(current_density):
                    if has_point:
                        if j - last_data > gap_threshold:
                            if sub_start < current_region[0] + last_data:
                                sub_regions.append(
                                    [sub_start, current_region[0] + last_data]
                                )
                            sub_start = current_region[0] + j
                        last_data = j

                if sub_start <= current_region[0] + last_data:
                    sub_regions.append([sub_start, current_region[0] + last_data])

                active_regions.extend(sub_regions)
            else:
                active_regions.append(current_region)

            current_region = None
            current_density = []

    if current_region is not None:
        active_regions.append(current_region)

    # Merge very close regions
    merged_regions = []
    if active_regions:
        current = active_regions[0]
        for region in active_regions[1:]:
            if region[0] - current[1] <= gap_threshold:
                current[1] = region[1]
            else:
                merged_regions.append(current)
                current = region
        merged_regions.append(current)

    # Add padding to regions
    padding = 1
    final_regions = []
    for start, end in merged_regions:
        final_regions.append(
            [max(0, start - padding), min(len(bins) - 2, end + padding)]
        )

    # Plot data only in active regions
    plt.style.use("dark_background")
    fig, axes = plt.subplots(
        len(final_regions), 1, figsize=(12, 3 * len(final_regions))
    )
    if len(final_regions) == 1:
        axes = [axes]

    for ax, (start, end) in zip(axes, final_regions):
        region_bins = bins[start : end + 2]
        region_times = times1[start : end + 1]

        ax.bar(
            region_bins[:-1],
            region_times,
            width=2,
            align="edge",
            color="#9999FF",
            alpha=0.8,
        )

        # Configure each subplot
        ax.set_xlim(region_bins[0] - 1, region_bins[-1] + 1)
        max_y = max(times1) * 1.1
        ax.set_ylim(0, max_y)

        # Smarter y-axis ticks
        if max_y > 20:
            # For larger numbers, show fewer ticks
            step = max(1, int(max_y / 10))  # Show around 10 ticks
            ax.yaxis.set_major_locator(plt.MultipleLocator(base=step))
        else:
            # For smaller numbers, show integers
            ax.yaxis.set_major_locator(plt.MultipleLocator(base=1.0))

        # Format y-axis to show only integers
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: str(int(x)) if x.is_integer() else "")
        )

        # Apply existing styling to each subplot
        ax.grid(True, axis="y", linestyle="--", alpha=0.2, color="gray")
        ax.grid(True, axis="x", linestyle="--", alpha=0.2, color="gray")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.set_facecolor("black")

        # X-axis ticks
        x_ticks = np.arange(int(region_bins[0]), int(region_bins[-1] + 1), 3)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([str(int(x)) for x in x_ticks], rotation=0)
        ax.tick_params(axis="both", which="major", labelsize=6)

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

    # Create multi-line title with metadata
    title_text = (
        f"Circle | Holdtime distribution\n"
        f"Player: {replay.username}\n"
        f"Beatmap ID: {replay.beatmap_hash[:8]}... | Score ID: {replay.replay_hash[:8]}...\n"
        f"Mods: {mods_str} | Played on: {play_timestamp}\n"
        f"K1: {stats['k1_count']} ({stats['k1_avg']}ms avg) "
        f"K2: {stats['k2_count']} ({stats['k2_avg']}ms avg)"
    )
    axes[0].set_title(title_text, pad=20, loc="left", fontsize=10, fontweight="normal")

    # Adjust figure height to accommodate larger title
    fig.set_figheight(3 * len(final_regions) + 1)

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
            PROGRAM_INFO.format(version=PROGRAM_VERSION),
            title="Welcome",
            border_style="blue",
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
    """
    Analyzes frame timing distribution from replay data
    Returns dictionary containing:
        - avg: Average frame time
        - min/max: Frame time range
        - distribution: Frame time frequency
        - unstable_rate: Frame timing consistency (lower is better)
    """
    frame_times = []
    for frame in replay_data:
        if frame.time_delta > 0:  # Ignore zero deltas
            frame_times.append(frame.time_delta)

    if not frame_times:
        return {"avg": 0, "min": 0, "max": 0, "distribution": {}, "unstable_rate": 0}

    # Calculate statistics
    avg_frame = np.mean(frame_times)
    min_frame = min(frame_times)
    max_frame = max(frame_times)
    unstable_rate = np.std(frame_times) * 10

    # Create distribution
    frame_dist = {}
    for time in frame_times:
        rounded = round(time)
        frame_dist[rounded] = frame_dist.get(rounded, 0) + 1

    return {
        "avg": round(avg_frame, 2),
        "min": round(min_frame, 2),
        "max": round(max_frame, 2),
        "distribution": frame_dist,
        "unstable_rate": round(unstable_rate, 2),
    }


def plot_frame_distribution(frame_stats: Dict[str, any], replay) -> plt.Figure:
    """
    Creates visualization of frame time distribution
    Shows:
    - Frame time frequency
    - Performance metrics
    - Player and replay metadata
    """
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 4))

    # Plot distribution
    times = sorted(frame_stats["distribution"].keys())
    values = [frame_stats["distribution"][t] for t in times]
    ax.bar(times, values, width=0.8, color="#FF99FF", alpha=0.8)

    # Configure axes
    ax.set_xlim(min(times) - 1, max(times) + 1)
    ax.set_ylim(0, max(values) * 1.1)

    # Get replay metadata
    mods_str = (
        "+" + ",".join([mod.name for mod in Mod if replay.mods & mod.value])
        if replay.mods
        else "NoMod"
    )
    play_timestamp = replay.timestamp.strftime("%Y-%m-%d %H:%M:%S")

    # Title with stats and metadata
    title_text = (
        f"Circle | Frame Time Analysis\n"
        f"Player: {replay.username} | {mods_str}\n"
        f"Average: {frame_stats['avg']}ms | Min: {frame_stats['min']}ms | "
        f"Max: {frame_stats['max']}ms | UR: {frame_stats['unstable_rate']}\n"
        f"Played on: {play_timestamp}"
    )
    ax.set_title(title_text, pad=20, loc="left", fontsize=10)

    # Styling
    ax.grid(True, axis="y", linestyle="--", alpha=0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_facecolor("black")
    fig.patch.set_facecolor("black")

    plt.tight_layout()
    return fig


def process_dropped_file(
    file_path: str, settings: Settings, analysis_type: str = "holdtime"
):
    """
    Process replay file with specified analysis type
    Types:
    - holdtime: Only analyze key hold times
    - frametime: Only analyze frame timings
    - all: Perform both analyses
    """
    try:
        clean_path = clean_file_path(file_path)
        console = Console()

        console.print(f"[blue]Original path: {file_path}[/blue]")
        console.print(f"[blue]Cleaned path: {clean_path}[/blue]")

        if not os.path.exists(clean_path):
            console.print(f"[red]Error: File not found: {clean_path}[/red]")
            return

        if not clean_path.endswith(".osr"):
            print("[red]Error: Not a valid replay file[/red]")
            return

        with console.status(
            f"[bold blue]Analyzing replay: {os.path.basename(clean_path)}..."
        ):
            replay = Replay.from_path(clean_path)
            base_path = os.path.join(settings.output_dir, Path(clean_path).stem)

            if analysis_type in ["holdtime", "all"]:
                dist1, dist2, replay, stats = analyze_replay_holdtimes(
                    clean_path, settings
                )
                hold_fig = plot_holdtime_distribution(dist1, dist2, replay, stats)
                hold_fig.savefig(f"{base_path}_holdtime.{settings.output_format}")
                plt.close(hold_fig)

            if analysis_type in ["frametime", "all"]:
                frame_stats = analyze_frame_times(replay.replay_data)
                frame_fig = plot_frame_distribution(frame_stats, replay)
                frame_fig.savefig(f"{base_path}_frametime.{settings.output_format}")
                plt.close(frame_fig)

            console.print(f"[green]Analysis saved in {settings.output_dir}[/green]")

    except Exception as e:
        console = Console()
        console.print(f"[red]Error processing file: {e}[/red]")


def show_menu(settings: Settings):
    """
    Main menu system
    Features:
    - Hold time analysis
    - Frame time analysis
    - Combined analysis
    - Settings management
    - Drag & drop mode
    """
    while True:
        console = Console()
        console.clear()
        show_welcome()

        choices = [
            "1. Analyze Hold Time",
            "2. Analyze Frame Time",
            "3. Full Analysis",
            "4. Settings",
            "5. Drop Mode",
            "6. Exit",
        ]

        for choice in choices:
            console.print(choice)

        option = Prompt.ask("\nSelect option", choices=["1", "2", "3", "4", "5", "6"])

        if option == "1":
            try:
                replay_path = select_replay()
                process_dropped_file(replay_path, settings, "holdtime")
                input("\nPress Enter to continue...")
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                time.sleep(2)

        elif option == "2":
            try:
                replay_path = select_replay()
                process_dropped_file(replay_path, settings, "frametime")
                input("\nPress Enter to continue...")
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                time.sleep(2)

        elif option == "3":
            try:
                replay_path = select_replay()
                process_dropped_file(replay_path, settings, "all")
                input("\nPress Enter to continue...")
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                time.sleep(2)

        elif option == "4":
            settings = configure_settings()
            settings.save()
            console.print("[green]Settings saved![/green]")
            time.sleep(1)

        elif option == "5":
            console.print(
                "\n[yellow]Drop Mode activated - Drag and drop .osr files here[/yellow]"
            )
            console.print(
                "[yellow]Analysis type: (h)oldtime, (f)rametime, (a)ll[/yellow]"
            )
            console.print("[yellow]Press Ctrl+C to exit drop mode[/yellow]\n")

            analysis_type = Prompt.ask(
                "Analysis type", choices=["h", "f", "a"], default="a"
            )
            analysis_map = {"h": "holdtime", "f": "frametime", "a": "all"}

            try:
                while True:
                    file_path = input()
                    process_dropped_file(
                        file_path, settings, analysis_map[analysis_type]
                    )
            except KeyboardInterrupt:
                console.print("\n[yellow]Exiting drop mode...[/yellow]")
                time.sleep(1)

        elif option == "6":
            console.print("[yellow]Goodbye![/yellow]")
            sys.exit(0)


def main():
    try:
        settings = Settings.load()
        show_menu(settings)
    except Exception as e:
        console = Console()
        console.print(f"[red]Fatal Error: {e}[/red]")
        input("\nPress Enter to exit...")


if __name__ == "__main__":
    main()
