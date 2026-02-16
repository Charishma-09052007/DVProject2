"""
G-G Diagram (Friction Circle) — Qualifying (Team-specific)
Creates 3 plots: Red Bull, Mercedes, McLaren

Uses FastF1 car_data and pos_data directly per docs:
https://docs.fastf1.dev/core.html
"""

import fastf1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.signal import savgol_filter
import warnings

warnings.filterwarnings('ignore')

# Cache
fastf1.Cache.enable_cache('/root/DV/DVProject2/ff1_cache')

YEAR = 2024
GP = 'Bahrain'
SESSION_TYPE = 'Q'

TEAM_MAP = {
    'Red Bull': 'Red Bull Racing',
    'Mercedes': 'Mercedes',
    'McLaren': 'McLaren',
}

# Force red/green for driver distinction
TEAM_COLORS = {
    'Red Bull': ('#E10600', '#00D000'),
    'Mercedes': ('#E10600', '#00D000'),
    'McLaren': ('#E10600', '#00D000'),
}

OUTPUTS = {
    'Red Bull': '/root/DV/DVProject2/Q1_GG_Qualifying_RedBull.png',
    'Mercedes': '/root/DV/DVProject2/Q1_GG_Qualifying_Mercedes.png',
    'McLaren': '/root/DV/DVProject2/Q1_GG_Qualifying_McLaren.png',
}


def load_session(year, gp, session_type):
    print(f"Loading {year} {gp} {session_type}...")
    session = fastf1.get_session(year, gp, session_type)
    session.load()
    print(f"  Loaded. Drivers: {len(session.drivers)}")
    return session


def get_merged_telemetry(session, driver_num):
    car = session.car_data[driver_num].copy()
    pos = session.pos_data[driver_num].copy()
    return car.merge_channels(pos)


def compute_g_forces(tel_df):
    df = tel_df.copy()
    df['dt'] = df['SessionTime'].diff().dt.total_seconds()
    df['Speed_ms'] = df['Speed'] / 3.6
    df['dv'] = df['Speed_ms'].diff()
    df['G_long'] = (df['dv'] / df['dt']) / 9.81

    if 'X' in df.columns and 'Y' in df.columns:
        dx = df['X'].diff()
        dy = df['Y'].diff()
        heading = np.arctan2(dy, dx)
        dheading = heading.diff()
        dheading = np.arctan2(np.sin(dheading), np.cos(dheading))
        omega = dheading / df['dt']
        df['G_lat'] = (df['Speed_ms'] * omega) / 9.81
    else:
        df['G_lat'] = 0.0

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['G_long', 'G_lat', 'dt'])
    df = df[df['dt'] > 0]

    if len(df) > 15:
        df['G_long'] = savgol_filter(df['G_long'], window_length=11, polyorder=3)
        df['G_lat'] = savgol_filter(df['G_lat'], window_length=11, polyorder=3)

    df = df[(df['G_long'].abs() < 6) & (df['G_lat'].abs() < 6)]
    return df


def mid_window(df):
    df = df[df['Speed'] > 50]
    total = df['SessionTime'].max().total_seconds()
    mid = total / 2
    mask = (df['SessionTime'].dt.total_seconds() > mid - 300) & \
           (df['SessionTime'].dt.total_seconds() < mid + 300)
    return df[mask]


def plot_team_gg(session, team_key, team_name):
    results = session.results
    team_results = results[results['TeamName'] == team_name]

    if len(team_results) < 2:
        raise RuntimeError(f"Not enough drivers for team: {team_name}")

    d1 = team_results.iloc[0]
    d2 = team_results.iloc[1]

    d1_num = str(d1['DriverNumber'])
    d2_num = str(d2['DriverNumber'])
    d1_abbr = d1['Abbreviation']
    d2_abbr = d2['Abbreviation']
    d1_name = d1['FullName']
    d2_name = d2['FullName']

    print(f"  Team: {team_key} — {d1_name} vs {d2_name}")

    tel1 = mid_window(get_merged_telemetry(session, d1_num))
    tel2 = mid_window(get_merged_telemetry(session, d2_num))

    gf1 = compute_g_forces(tel1)
    gf2 = compute_g_forces(tel2)

    color1, color2 = TEAM_COLORS[team_key]

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.patch.set_facecolor('#1a1a2e')

    fig.suptitle(
        f"G-G Diagram (Friction Circle) — Qualifying {YEAR} {GP}\n"
        f"{team_key} • Mid-Session Sample (~10 min)",
        fontsize=14, fontweight='bold', color='white', y=1.02
    )

    max_g1 = max(gf1['G_long'].abs().quantile(0.98), gf1['G_lat'].abs().quantile(0.98))
    max_g2 = max(gf2['G_long'].abs().quantile(0.98), gf2['G_lat'].abs().quantile(0.98))
    max_g_both = max(max_g1, max_g2) * 1.15

    for ax, gf, color, name, abbr in [
        (axes[0], gf1, color1, d1_name, d1_abbr),
        (axes[1], gf2, color2, d2_name, d2_abbr),
    ]:
        ax.set_facecolor('#16213e')
        ax.scatter(gf['G_lat'], gf['G_long'], s=2, alpha=0.35, c=color, label=abbr)
        circle = Circle((0, 0), max_g_both * 0.85, fill=False, color='gray',
                        linestyle='--', linewidth=1.5, alpha=0.5)
        ax.add_patch(circle)
        ax.set_xlim(-max_g_both, max_g_both)
        ax.set_ylim(-max_g_both, max_g_both)
        ax.set_aspect('equal')
        ax.axhline(0, color='gray', linewidth=0.5, alpha=0.4)
        ax.axvline(0, color='gray', linewidth=0.5, alpha=0.4)
        ax.set_xlabel('Lateral G (Turning)', fontsize=11, color='white')
        ax.set_ylabel('Longitudinal G (Braking ↓ / Accel ↑)', fontsize=11, color='white')
        ax.set_title(f'{name}', fontsize=12, fontweight='bold', color=color)
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.15, color='white')
        for spine in ax.spines.values():
            spine.set_color('gray')

    ax = axes[2]
    ax.set_facecolor('#16213e')
    ax.scatter(gf1['G_lat'], gf1['G_long'], s=2, alpha=0.3, c=color1, label=d1_abbr)
    ax.scatter(gf2['G_lat'], gf2['G_long'], s=2, alpha=0.3, c=color2, label=d2_abbr)
    circle3 = Circle((0, 0), max_g_both * 0.85, fill=False, color='white',
                     linestyle='--', linewidth=1.5, alpha=0.4, label='Friction Limit')
    ax.add_patch(circle3)
    ax.set_xlim(-max_g_both, max_g_both)
    ax.set_ylim(-max_g_both, max_g_both)
    ax.set_aspect('equal')
    ax.axhline(0, color='gray', linewidth=0.5, alpha=0.4)
    ax.axvline(0, color='gray', linewidth=0.5, alpha=0.4)
    ax.set_xlabel('Lateral G (Turning)', fontsize=11, color='white')
    ax.set_ylabel('Longitudinal G (Braking ↓ / Accel ↑)', fontsize=11, color='white')
    ax.set_title('Overlay Comparison', fontsize=12, fontweight='bold', color='white')
    ax.legend(fontsize=9, loc='upper right', facecolor='#16213e', labelcolor='white')
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.15, color='white')
    for spine in ax.spines.values():
        spine.set_color('gray')

    for a in axes:
        a.annotate('BRAKING', xy=(0, -max_g_both * 0.75), fontsize=8,
                   ha='center', color='#aaa', style='italic')
        a.annotate('ACCEL', xy=(0, max_g_both * 0.75), fontsize=8,
                   ha='center', color='#aaa', style='italic')
        a.annotate('LEFT', xy=(-max_g_both * 0.75, 0), fontsize=8,
                   ha='center', va='center', color='#aaa', style='italic', rotation=90)
        a.annotate('RIGHT', xy=(max_g_both * 0.75, 0), fontsize=8,
                   ha='center', va='center', color='#aaa', style='italic', rotation=90)

    out_path = OUTPUTS[team_key]
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓ Saved: {out_path}")


if __name__ == '__main__':
    session = load_session(YEAR, GP, SESSION_TYPE)
    for team_key, team_name in TEAM_MAP.items():
        plot_team_gg(session, team_key, team_name)
    print("Done.")
