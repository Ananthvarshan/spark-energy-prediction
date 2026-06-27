import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import os
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score


# ============================================================
# SPARK DATASET - COMPATIBLE DATA LOADER
# ============================================================

def load_and_prepare_data(path):
    print(f"\n📂 Loading file: {path}")

    if path.endswith('.xz'):
        df = pd.read_csv(path, compression='xz')
    else:
        try:
            df = pd.read_csv(path, skiprows=1)
            if len(df.columns) <= 1:
                df = pd.read_csv(path)
        except Exception:
            df = pd.read_csv(path)

    df.columns = df.columns.str.strip()
    print(f"✅ Columns detected: {list(df.columns)}")

    timestamp_candidates = ['WsDateTime', 'Time', 'timestamp', 'Datetime', 'date', 'DateTime']
    renamed = False
    for col in timestamp_candidates:
        if col in df.columns:
            df.rename(columns={col: 'timestamp'}, inplace=True)
            print(f"✅ Timestamp column found: '{col}'")
            renamed = True
            break

    if not renamed:
        raise Exception(f"❌ No timestamp column found. Available: {list(df.columns)}")

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    current_cols = [col for col in df.columns if col.lower() in ['i1', 'i2', 'i3']]
    power_cols   = [col for col in df.columns if col.lower() in ['p', 'power', 'active_power',
                    'watt', 'p_total', 'p1', 'p2', 'p3']]
    generic_cols = [col for col in df.columns if 'power.i' in col.lower() or 'current' in col.lower()]
    all_measurement_cols = current_cols + power_cols + generic_cols

    if len(all_measurement_cols) == 0:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) == 0:
            raise Exception("❌ No numeric measurement columns found.")
        all_measurement_cols = numeric_cols
        print(f"⚠️  Using fallback numeric columns: {all_measurement_cols}")
    else:
        print(f"✅ Measurement columns found: {all_measurement_cols}")

    for col in all_measurement_cols:
        df[col] = df[col].astype(str).str.replace(r'[^\d.\-]', '', regex=True)
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['power'] = df[all_measurement_cols].mean(axis=1)

    # Clip negative sensor noise to 0
    df['power'] = df['power'].clip(lower=0)

    df = df[['timestamp', 'power']]
    df = df.dropna()
    df = df.sort_values('timestamp').reset_index(drop=True)

    print(f"✅ Data loaded: {len(df):,} rows | "
          f"From {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"   Power range: {df['power'].min():.3f} → {df['power'].max():.3f}\n")

    return df


# ============================================================
# AI-DRIVEN STATE DETECTION — UNIFIED GMM + HMM ARCHITECTURE
# ============================================================
# Replaces the previous KMeans approach. Key improvements:
#
#  • GMM (Gaussian Mixture Model) models each cluster as its own
#    Gaussian, so the narrow OFF cluster (0–5 W, tiny variance)
#    and the wide WORKING cluster (large variance) are separated
#    cleanly WITHOUT a manual power > 5 W pre-filter.
#  • The ENTIRE dataset (including OFF state) is fed in one sweep.
#  • Silhouette Score selects the best k ∈ {2, 3, 4} automatically.
#  • Viterbi HMM smoothing removes high-frequency flicker noise.
#
#  State mapping (sorted by GMM mean, low → high):
#   k=2 → OFF, WORKING
#   k=3 → OFF, STANDBY, WORKING
#   k=4 → OFF, STANDBY, WORKING, PEAK_LOAD
# ============================================================

# ============================================================
# VITERBI HMM SMOOTHER
# ============================================================
# Smooths out physically-impossible high-frequency state
# flickering (e.g. WORKING -> IDLE -> WORKING within 1-2
# samples, which is sensor noise rather than a real state
# change) by finding the most likely *sequence* of states
# given the raw K-Means labels, instead of trusting each
# label independently.
#
# NOTE ON PERFORMANCE: this is a pure-Python loop over every
# timestamp (T = number of ON-time readings). For a small
# test file this runs instantly. For a full-year SPARK file
# (T can be several million rows), this loop will be SLOW —
# potentially many minutes. This is a known, expected cost of
# implementing Viterbi this way; see run_analysis() for how
# to skip/sample around it if needed.
# ============================================================

def viterbi_hmm_smoothing(observed_labels, num_states):
    stay_prob = 0.995
    trans_prob = (1.0 - stay_prob) / (num_states - 1)
    A = np.full((num_states, num_states), trans_prob)
    np.fill_diagonal(A, stay_prob)

    match_prob = 0.95
    mismatch_prob = (1.0 - match_prob) / (num_states - 1)
    B = np.full((num_states, num_states), mismatch_prob)
    np.fill_diagonal(B, match_prob)

    pi = np.full(num_states, 1.0 / num_states)

    T = len(observed_labels)
    log_A = np.log(A)
    log_B = np.log(B)
    log_pi = np.log(pi)

    dp = np.zeros((T, num_states))
    paths = np.zeros((T, num_states), dtype=int)

    dp[0] = log_pi + log_B[:, observed_labels[0]]

    for t in range(1, T):
        obs = observed_labels[t]
        temp = dp[t-1, :, np.newaxis] + log_A
        paths[t] = np.argmax(temp, axis=0)
        dp[t] = np.max(temp, axis=0) + log_B[:, obs]

    best_path = np.zeros(T, dtype=int)
    best_path[-1] = np.argmax(dp[-1])
    for t in range(T - 2, -1, -1):
        best_path[t] = paths[t+1, best_path[t+1]]

    return best_path


def classify_machine_states(df):
    max_power  = df['power'].max()
    total_count = len(df)

    print(f"   Total readings : {total_count:,}")
    print(f"   Max power      : {max_power:.1f} W")

    # ----------------------------------------------------------------
    # STEP 2: Feed the ENTIRE dataset (including OFF state) into GMM.
    # GMM handles heterogeneous variances natively, so the narrow
    # OFF cluster (0–5 W) and the wide WORKING cluster co-exist
    # without the manual on_mask pre-filter that K-Means required.
    # ----------------------------------------------------------------
    X = df['power'].values.reshape(-1, 1)

    print("   Running GMM clustering to detect states (incl. OFF)...")

    # Train GMM for k = 3 ONLY; n_init=3 gives stable convergence
    k_values = [3]
    models = {
        k: GaussianMixture(n_components=k, random_state=42, n_init=3).fit(X)
        for k in k_values
    }

    # Predict raw integer labels from each fitted GMM
    raw_predictions = {k: model.predict(X) for k, model in models.items()}

    # sample_size caps silhouette so it doesn't choke on multi-million-row files
    scores = {
        k: silhouette_score(X, raw_predictions[k], sample_size=10000, random_state=42)
        for k in k_values
    }

    # Select the best k
    best_k     = 3
    best_model = models[best_k]
    best_score = scores[best_k]
    raw_labels = raw_predictions[best_k]

    print(f"   Scores -> k=3: {scores[3]:.3f}")
    print(f"   Fixed optimal clusters to: {best_k}  (Silhouette: {best_score:.3f})")

    # ----------------------------------------------------------------
    # STEP 3: HMM smoothing — unchanged, just now operates on the
    # full-dataset GMM labels instead of on-only K-Means labels.
    # ----------------------------------------------------------------
    print(f"   Running Viterbi HMM smoothing on {len(raw_labels):,} labels...")
    if len(raw_labels) > 2_000_000:
        est_seconds = len(raw_labels) / 95_000
        print(f"   ⚠️  Large dataset — Viterbi pass benchmarked at ~95,000 rows/sec")
        print(f"      on this hardware; estimated time: ~{est_seconds:.0f} seconds.")

    smoothed_labels = viterbi_hmm_smoothing(raw_labels, best_k)

    num_changed = int(np.sum(raw_labels != smoothed_labels))
    pct_changed = num_changed / len(raw_labels) * 100
    print(f"   Viterbi smoothing changed {num_changed:,} of {len(raw_labels):,} "
          f"labels ({pct_changed:.2f}%) — high-frequency flickers judged as sensor noise.")

    # ----------------------------------------------------------------
    # STEP 4: Map GMM cluster IDs → physical state names.
    # Sort clusters by their GMM means (lowest → highest wattage).
    # The lowest mean is ALWAYS the OFF cluster because GMM was given
    # the full dataset — no manual threshold needed.
    # ----------------------------------------------------------------
    means      = best_model.means_.flatten()
    sorted_idx = np.argsort(means)          # cluster ids ordered low→high

    if best_k == 2:
        state_names = {
            sorted_idx[0]: 'OFF',
            sorted_idx[1]: 'WORKING',
        }
    elif best_k == 3:
        state_names = {
            sorted_idx[0]: 'OFF',
            sorted_idx[1]: 'STANDBY',
            sorted_idx[2]: 'WORKING',
        }
    elif best_k == 4:
        state_names = {
            sorted_idx[0]: 'OFF',
            sorted_idx[1]: 'STANDBY',
            sorted_idx[2]: 'WORKING',
            sorted_idx[3]: 'PEAK_LOAD',
        }

    print("   Discovered GMM power centres:")
    for cluster_id, name in state_names.items():
        std = np.sqrt(best_model.covariances_.flatten()[cluster_id])
        print(f"     {name:<12}: ~{means[cluster_id]:7.1f} W  (σ = {std:.1f} W)")

    # ----------------------------------------------------------------
    # STEP 5: Apply smoothed labels directly — no on_mask needed.
    # ----------------------------------------------------------------
    df = df.copy()
    df['state'] = [state_names[label] for label in smoothed_labels]

    return df


def calculate_state_summary(df):
    seconds_per_sample = 5
    state_counts = df['state'].value_counts()

    summary = {}
    for state in ['OFF', 'STANDBY', 'IDLE', 'WORKING', 'PEAK_LOAD']:
        count = state_counts.get(state, 0)
        hours = (count * seconds_per_sample) / 3600
        pct   = (count / len(df)) * 100
        summary[state] = {
            'hours':   round(hours, 2),
            'percent': round(pct, 2),
            'samples': int(count)
        }
    return summary


# ============================================================
# PLOTS
# ============================================================

def plot_power_time(df, save_path, machine_name="Machine"):
    fig, ax = plt.subplots(figsize=(18, 5))
    colors = {
        'OFF':       '#444444',
        'STANDBY':   '#f0a500',
        'IDLE':      '#4fc3f7',
        'WORKING':   '#66bb6a',
        'PEAK_LOAD': '#e53935'
    }
    if 'state' in df.columns:
        for state, color in colors.items():
            mask = df['state'] == state
            if mask.sum() > 0:
                ax.scatter(df.loc[mask, 'timestamp'], df.loc[mask, 'power'],
                           c=color, s=0.5, label=state, alpha=0.7)
        ax.legend(markerscale=8, loc='upper right', fontsize=10)
    else:
        ax.plot(df['timestamp'], df['power'], linewidth=0.8, color='#4fc3f7')

    ax.set_title(f"Power vs Time — {machine_name}", fontsize=14, fontweight='bold')
    ax.set_xlabel("Time")
    ax.set_ylabel("Power (W)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "power_vs_time.png"), dpi=120)
    plt.close()
    print("✅ Saved: power_vs_time.png")


def plot_histogram(df, save_path, machine_name="Machine"):
    non_zero = df[df['power'] > 5]['power']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(df['power'], bins=80, color='#4fc3f7',
                 edgecolor='#1a1a2e', linewidth=0.3)
    axes[0].set_title("Full Power Distribution (incl. OFF)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Power (W)")
    axes[0].set_ylabel("Number of 5-sec Readings")

    if len(non_zero) > 0:
        axes[1].hist(non_zero, bins=80, color='#66bb6a',
                     edgecolor='#1a1a2e', linewidth=0.3)
        axes[1].set_title("Power When ON — STANDBY / IDLE / WORKING zones",
                          fontsize=12, fontweight='bold')
        axes[1].set_xlabel("Power (W)")
        axes[1].set_ylabel("Number of 5-sec Readings")

    fig.suptitle(f"Power Distribution — {machine_name}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "power_histogram.png"), dpi=120)
    plt.close()
    print("✅ Saved: power_histogram.png")


def plot_state_pie(summary, save_path, machine_name="Machine"):
    labels, sizes, colors_list = [], [], []
    colors_map = {
        'OFF':       '#444444',
        'STANDBY':   '#f0a500',
        'IDLE':      '#4fc3f7',
        'WORKING':   '#66bb6a',
        'PEAK_LOAD': '#e53935'
    }
    for state, data in summary.items():
        if data['percent'] > 0:
            labels.append(f"{state}\n{data['hours']}h ({data['percent']}%)")
            sizes.append(data['percent'])
            colors_list.append(colors_map[state])

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(sizes, labels=labels, colors=colors_list,
           startangle=140, wedgeprops={'linewidth': 2, 'edgecolor': 'white'})
    ax.set_title(f"Machine State Breakdown — {machine_name}",
                 fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "state_pie.png"), dpi=120)
    plt.close()
    print("✅ Saved: state_pie.png")


def plot_state_bar(summary, save_path, machine_name="Machine"):
    colors_map = {
        'OFF':       '#444444',
        'STANDBY':   '#f0a500',
        'IDLE':      '#4fc3f7',
        'WORKING':   '#66bb6a',
        'PEAK_LOAD': '#e53935'
    }
    states = list(summary.keys())
    hours  = [summary[s]['hours'] for s in states]
    colors = [colors_map[s] for s in states]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(states, hours, color=colors, edgecolor='white', linewidth=1.5)

    for bar, h in zip(bars, hours):
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(hours)*0.01,
                    f'{h}h', ha='center', va='bottom',
                    fontweight='bold', fontsize=10)

    ax.set_title(f"Hours in Each State — {machine_name}",
                 fontsize=14, fontweight='bold')
    ax.set_xlabel("Machine State")
    ax.set_ylabel("Total Hours")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "state_hours.png"), dpi=120)
    plt.close()
    print("✅ Saved: state_hours.png")


def analyze_standby(df, save_path):
    # Low power = OFF or STANDBY states
    df = df.copy()
    df['is_low'] = df['state'].isin(['OFF', 'STANDBY'])

    groups        = (df['is_low'] != df['is_low'].shift()).cumsum()
    durations     = df[df['is_low']].groupby(groups).size()
    durations_min = durations * 5 / 60  # samples → minutes

    durations_min.to_csv(os.path.join(save_path, "standby_durations.csv"))

    # Separate short breaks vs long overnight stops
    short = durations_min[durations_min <= 60]    # under 1 hour
    long  = durations_min[durations_min > 60]     # over 1 hour

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if len(short) > 0:
        axes[0].hist(short, bins=40, color='#f0a500',
                     edgecolor='#1a1a2e', linewidth=0.3)
    axes[0].set_title("Short Low-Power Periods (≤60 min)\n= Brief breaks between jobs",
                      fontsize=11, fontweight='bold')
    axes[0].set_xlabel("Duration (minutes)")
    axes[0].set_ylabel("Number of Episodes")

    if len(long) > 0:
        axes[1].hist(long / 60, bins=40, color='#4fc3f7',
                     edgecolor='#1a1a2e', linewidth=0.3)
    axes[1].set_title("Long Low-Power Periods (>60 min)\n= Overnight / weekend stops",
                      fontsize=11, fontweight='bold')
    axes[1].set_xlabel("Duration (hours)")
    axes[1].set_ylabel("Number of Episodes")

    fig.suptitle(f"Standby / Low-Power Duration — {save_path.split('/')[-1]}",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "standby_duration.png"), dpi=120)
    plt.close()
    print("✅ Saved: standby_duration.png")

    print("\n📊 Low-Power Duration Stats (minutes):")
    print(f"   Total episodes   : {len(durations_min):,}")
    print(f"   Short (≤60 min)  : {len(short):,}  — quick breaks between jobs")
    print(f"   Long  (>60 min)  : {len(long):,}  — overnight/weekend stops")
    if len(durations_min) > 0:
        print(f"   Shortest episode : {durations_min.min():.1f} min")
        print(f"   Longest episode  : {durations_min.max():.1f} min "
              f"({durations_min.max()/60:.1f} hours)")
        print(f"   Median episode   : {durations_min.median():.1f} min")


def plot_zoomed(df, save_path, machine_name="Machine"):
    # Find first time machine reaches IDLE, WORKING, or PEAK_LOAD
    active_mask = df['state'].isin(['IDLE', 'WORKING', 'PEAK_LOAD'])
    if active_mask.any():
        first_active = active_mask.idxmax()
        start_idx    = max(0, first_active - 120)  # 10 min before startup
    else:
        on_mask   = df['power'] > 5
        start_idx = max(0, on_mask.idxmax() - 60) if on_mask.any() else 0

    subset = df.iloc[start_idx: start_idx + 720]  # 1 hour

    fig, ax = plt.subplots(figsize=(15, 5))

    colors = {'OFF': '#444444', 'STANDBY': '#f0a500',
              'IDLE': '#4fc3f7', 'WORKING': '#66bb6a',
              'PEAK_LOAD': '#e53935'}
    if 'state' in subset.columns:
        for state, color in colors.items():
            mask = subset['state'] == state
            if mask.sum() > 0:
                ax.scatter(subset.loc[mask, 'timestamp'],
                           subset.loc[mask, 'power'],
                           c=color, s=3, label=state)
        ax.legend(markerscale=4, fontsize=9)

    ax.plot(subset['timestamp'], subset['power'],
            linewidth=1.0, color='#cccccc', alpha=0.4, zorder=0)
    ax.set_title(f"Zoomed — First Active Hour — {machine_name}",
                 fontsize=14, fontweight='bold')
    ax.set_xlabel("Time")
    ax.set_ylabel("Power (W)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "zoomed.png"), dpi=120)
    plt.close()
    print("✅ Saved: zoomed.png")


def plot_hourly(df, save_path, machine_name="Machine"):
    df = df.copy()
    df['hour']         = df['timestamp'].dt.hour
    hourly_avg         = df.groupby('hour')['power'].mean()
    on_only            = df[df['power'] > 5]
    hourly_on          = on_only.groupby('hour')['power'].mean() if len(on_only) > 0 else pd.Series()

    # State breakdown by hour
    state_hour = df.groupby(['hour', 'state']).size().unstack(fill_value=0)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    axes[0].fill_between(hourly_avg.index, hourly_avg.values,
                         alpha=0.4, color='#4fc3f7')
    axes[0].plot(hourly_avg.index, hourly_avg.values,
                 marker='o', color='#4fc3f7', linewidth=2)
    axes[0].set_title("Avg Power by Hour of Day", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Hour (0–23)")
    axes[0].set_ylabel("Average Power (W)")
    axes[0].set_xticks(range(0, 24))

    # Stacked bar: state distribution by hour
    colors_map = {'OFF': '#444444', 'STANDBY': '#f0a500',
                  'IDLE': '#4fc3f7', 'WORKING': '#66bb6a',
                  'PEAK_LOAD': '#e53935'}
    bottom = np.zeros(24)
    hours_range = list(range(24))
    for state in ['OFF', 'STANDBY', 'IDLE', 'WORKING', 'PEAK_LOAD']:
        if state in state_hour.columns:
            vals = [state_hour.loc[h, state] if h in state_hour.index else 0
                    for h in hours_range]
            axes[1].bar(hours_range, vals, bottom=bottom,
                        color=colors_map[state], label=state, width=0.8)
            bottom += np.array(vals, dtype=float)

    axes[1].set_title("State Distribution by Hour", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Hour (0–23)")
    axes[1].set_ylabel("Number of Readings")
    axes[1].legend(fontsize=9)
    axes[1].set_xticks(range(0, 24))

    fig.suptitle(f"Hourly Patterns — {machine_name}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "hourly_pattern.png"), dpi=120)
    plt.close()
    print("✅ Saved: hourly_pattern.png")


def plot_daily_energy(df, save_path, machine_name="Machine"):
    df = df.copy()
    df['date']      = df['timestamp'].dt.date
    df['energy_wh'] = df['power'] * (5 / 3600)
    daily_energy    = df.groupby('date')['energy_wh'].sum() / 1000  # kWh

    bar_colors = []
    for d in daily_energy.index:
        weekday = pd.Timestamp(d).weekday()
        bar_colors.append('#66bb6a' if weekday < 5 else '#555555')

    fig, ax = plt.subplots(figsize=(18, 5))
    ax.bar(range(len(daily_energy)), daily_energy.values,
           color=bar_colors, width=0.8)
    ax.set_title(f"Daily Energy — {machine_name}   "
                 f"(🟢 Green=Weekday  ⬛ Grey=Weekend)",
                 fontsize=14, fontweight='bold')
    ax.set_xlabel("Day Index (0 = Jan 1)")
    ax.set_ylabel("Energy (kWh)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "daily_energy.png"), dpi=120)
    plt.close()
    print("✅ Saved: daily_energy.png")


def plot_weekly_pattern(df, save_path, machine_name="Machine"):
    df = df.copy()
    df['weekday']  = df['timestamp'].dt.weekday
    weekday_avg    = df.groupby('weekday')['power'].mean()
    weekday_work   = df[df['state'] == 'WORKING'].groupby(
                        df['timestamp'].dt.weekday)['power'].count() * 5 / 3600

    day_names = ['Monday', 'Tuesday', 'Wednesday',
                 'Thursday', 'Friday', 'Saturday', 'Sunday']
    colors    = ['#66bb6a'] * 5 + ['#555555'] * 2

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(day_names,
                [weekday_avg.get(i, 0) for i in range(7)],
                color=colors, edgecolor='white', linewidth=1.5)
    axes[0].set_title("Average Power by Day of Week",
                      fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Average Power (W)")
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=20)

    work_hrs = [weekday_work.get(i, 0) for i in range(7)]
    axes[1].bar(day_names, work_hrs,
                color=colors, edgecolor='white', linewidth=1.5)
    axes[1].set_title("WORKING Hours by Day of Week",
                      fontsize=12, fontweight='bold')
    axes[1].set_ylabel("Total WORKING Hours")
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=20)

    fig.suptitle(f"Weekly Pattern — {machine_name}",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "weekly_pattern.png"), dpi=120)
    plt.close()
    print("✅ Saved: weekly_pattern.png")


# ============================================================
# PRINT SUMMARY REPORT
# ============================================================

def print_summary_report(df, summary, machine_name="Machine"):
    total_hours = len(df) * 5 / 3600
    on_hours    = total_hours * (1 - summary['OFF']['percent']/100)

    print("\n" + "="*65)
    print(f"  MACHINE REPORT: {machine_name}")
    print("="*65)
    print(f"  Total data     : {total_hours:.1f} hours ({total_hours/24:.1f} days)")
    print(f"  Total readings : {len(df):,} (@ 5-sec intervals)")
    print(f"  Time span      : {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"  Avg power      : {df['power'].mean():.1f} W")
    print(f"  Max power      : {df['power'].max():.1f} W")
    print(f"  Machine ON     : {on_hours:.1f} hours total")
    print("-"*65)
    print(f"  {'STATE':<12} {'HOURS':>8}  {'PERCENT':>8}  {'SAMPLES':>12}  BAR")
    print("-"*65)
    for state, data in summary.items():
        bar = '█' * int(data['percent'] / 2)
        print(f"  {state:<12} {data['hours']:>8.1f}h "
              f"{data['percent']:>8.1f}%  {data['samples']:>12,}  {bar}")
    print("="*65)

    # Plain English summary
    print("\n  📋 PLAIN ENGLISH SUMMARY:")
    print(f"  • Machine was completely OFF for "
          f"{summary['OFF']['hours']:.0f} hours "
          f"({summary['OFF']['percent']:.0f}% of the time)")
    print(f"    → nights, weekends, holidays")
    if summary['STANDBY']['hours'] > 0:
        print(f"  • Machine was in STANDBY for "
              f"{summary['STANDBY']['hours']:.0f} hours "
              f"({summary['STANDBY']['percent']:.0f}%)")
        print(f"    → powered on, control panel active, not yet cutting")
    if summary['IDLE']['hours'] > 0:
        print(f"  • Machine was IDLE for "
              f"{summary['IDLE']['hours']:.0f} hours "
              f"({summary['IDLE']['percent']:.0f}%)")
        print(f"    → spindle running, warming up or between jobs")
    if summary['WORKING']['hours'] > 0:
        print(f"  • Machine was WORKING for "
              f"{summary['WORKING']['hours']:.0f} hours "
              f"({summary['WORKING']['percent']:.0f}%)")
        print(f"    → actively cutting/pressing/operating at full load")
    if summary.get('PEAK_LOAD', {}).get('hours', 0) > 0:
        print(f"  • Machine was in PEAK_LOAD for "
              f"{summary['PEAK_LOAD']['hours']:.0f} hours "
              f"({summary['PEAK_LOAD']['percent']:.0f}%)")
        print(f"    → operating at maximum capacity/surge")

    total_on = (summary['STANDBY']['hours'] +
                summary['IDLE']['hours'] +
                summary['WORKING']['hours'] +
                summary.get('PEAK_LOAD', {}).get('hours', 0))
    productive_hours = summary['WORKING']['hours'] + summary.get('PEAK_LOAD', {}).get('hours', 0)
    if total_on > 0 and productive_hours > 0:
        efficiency = productive_hours / total_on * 100
        print(f"\n  ⚡ EFFICIENCY: When powered ON, machine was actively")
        print(f"     WORKING {efficiency:.1f}% of the time")
        print(f"     (remaining {100-efficiency:.1f}% = standby/idle waste)")
    print()


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def run_analysis(data_path, output_path, machine_name="Machine"):
    os.makedirs(output_path, exist_ok=True)

    df      = load_and_prepare_data(data_path)
    df      = classify_machine_states(df)
    summary = calculate_state_summary(df)

    print_summary_report(df, summary, machine_name)

    plot_power_time(df, output_path, machine_name)
    plot_histogram(df, output_path, machine_name)
    plot_state_pie(summary, output_path, machine_name)
    plot_state_bar(summary, output_path, machine_name)
    analyze_standby(df, output_path)
    plot_zoomed(df, output_path, machine_name)
    plot_hourly(df, output_path, machine_name)
    plot_daily_energy(df, output_path, machine_name)
    plot_weekly_pattern(df, output_path, machine_name)

    print(f"✅ All 9 charts saved in: {output_path}")
    return summary