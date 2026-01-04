import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List

# Set pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    """Configuration parameters for the analysis"""
    TICKER = "BTC-USD"
    START_DATE = "2025-01-01"
    END_DATE = "2026-01-03"
    N_STATES = 5  # Bullish Low Vol, Bullish High Vol, Bearish Low Vol, Bearish High Vol, Sideways
    AUC_WINDOW = 14
    FORECAST_STEPS = 5  # How many steps ahead to propagate
    RECENT_WINDOW = 60  # For visualization


# ============================================================================
# DATA FUNCTIONS
# ============================================================================
def download_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Download financial data from Yahoo Finance"""
    print(f"📥 Downloading data for {ticker} from {start_date} to {end_date}...")
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)

    if len(data) == 0:
        raise ValueError("No data downloaded. Check ticker or date range.")

    return data.copy()


def calculate_rolling_auc(series: pd.Series, window: int) -> pd.Series:
    """Calculate rolling Area Under Curve (AUC)"""
    return series.rolling(window).apply(lambda x: np.sum(x - x[0]), raw=True)


def create_features(df: pd.DataFrame, auc_window: int) -> pd.DataFrame:
    """Create engineered features for the model"""
    df = df.copy()

    # Feature 1: LOG RETURNS
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))

    # Feature 2: LOG RANGE (Volatility)
    df['Log_Range'] = np.log(df['High'] / df['Low'])

    # Feature 3: ROLLING AUC
    df['Rolling_AUC'] = calculate_rolling_auc(df['Close'], auc_window)

    df.dropna(inplace=True)

    return df


# ============================================================================
# MODEL FUNCTIONS
# ============================================================================
def prepare_features(df: pd.DataFrame, feature_columns: List[str]) -> Tuple[np.ndarray, StandardScaler]:
    """Scale and prepare features for HMM training"""
    X = df[feature_columns].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler


def train_hmm(X_scaled: np.ndarray, n_states: int) -> Tuple[GaussianHMM, np.ndarray]:
    """Train Gaussian HMM model"""
    print("🤖 Fitting Gaussian HMM...")
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=1000,
        random_state=42,
        init_params="stmc"  # Initialize start, transition, means, covariance
    )
    model.fit(X_scaled)

    hidden_states = model.predict(X_scaled)

    return model, hidden_states


# ============================================================================
# REGIME LABELING FUNCTIONS - HIERARCHICAL CLUSTERING
# ============================================================================
def label_regimes_clustering(state_stats: pd.DataFrame, transition_matrix: np.ndarray,
                             n_states: int) -> Tuple[Dict[int, str], np.ndarray]:
    """
    Label HMM states using hierarchical clustering for guaranteed unique labels.

    Strategy:
    1. Create a feature matrix from state characteristics
    2. Apply hierarchical clustering with n_clusters = n_states
    3. Label each cluster based on its centroid characteristics
    4. Each state gets exactly one unique cluster assignment
    """

    # Extract transition matrix properties
    persistence = np.diag(transition_matrix)
    avg_duration = np.array([1 / (1 - p) if p < 0.999 else 999 for p in persistence])

    # Calculate exit probabilities (sum of transitions to other states)
    exit_probs = 1 - persistence

    state_stats['Persistence'] = persistence
    state_stats['Avg_Duration'] = avg_duration
    state_stats['Exit_Prob'] = exit_probs

    # Create feature matrix for clustering
    # Normalize features to put them on comparable scales
    clustering_features = state_stats[['Log_Ret', 'Log_Range', 'Rolling_AUC', 'Persistence']].copy()

    # Standardize features for clustering
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(clustering_features)

    print("\n📊 HIERARCHICAL CLUSTERING REGIME ANALYSIS:")
    print("-" * 80)
    print("Clustering features: Log_Ret, Log_Range, Rolling_AUC, Persistence")

    # Perform hierarchical clustering
    clustering = AgglomerativeClustering(n_clusters=n_states, linkage='ward')
    cluster_labels = clustering.fit_predict(features_scaled)

    state_stats['Cluster'] = cluster_labels

    # For each cluster, determine its semantic label based on characteristics
    labels = {}
    cluster_summaries = []

    for cluster_id in range(n_states):
        # Get states in this cluster
        cluster_states = state_stats[state_stats['Cluster'] == cluster_id]

        # Calculate cluster centroid characteristics
        centroid_ret = cluster_states['Log_Ret'].mean()
        centroid_vol = cluster_states['Log_Range'].mean()
        centroid_auc = cluster_states['Rolling_AUC'].mean()
        centroid_persist = cluster_states['Persistence'].mean()

        # Classify direction
        if centroid_ret > 0.001:
            direction = 'Bullish'
        elif centroid_ret < -0.001:
            direction = 'Bearish'
        else:
            direction = 'Sideways'

        # Classify volatility relative to all states
        vol_rank = (state_stats['Log_Range'] < centroid_vol).sum()
        if vol_rank <= n_states * 0.33:
            volatility = 'Low Vol'
        elif vol_rank <= n_states * 0.67:
            volatility = 'Medium Vol'
        else:
            volatility = 'High Vol'

        # Classify persistence
        if centroid_persist > 0.75:
            stability = 'Stable'
        elif centroid_persist < 0.5:
            stability = 'Transitional'
        else:
            stability = 'Normal'

        # Classify momentum
        if centroid_auc > state_stats['Rolling_AUC'].median():
            momentum = 'Strong Momentum'
        else:
            momentum = 'Weak Momentum'

        # Construct semantic label
        base_label = f"{direction} {volatility}"

        # Add stability qualifier if extreme
        if stability == 'Transitional':
            full_label = f"{base_label} (Transitional)"
        elif stability == 'Stable' and direction != 'Sideways':
            full_label = f"{base_label} (Stable)"
        else:
            full_label = base_label

        # Assign label to all states in this cluster
        for state_id in cluster_states['State'].values:
            labels[state_id] = full_label

        cluster_summaries.append({
            'Cluster': cluster_id,
            'Label': full_label,
            'States': list(cluster_states['State'].values),
            'Avg_Return': centroid_ret,
            'Avg_Vol': centroid_vol,
            'Avg_Persistence': centroid_persist
        })

    # Check for duplicate labels and refine if needed
    label_values = list(labels.values())
    if len(label_values) != len(set(label_values)):
        print("⚠️  Detected duplicate labels after clustering, refining...")

        # Add magnitude qualifiers based on return strength within each label group
        label_groups = {}
        for state_id, label in labels.items():
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(state_id)

        for label, state_ids in label_groups.items():
            if len(state_ids) > 1:
                # Sort by return magnitude
                returns = [(sid, state_stats[state_stats['State'] == sid]['Log_Ret'].values[0])
                           for sid in state_ids]
                returns_sorted = sorted(returns, key=lambda x: abs(x[1]), reverse=True)

                qualifiers = ['Strong', 'Moderate', 'Weak', 'Very Weak', 'Extremely Weak']
                for i, (state_id, _) in enumerate(returns_sorted):
                    if i < len(qualifiers):
                        labels[state_id] = f"{qualifiers[i]} {label}"
                    else:
                        labels[state_id] = f"{label} (State {state_id})"

    # Verify uniqueness
    label_values = list(labels.values())
    assert len(label_values) == len(set(label_values)), "ERROR: Labels are not unique!"

    # Print cluster analysis
    print("\nCluster Summary:")
    for summary in cluster_summaries:
        print(f"\nCluster {summary['Cluster']}: {summary['Label']}")
        print(f"  States: {summary['States']}")
        print(f"  Avg Return: {summary['Avg_Return']:.6f}")
        print(f"  Avg Volatility: {summary['Avg_Vol']:.6f}")
        print(f"  Avg Persistence: {summary['Avg_Persistence']:.4f}")

    print("\n" + "-" * 80)
    analysis_df = state_stats[['State', 'Log_Ret', 'Log_Range', 'Rolling_AUC',
                               'Persistence', 'Avg_Duration', 'Cluster']].copy()
    analysis_df['Assigned_Label'] = analysis_df['State'].map(labels)
    print(analysis_df.to_string(index=False))
    print("\n✅ All labels are unique (via hierarchical clustering)!")

    return labels, features_scaled


def plot_dendrogram(features_scaled: np.ndarray, state_stats: pd.DataFrame, labels: Dict[int, str]) -> None:
    """Plot hierarchical clustering dendrogram"""
    plt.figure(figsize=(12, 6))

    # Create linkage matrix
    linkage_matrix = linkage(features_scaled, method='ward')

    # Create dendrogram
    dendrogram(
        linkage_matrix,
        labels=[f"State {i}: {labels[i]}" for i in range(len(labels))],
        leaf_font_size=10,
        leaf_rotation=45
    )

    plt.title('Hierarchical Clustering Dendrogram of HMM States', fontsize=14, fontweight='bold')
    plt.xlabel('State', fontsize=12)
    plt.ylabel('Distance', fontsize=12)
    plt.tight_layout()
    plt.show()


# ============================================================================
# FORWARD ALGORITHM FUNCTIONS
# ============================================================================
def compute_stationary_distribution(transition_matrix: np.ndarray, max_iter: int = 1000,
                                    tol: float = 1e-10) -> np.ndarray:
    """Compute stationary distribution of a transition matrix"""
    n_states = transition_matrix.shape[0]
    pi = np.ones(n_states) / n_states

    for _ in range(max_iter):
        pi_new = pi @ transition_matrix
        if np.allclose(pi, pi_new, atol=tol):
            break
        pi = pi_new

    return pi


def forward_algorithm_stable(model: GaussianHMM, observations: np.ndarray) -> np.ndarray:
    """Robust forward algorithm using scaling to prevent numerical underflow"""
    n_samples = len(observations)
    n_states = model.n_components

    # Use stationary distribution as initial probabilities
    initial_probs = compute_stationary_distribution(model.transmat_)
    initial_probs = np.maximum(initial_probs, 1e-10)
    initial_probs = initial_probs / initial_probs.sum()

    # Compute emission probabilities
    log_likelihood = model._compute_log_likelihood(observations)
    emission_probs = np.exp(log_likelihood)
    emission_probs = np.maximum(emission_probs, 1e-300)

    # Initialize forward variables
    alpha = np.zeros((n_samples, n_states))
    scale_factors = np.zeros(n_samples)

    # Initialization (t=0)
    alpha[0] = initial_probs * emission_probs[0]
    scale_factors[0] = alpha[0].sum()
    if scale_factors[0] > 0:
        alpha[0] = alpha[0] / scale_factors[0]
    else:
        alpha[0] = initial_probs

    # Forward recursion with scaling
    for t in range(1, n_samples):
        for j in range(n_states):
            alpha[t, j] = np.sum(alpha[t - 1] * model.transmat_[:, j]) * emission_probs[t, j]

        scale_factors[t] = alpha[t].sum()
        if scale_factors[t] > 0:
            alpha[t] = alpha[t] / scale_factors[t]
        else:
            alpha[t] = alpha[t - 1] @ model.transmat_
            alpha[t] = alpha[t] / alpha[t].sum()

    return alpha


def propagate_transitions(current_probs: np.ndarray, transition_matrix: np.ndarray, n_steps: int) -> np.ndarray:
    """Propagate current state probabilities forward using transition matrix"""
    forecast = np.zeros((n_steps, len(current_probs)))
    forecast[0] = current_probs

    for t in range(1, n_steps):
        forecast[t] = forecast[t - 1] @ transition_matrix

    return forecast


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================
def plot_results(df: pd.DataFrame, forward_probs: np.ndarray, trans_df: pd.DataFrame,
                 forecast_df: pd.DataFrame, labels: Dict[int, str], config: Config) -> None:
    """Create comprehensive visualization plots"""
    sns.set_theme(style="darkgrid")
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)

    # Plot 1: Price with Regimes
    ax1 = fig.add_subplot(gs[0, :])
    for state in range(config.N_STATES):
        idx = df[df['Regime'] == state].index
        ax1.scatter(idx, df.loc[idx, 'Close'], label=labels[state], s=10, alpha=0.7)

    ax1.set_title(f"{config.TICKER} Price Action Classified by HMM Regime (Hierarchical Clustering)",
                  fontsize=14, fontweight='bold')
    ax1.set_ylabel("Price")
    ax1.legend(loc='upper left', markerscale=3, ncol=5)
    ax1.grid(alpha=0.3)

    # Plot 2: Volatility
    ax2 = fig.add_subplot(gs[1, :])
    median_vol = df['Log_Range'].median()
    ax2.plot(df.index, df['Log_Range'], color='purple', alpha=0.6, linewidth=1)
    ax2.axhline(median_vol, color='red', linestyle='--', label=f'Median Vol', alpha=0.7)
    ax2.set_title("Market Volatility (Log Range)", fontsize=12)
    ax2.set_ylabel("Log Volatility")
    ax2.legend()
    ax2.grid(alpha=0.3)

    # Plot 3: Forward Probabilities (Recent)
    ax3 = fig.add_subplot(gs[2, :])
    for i in range(config.N_STATES):
        ax3.plot(df.index[-config.RECENT_WINDOW:], forward_probs[-config.RECENT_WINDOW:, i],
                 label=labels[i], linewidth=2, alpha=0.8)
    ax3.set_title(f"Forward Algorithm State Probabilities (Last {config.RECENT_WINDOW} Days)", fontsize=12)
    ax3.set_ylabel("Probability")
    ax3.set_ylim([0, 1])
    ax3.legend(loc='best')
    ax3.grid(alpha=0.3)

    # Plot 4: Transition Matrix Heatmap
    ax4 = fig.add_subplot(gs[3, 0])
    sns.heatmap(trans_df, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax4,
                cbar_kws={'label': 'Transition Probability'})
    ax4.set_title("Transition Matrix Heatmap", fontsize=12, fontweight='bold')
    ax4.set_xlabel("To State")
    ax4.set_ylabel("From State")

    # Plot 5: Future State Probabilities
    ax5 = fig.add_subplot(gs[3, 1])
    forecast_df.T.plot(kind='bar', ax=ax5, width=0.8, colormap='viridis')
    ax5.set_title("Forecasted State Probabilities", fontsize=12, fontweight='bold')
    ax5.set_xlabel("Regime")
    ax5.set_ylabel("Probability")
    ax5.legend(title="Forecast Day", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax5.set_xticklabels(ax5.get_xticklabels(), rotation=45, ha='right')
    ax5.grid(axis='y', alpha=0.3)

    plt.show()


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================
def print_regime_distribution(df: pd.DataFrame) -> None:
    """Print regime distribution statistics"""
    print("\n" + "=" * 80)
    print("REGIME DISTRIBUTION")
    print("=" * 80)
    regime_counts = df['Regime_Label'].value_counts()
    regime_pcts = df['Regime_Label'].value_counts(normalize=True) * 100
    regime_summary = pd.DataFrame({
        'Count': regime_counts,
        'Percentage': regime_pcts
    })
    print(regime_summary.round(2))


def print_returns_by_regime(df: pd.DataFrame) -> None:
    """Print returns analysis by regime"""
    print("\n" + "=" * 80)
    print("AVERAGE RETURNS BY REGIME")
    print("=" * 80)
    returns_by_regime = df.groupby('Regime_Label')['Log_Ret'].agg(['mean', 'std', 'count'])
    returns_by_regime['annualized_return'] = returns_by_regime['mean'] * 252
    returns_by_regime['annualized_vol'] = returns_by_regime['std'] * np.sqrt(252)
    print(returns_by_regime.round(4))


def print_stationary_distribution(model: GaussianHMM, labels: Dict[int, str]) -> None:
    """Print long-term stationary distribution"""
    print("\n" + "=" * 80)
    print("CONVERGENCE TO STATIONARY DISTRIBUTION")
    print("=" * 80)
    stationary_dist = compute_stationary_distribution(model.transmat_)
    stat_df = pd.DataFrame({
        'Regime': [labels[i] for i in range(len(labels))],
        'Long-term Probability': stationary_dist
    }).sort_values('Long-term Probability', ascending=False)
    print(stat_df.to_string(index=False))


def print_transition_matrix_analysis(model: GaussianHMM, labels: Dict[int, str]) -> None:
    """Print analysis of the transition matrix"""
    print("\n" + "=" * 80)
    print("FULL TRANSITION MATRIX - State-to-State Probabilities")
    print("=" * 80)

    trans_df = pd.DataFrame(
        model.transmat_,
        index=[f"{labels[i]}" for i in range(model.n_components)],
        columns=[f"{labels[i]}" for i in range(model.n_components)]
    )
    print(trans_df.round(4))

    # Persistence probabilities
    print("\nRegime Persistence Probabilities:")
    for i in range(model.n_components):
        persistence = model.transmat_[i, i]
        avg_duration = 1 / (1 - persistence) if persistence < 0.999 else float('inf')
        print(f"  {labels[i]:40s}: {persistence:.4f} (avg duration: {avg_duration:.1f} days)")

    # Most likely transitions
    print("\nMost Likely Regime Transitions (excluding persistence):")
    for i in range(model.n_components):
        trans_probs = model.transmat_[i].copy()
        trans_probs[i] = 0
        most_likely_next = np.argmax(trans_probs)
        print(f"  {labels[i]:40s} → {labels[most_likely_next]:40s}: {trans_probs[most_likely_next]:.4f}")


# ============================================================================
# MAIN FUNCTION
# ============================================================================
def main():
    """Main function to run the HMM market regime analysis"""
    print("=" * 80)
    print(f"TRAINING MULTIVARIATE HMM FOR {Config.TICKER}")
    print(f"Regime Labeling: HIERARCHICAL CLUSTERING")
    print("=" * 80)

    # 1. Data ingestion & feature engineering
    data = download_data(Config.TICKER, Config.START_DATE, Config.END_DATE)
    df = create_features(data, Config.AUC_WINDOW)

    print(f"Training on {len(df)} days of data.")
    print(f"Features: ['Log_Ret', 'Log_Range', 'Rolling_AUC']")

    # 2. Preprocessing
    X_scaled, scaler = prepare_features(df, ['Log_Ret', 'Log_Range', 'Rolling_AUC'])

    # 3. Model training
    model, hidden_states = train_hmm(X_scaled, Config.N_STATES)
    df['Regime'] = hidden_states

    # 4. Regime labeling (using hierarchical clustering)
    real_means = scaler.inverse_transform(model.means_)
    state_stats = pd.DataFrame(real_means, columns=['Log_Ret', 'Log_Range', 'Rolling_AUC'])
    state_stats['State'] = range(Config.N_STATES)

    labels, features_scaled = label_regimes_clustering(state_stats, model.transmat_, Config.N_STATES)
    df['Regime_Label'] = df['Regime'].map(labels)

    # 4.5. Plot dendrogram
    plot_dendrogram(features_scaled, state_stats, labels)

    # 5. Forward algorithm
    print("\n" + "=" * 80)
    print("FORWARD ALGORITHM - Computing State Probabilities")
    print("=" * 80)

    forward_probs = forward_algorithm_stable(model, X_scaled)

    # Add forward probabilities to dataframe
    for i in range(Config.N_STATES):
        df[f'P_{labels[i]}'] = forward_probs[:, i]

    print("\nForward probabilities computed for all time steps.")
    print("Sample of recent forward probabilities:")
    prob_cols = [f'P_{labels[i]}' for i in range(Config.N_STATES)]
    print(df[['Close', 'Regime_Label'] + prob_cols].tail(10).round(3))

    # 6. Transition propagation
    print("\n" + "=" * 80)
    print("TRANSITION PROPAGATION - Forecasting Future State Probabilities")
    print("=" * 80)

    current_state_probs = forward_probs[-1]
    future_probs = propagate_transitions(current_state_probs, model.transmat_, Config.FORECAST_STEPS)

    forecast_df = pd.DataFrame(future_probs, columns=[labels[i] for i in range(Config.N_STATES)])
    forecast_df.index = [f"Day +{i + 1}" for i in range(Config.FORECAST_STEPS)]

    print(f"\nCurrent State Probabilities (Latest Observation):")
    for i in range(Config.N_STATES):
        print(f"  {labels[i]:40s}: {current_state_probs[i]:.4f}")

    print(f"\nForecasted State Probabilities (Next {Config.FORECAST_STEPS} Days):")
    print(forecast_df.round(4))

    # Most likely future regime
    most_likely = forecast_df.idxmax(axis=1)
    print("\nMost Likely Regime by Day:")
    for day, regime in most_likely.items():
        prob = forecast_df.loc[day, regime]
        if isinstance(prob, pd.Series):
            prob = prob.iloc[0]
        print(f"  {day}: {regime} ({prob:.2%})")

    # 7. Transition matrix analysis
    print_transition_matrix_analysis(model, labels)

    # 8. Create transition matrix dataframe for visualization
    trans_df = pd.DataFrame(
        model.transmat_,
        index=[f"{labels[i]}" for i in range(Config.N_STATES)],
        columns=[f"{labels[i]}" for i in range(Config.N_STATES)]
    )

    # 9. Visualization
    plot_results(df, forward_probs, trans_df, forecast_df, labels, Config)

    # 10. Summary statistics
    print_regime_distribution(df)
    print_returns_by_regime(df)
    print_stationary_distribution(model, labels)


if __name__ == "__main__":
    main()