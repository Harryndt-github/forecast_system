"""
==============================================
TRANSFER LEARNING AGENT
==============================================
Chia sẻ kiến thức giữa các nhà hàng tương tự.

Addresses Limitation #4: Không có transfer learning.

Approach:
1. Cluster restaurants by behavior patterns (volume, weekday shape, trend)
2. For NEW/YOUNG restaurants with < 30 days data:
   - Find cluster siblings (similar restaurants)
   - Use cluster average bias/corrections as prior
3. Share learned patterns within clusters
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from forecast_system.config.settings import PROJECT_ROOT
from forecast_system.utils.logger import get_logger

logger = get_logger('transfer_learning')

# Check sklearn
try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    HAS_CLUSTER = True
except ImportError:
    HAS_CLUSTER = False


class TransferLearningAgent:
    """
    Transfer knowledge between similar restaurants.
    Uses clustering to find restaurant "siblings" and share corrections.
    """

    CLUSTER_FILE = PROJECT_ROOT / 'restaurant_clusters.json'
    N_CLUSTERS = 8
    MIN_DATA_DAYS = 14  # Min days for a restaurant to be clustered

    @staticmethod
    def build_clusters(df_train: pd.DataFrame, analysis_reports: Dict) -> Dict:
        """
        Cluster restaurants based on behavior patterns.

        Features per restaurant:
        - avg_daily_guests (volume level)
        - weekday_pattern (7 values: Mon-Sun relative proportions)
        - weekend_ratio (Sat+Sun avg / weekday avg)
        - volatility (std/mean of daily guests)
        - trend_score (from analysis)
        """
        if not HAS_CLUSTER:
            logger.warning("sklearn not available, skipping clustering")
            return {}

        logger.info("🔗 Building restaurant clusters for Transfer Learning...")

        # Compute features per restaurant
        features = {}
        restaurant_codes = []

        for res_code, df_res in df_train.groupby('restaurant_code'):
            daily = df_res.groupby('date')['guest_count'].sum()

            if len(daily) < TransferLearningAgent.MIN_DATA_DAYS:
                continue

            avg_daily = daily.mean()
            if avg_daily < 1:
                continue

            # Weekday pattern (relative proportions)
            df_res_copy = df_res.copy()
            df_res_copy['weekday'] = pd.to_datetime(df_res_copy['date']).dt.dayofweek
            wd_avg = df_res_copy.groupby(['date', 'weekday'])['guest_count'].sum().reset_index()
            wd_pattern = wd_avg.groupby('weekday')['guest_count'].mean()

            # Normalize to proportions
            wd_total = wd_pattern.sum()
            wd_props = [wd_pattern.get(i, 0) / max(wd_total, 1) for i in range(7)]  # type: ignore[reportOptionalOperand]

            # Weekend ratio
            weekday_avg = np.mean([wd_pattern.get(i, 0) for i in range(5)])  # type: ignore[reportArgumentType, reportCallIssue]
            weekend_avg = np.mean([wd_pattern.get(i, 0) for i in [5, 6]])  # type: ignore[reportArgumentType, reportCallIssue]
            weekend_ratio = weekend_avg / max(weekday_avg, 1)

            # Volatility
            volatility = daily.std() / max(daily.mean(), 1)

            # Trend score
            report = analysis_reports.get(str(res_code), {})
            trend_score = report.get('trend_score', 0)

            feat = [avg_daily] + wd_props + [weekend_ratio, volatility, trend_score]
            features[str(res_code)] = feat
            restaurant_codes.append(str(res_code))

        if len(features) < TransferLearningAgent.N_CLUSTERS * 2:
            logger.info(f"   Not enough restaurants ({len(features)}) for clustering")
            return {}

        # Build feature matrix and cluster
        X = np.array([features[rc] for rc in restaurant_codes])
        scaler = StandardScaler()  # type: ignore[reportPossiblyUnboundVariable]
        X_scaled = scaler.fit_transform(X)

        n_clusters = min(TransferLearningAgent.N_CLUSTERS, len(X) // 3)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)  # type: ignore[reportArgumentType, reportPossiblyUnboundVariable]
        labels = kmeans.fit_predict(X_scaled)

        # Build cluster map
        clusters = defaultdict(list)
        restaurant_cluster = {}

        for rc, label in zip(restaurant_codes, labels):
            cluster_id = int(label)
            clusters[cluster_id].append(rc)
            restaurant_cluster[rc] = cluster_id

        # Compute cluster profiles
        cluster_profiles = {}
        for cid, members in clusters.items():
            member_features = [features[m] for m in members]
            avg_feat = np.mean(member_features, axis=0)
            cluster_profiles[cid] = {
                'n_members': len(members),
                'avg_daily_guests': round(float(avg_feat[0]), 1),
                'weekend_ratio': round(float(avg_feat[8]), 2),
                'volatility': round(float(avg_feat[9]), 2),
            }

        result = {
            'n_clusters': n_clusters,
            'n_restaurants': len(restaurant_codes),
            'created_at': str(pd.Timestamp.now()),
            'restaurant_cluster': restaurant_cluster,
            'cluster_profiles': {str(k): v for k, v in cluster_profiles.items()},
            'cluster_members': {str(k): v for k, v in clusters.items()},
        }

        # Save
        with open(TransferLearningAgent.CLUSTER_FILE, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        logger.info(f"   ✅ {n_clusters} clusters from {len(restaurant_codes)} restaurants")
        for cid, profile in cluster_profiles.items():
            logger.info(
                f"      Cluster {cid}: {profile['n_members']} restaurants, "
                f"avg={profile['avg_daily_guests']:.0f}/day, "
                f"weekend_ratio={profile['weekend_ratio']:.2f}"
            )

        return result

    @staticmethod
    def get_transfer_corrections(
        res_code: str,
        brain_memory: Dict = None,  # type: ignore[reportArgumentType]
    ) -> Optional[Dict]:
        """
        Get correction factors from similar restaurants (cluster siblings).

        For NEW/YOUNG restaurants with limited brain memory,
        use the cluster average corrections as a prior.

        Returns:
            Dict with transfer corrections or None
        """
        try:
            if not TransferLearningAgent.CLUSTER_FILE.exists():
                return None

            with open(TransferLearningAgent.CLUSTER_FILE, 'r', encoding='utf-8') as f:
                clusters = json.load(f)

            res_code_str = str(res_code)
            cluster_id = clusters.get('restaurant_cluster', {}).get(res_code_str)

            if cluster_id is None:
                return None

            # Get siblings
            siblings = clusters.get('cluster_members', {}).get(str(cluster_id), [])
            siblings = [s for s in siblings if s != res_code_str]

            if not siblings:
                return None

            # Load brain memory
            if brain_memory is None:
                brain_file = PROJECT_ROOT / 'brain_memory.json'
                if brain_file.exists():
                    with open(brain_file, 'r', encoding='utf-8') as f:
                        brain_memory = json.load(f)
                else:
                    return None

            # Check if this restaurant needs transfer (limited own data)
            res_mem = brain_memory.get('restaurants', {}).get(res_code_str, {})
            own_mape = res_mem.get('last_mape', 0)
            has_own_data = bool(res_mem.get('overall_bias') is not None)

            # Only transfer if restaurant has poor/no own data
            if has_own_data and own_mape < 20:
                return None  # Good own data, no need for transfer

            # Aggregate sibling corrections
            sibling_biases = []
            sibling_corrections = []
            sibling_weekday_bias = defaultdict(list)
            sibling_holiday_bias = []

            for sib in siblings:
                sib_mem = brain_memory.get('restaurants', {}).get(sib, {})
                if not sib_mem:
                    continue

                bias = sib_mem.get('overall_bias', 0)
                cf = sib_mem.get('correction_factor', 1.0)
                hol_bias = sib_mem.get('holiday_bias', 0)

                sibling_biases.append(bias)
                sibling_corrections.append(cf)

                if abs(hol_bias) > 1:
                    sibling_holiday_bias.append(hol_bias)

                for wd, wb in sib_mem.get('weekday_bias', {}).items():
                    sibling_weekday_bias[wd].append(wb)

            if not sibling_biases:
                return None

            # Compute transfer priors (use median for robustness)
            # Blend: 60% own data + 40% cluster average (if own data exists)
            own_weight = 0.6 if has_own_data else 0.2
            transfer_weight = 1.0 - own_weight

            transfer = {
                'cluster_id': cluster_id,
                'n_siblings': len(siblings),
                'n_siblings_with_data': len(sibling_biases),
                'overall_bias': round(
                    own_weight * res_mem.get('overall_bias', 0) +
                    transfer_weight * float(np.median(sibling_biases)),
                    2
                ),
                'correction_factor': round(
                    own_weight * res_mem.get('correction_factor', 1.0) +
                    transfer_weight * float(np.median(sibling_corrections)),
                    3
                ),
                'weekday_bias': {
                    wd: round(float(np.median(vals)), 1)
                    for wd, vals in sibling_weekday_bias.items()
                    if len(vals) >= 3
                },
                'holiday_bias': round(
                    float(np.median(sibling_holiday_bias)), 1
                ) if sibling_holiday_bias else 0,
                'source': 'transfer_learning',
            }

            logger.debug(
                f"🔗 Transfer for {res_code}: cluster={cluster_id}, "
                f"{len(sibling_biases)} siblings → "
                f"bias={transfer['overall_bias']:.1f}, "
                f"cf={transfer['correction_factor']:.3f}"
            )

            return transfer

        except Exception as e:
            logger.debug(f"Transfer learning failed for {res_code}: {e}")
            return None

    @staticmethod
    def load_clusters() -> Optional[Dict]:
        """Load existing clusters from file."""
        if TransferLearningAgent.CLUSTER_FILE.exists():
            try:
                with open(TransferLearningAgent.CLUSTER_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return None
