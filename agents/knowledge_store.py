"""
==============================================
KNOWLEDGE STORE - RAG Memory Engine
==============================================
Vector database để lưu trữ và truy xuất "kinh nghiệm"
dự đoán cho hệ thống forecast.

Collections:
    1. restaurant_patterns   → Weekly/hourly patterns per restaurant
    2. prediction_feedback   → Past predictions + errors + corrections that worked
    3. brain_insights        → Brain memory: biases, issues, strategies

Khi predict:
    → Query similar context → inject vào prompt → LLM predict chính xác hơn to

Self-Learning Loop:
    → Mỗi lần có actual data → store feedback → improve future predictions
"""

import os
import json
import datetime
import hashlib
import traceback
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from forecast_system.config.settings import PROJECT_ROOT
from forecast_system.utils.logger import get_logger

logger = get_logger('knowledge_store')

# ChromaDB import
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False
    logger.warning("ChromaDB not installed. Run: pip install chromadb")

# Sentence Transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    logger.warning(
        "sentence-transformers not installed. "
        "Run: pip install sentence-transformers"
    )


# ==========================================
# KNOWLEDGE STORE
# ==========================================

CHROMA_DB_PATH = str(PROJECT_ROOT / "knowledge_db")

# Collections
COL_PATTERNS = "restaurant_patterns"
COL_FEEDBACK = "prediction_feedback"
COL_INSIGHTS = "brain_insights"


class KnowledgeStore:
    """
    RAG Knowledge Store dùng ChromaDB.
    
    Lưu trữ và truy xuất kinh nghiệm dự đoán để enrich prompts.
    
    Usage:
        store = KnowledgeStore()
        
        # Index kinh nghiệm
        store.index_brain_memory(brain_memory_dict)
        store.index_prediction_feedback(df_master)
        store.index_restaurant_patterns(df_train, res_code)
        
        # Truy xuất context cho prediction
        context = store.retrieve_context(
            res_code="R001",
            target_date=date(2026, 2, 15),
            weekday="Saturday",
            is_holiday=False,
        )
    """
    
    def __init__(self, db_path: str = None):  # type: ignore[reportArgumentType]
        self.db_path = db_path or CHROMA_DB_PATH
        self._client = None
        self._embedder = None
        self._collections = {}
    
    # ==========================================
    # INITIALIZATION
    # ==========================================
    
    def _get_client(self):
        """Lazy init ChromaDB client."""
        if self._client is None:
            if not HAS_CHROMADB:
                raise ImportError("ChromaDB is required. Run: pip install chromadb")
            
            os.makedirs(self.db_path, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self.db_path)  # type: ignore[reportPossiblyUnboundVariable]
            logger.info(f"ChromaDB initialized at {self.db_path}")
        return self._client
    
    def _get_embedder(self):
        """Lazy init sentence transformer for embeddings."""
        if self._embedder is None:
            if not HAS_SENTENCE_TRANSFORMERS:
                raise ImportError(
                    "sentence-transformers required. Run: pip install sentence-transformers"
                )
            # Small, fast model optimized for similarity search
            try:
                # Try offline first (faster, avoids DNS/network errors)
                self._embedder = SentenceTransformer('all-MiniLM-L6-v2', local_files_only=True)  # type: ignore[reportPossiblyUnboundVariable]
            except Exception:
                # Fallback to online download if not cached locally
                self._embedder = SentenceTransformer('all-MiniLM-L6-v2')  # type: ignore[reportPossiblyUnboundVariable]
            logger.info("Sentence embedder loaded: all-MiniLM-L6-v2")
        return self._embedder
    
    def _get_collection(self, name: str):
        """Get or create a collection."""
        if name not in self._collections:
            client = self._get_client()
            self._collections[name] = client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collections[name]
    
    def _embed(self, texts: List[str]) -> List[List[float]]:
        """Embed texts using sentence transformer."""
        embedder = self._get_embedder()
        embeddings = embedder.encode(texts, show_progress_bar=False)
        return embeddings.tolist()
    
    @staticmethod
    def _make_id(*parts) -> str:
        """Create deterministic ID from parts."""
        key = "|".join(str(p) for p in parts)
        return hashlib.md5(key.encode()).hexdigest()[:16]
    
    # ==========================================
    # INDEXING: Brain Memory → Knowledge Store
    # ==========================================
    
    def index_brain_memory(self, brain_memory: Dict):
        """
        Index ForecastBrain memory vào knowledge store.
        
        Chuyển brain_memory.json thành documents có thể search:
        - Per-restaurant bias patterns
        - Global patterns
        - Issues và corrections
        """
        col = self._get_collection(COL_INSIGHTS)
        
        documents = []
        metadatas = []
        ids = []
        
        # Global patterns
        global_p = brain_memory.get('global_patterns', {})
        if global_p:
            doc = (
                f"Global forecast patterns: "
                f"Holiday over-prediction: {global_p.get('holiday_overpredict_pct', 0):.1f}%. "
                f"Weekend bias: {global_p.get('weekend_bias', 0):.1f} guests. "
            )
            weekday_biases = global_p.get('weekday_biases', {})
            if weekday_biases:
                for wd, bias in weekday_biases.items():
                    doc += f"{wd} bias: {bias:+.1f} guests. "
            
            hourly_biases = global_p.get('hourly_biases', {})
            if hourly_biases:
                top_hours = sorted(hourly_biases.items(), key=lambda x: abs(float(x[1])), reverse=True)[:5]
                for h, b in top_hours:
                    doc += f"Hour {h} bias: {float(b):+.1f}. "
            
            documents.append(doc)
            metadatas.append({
                'type': 'global_pattern',
                'restaurant': 'ALL',
                'updated': str(brain_memory.get('last_updated', '')),
            })
            ids.append(self._make_id('global', 'patterns'))
        
        # Per-restaurant patterns
        restaurants = brain_memory.get('restaurants', {})
        for res_code, res_mem in restaurants.items():
            # Bias summary document
            doc = (
                f"Restaurant {res_code} forecast patterns: "
                f"Overall bias: {res_mem.get('overall_bias', 0):+.1f} guests. "
                f"Correction factor: {res_mem.get('correction_factor', 1.0):.3f}. "
                f"Best strategy: {res_mem.get('best_strategy', 'ENSEMBLE_EQUAL')}. "
                f"ML MAPE: {res_mem.get('ml_mape', 'N/A')}%. "
                f"AI MAPE: {res_mem.get('ai_mape', 'N/A')}%. "
            )
            
            # Weekday biases
            wd_bias = res_mem.get('weekday_bias', {})
            if wd_bias:
                doc += "Weekday biases: "
                for wd, bias in wd_bias.items():
                    doc += f"{wd}={bias:+.1f}, "
            
            # Holiday bias
            h_bias = res_mem.get('holiday_bias', 0)
            if h_bias:
                doc += f"Holiday bias: {h_bias:+.1f} guests. "
            
            # MAPE history
            mape_hist = res_mem.get('mape_history', [])
            if mape_hist:
                doc += f"MAPE trend: {mape_hist[-min(5, len(mape_hist)):]}. "
                if len(mape_hist) >= 2:
                    improving = mape_hist[-1] < mape_hist[-2]
                    doc += f"Accuracy {'improving' if improving else 'declining'}. "
            
            documents.append(doc)
            metadatas.append({
                'type': 'restaurant_bias',
                'restaurant': str(res_code),
                'overall_bias': float(res_mem.get('overall_bias', 0)),
                'correction_factor': float(res_mem.get('correction_factor', 1.0)),
                'best_strategy': res_mem.get('best_strategy', ''),
                'updated': str(res_mem.get('learned_at', '')),
            })
            ids.append(self._make_id('bias', res_code))
            
            # Issues as separate documents
            issues = res_mem.get('issues', [])
            for i, issue in enumerate(issues[-10:]):  # Keep latest 10 issues
                iss_doc = (
                    f"Restaurant {res_code} prediction issue: "
                    f"Date {issue.get('date')}, "
                    f"Type: {issue.get('type', 'UNKNOWN')}, "
                    f"Error: {issue.get('error_pct', 0):.0f}%. "
                    f"Predicted: {issue.get('predicted', 'N/A')}, "
                    f"Actual: {issue.get('actual', 'N/A')}. "
                )
                documents.append(iss_doc)
                metadatas.append({
                    'type': 'issue',
                    'restaurant': str(res_code),
                    'issue_type': issue.get('type', ''),
                    'date': str(issue.get('date', '')),
                })
                # Include index + issue type to avoid duplicate IDs
                # for same restaurant + same date
                ids.append(self._make_id(
                    'issue', res_code,
                    issue.get('date', ''),
                    issue.get('type', ''),
                    str(i),
                ))
        
        if documents:
            # Deduplicate IDs (keep last occurrence)
            seen = {}
            for idx, doc_id in enumerate(ids):
                seen[doc_id] = idx
            unique_indices = sorted(seen.values())
            
            if len(unique_indices) < len(ids):
                logger.info(
                    f"Deduplicated {len(ids) - len(unique_indices)} "
                    f"duplicate IDs before upsert"
                )
                ids = [ids[i] for i in unique_indices]
                documents = [documents[i] for i in unique_indices]
                metadatas = [metadatas[i] for i in unique_indices]
            
            embeddings = self._embed(documents)
            col.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            logger.info(
                f"Indexed {len(documents)} brain insights "
                f"({len(restaurants)} restaurants)"
            )
    
    # ==========================================
    # INDEXING: Prediction Feedback
    # ==========================================
    
    def index_prediction_feedback(self, df_master: pd.DataFrame, max_records: int = 2000):
        """
        Index past predictions + actuals → learn what worked and what didn't.
        Tạo feedback documents từ master file.
        """
        col = self._get_collection(COL_FEEDBACK)
        
        # Filter valid records
        mask = (
            pd.notna(df_master.get('Final_Predicted_Guests')) &  # type: ignore[reportOperatorIssue]
            pd.notna(df_master.get('Actual_Guest')) &
            (df_master['Actual_Guest'] >= 0)
        )
        df = df_master[mask].copy()
        
        if df.empty:
            logger.warning("No valid feedback data to index.")
            return
        
        # Aggregate to daily level
        daily = df.groupby(['Restaurant_Code', 'Date']).agg({
            'Final_Predicted_Guests': 'sum',
            'Actual_Guest': 'sum',
            'Weekday': 'first',
            'Is_Holiday': 'first',
        }).reset_index()
        
        daily['error'] = daily['Final_Predicted_Guests'] - daily['Actual_Guest']
        daily['abs_error'] = daily['error'].abs()
        nonzero = daily['Actual_Guest'] > 0
        daily.loc[nonzero, 'pct_error'] = (
            daily.loc[nonzero, 'abs_error'] / daily.loc[nonzero, 'Actual_Guest'] * 100
        )
        
        # Sample if too many
        if len(daily) > max_records:
            # Prioritize high-error records (more to learn from)
            daily = daily.nlargest(max_records, 'abs_error')
        
        documents = []
        metadatas = []
        ids = []
        
        for _, row in daily.iterrows():
            doc = (
                f"Restaurant {row['Restaurant_Code']} on {row['Date']} ({row['Weekday']}): "
                f"Predicted {int(row['Final_Predicted_Guests'])} guests, "
                f"Actual {int(row['Actual_Guest'])} guests. "
                f"Error: {row['error']:+.0f} ({row.get('pct_error', 0):.0f}%). "
            )
            if row.get('Is_Holiday'):
                doc += "This was a HOLIDAY. "
            
            direction = "over-predicted" if row['error'] > 0 else "under-predicted"
            doc += f"Model {direction}. "
            
            documents.append(doc)
            metadatas.append({
                'type': 'feedback',
                'restaurant': str(row['Restaurant_Code']),
                'date': str(row['Date']),
                'weekday': str(row['Weekday']),
                'predicted': float(row['Final_Predicted_Guests']),
                'actual': float(row['Actual_Guest']),
                'error_pct': float(row.get('pct_error', 0)),  # type: ignore[reportArgumentType]
                'is_holiday': bool(row.get('Is_Holiday', False)),
            })
            ids.append(self._make_id('feedback', row['Restaurant_Code'], row['Date']))
        
        if documents:
            embeddings = self._embed(documents)
            col.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            logger.info(f"Indexed {len(documents)} prediction feedback records")
    
    # ==========================================
    # INDEXING: Restaurant Patterns
    # ==========================================
    
    def index_restaurant_patterns(self, df_train: pd.DataFrame, res_code: str):
        """
        Index weekly/hourly patterns cho 1 nhà hàng.
        """
        col = self._get_collection(COL_PATTERNS)
        
        df_res = df_train[df_train['restaurant_code'] == res_code]
        if df_res.empty:
            return
        
        daily = df_res.groupby(['date', 'weekday'])['guest_count'].sum().reset_index()
        
        # Weekday averages
        wd_avg = daily.groupby('weekday')['guest_count'].mean()
        overall_avg = daily['guest_count'].mean()
        active_days = daily['date'].nunique()
        
        doc = (
            f"Restaurant {res_code} patterns: "
            f"Average {overall_avg:.0f} guests/day, {active_days} active days. "
            f"Weekday averages: "
        )
        for wd in ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
                    'Friday', 'Saturday', 'Sunday']:
            if wd in wd_avg.index:
                doc += f"{wd}={wd_avg[wd]:.0f}, "
        
        # Recent trend
        recent_7 = daily.sort_values('date').tail(7)
        recent_avg = recent_7['guest_count'].mean()
        trend_pct = ((recent_avg - overall_avg) / overall_avg * 100) if overall_avg > 0 else 0
        doc += f"Recent 7-day avg: {recent_avg:.0f} ({trend_pct:+.1f}% vs overall). "
        
        col.upsert(
            ids=[self._make_id('pattern', res_code)],
            embeddings=self._embed([doc]),
            documents=[doc],
            metadatas=[{
                'type': 'pattern',
                'restaurant': str(res_code),
                'avg_daily': float(overall_avg),
                'active_days': int(active_days),
            }],
        )
    
    def index_all_restaurant_patterns(self, df_train: pd.DataFrame):
        """Index patterns cho TẤT CẢ nhà hàng."""
        restaurants = df_train['restaurant_code'].unique()
        for res_code in restaurants:
            try:
                self.index_restaurant_patterns(df_train, str(res_code))
            except Exception:
                continue
        logger.info(f"Indexed patterns for {len(restaurants)} restaurants")
    
    # ==========================================
    # RETRIEVAL: Context for Prediction
    # ==========================================
    
    def retrieve_context(
        self,
        res_code: str,
        target_date: datetime.date,
        weekday: str = None,  # type: ignore[reportArgumentType]
        is_holiday: bool = False,
        top_k: int = 8,
    ) -> str:
        """
        Retrieve relevant context cho một dự đoán cụ thể.
        
        Kết hợp results từ 3 collections:
        1. Brain insights (biases, issues)
        2. Past prediction feedback (similar situations)
        3. Restaurant patterns (weekly/daily baselines)
        
        Returns:
            str: Formatted context text để inject vào LLM prompt
        """
        context_parts = []
        
        # Build query
        query = (
            f"Restaurant {res_code} forecast for {target_date} "
            f"({weekday or 'unknown'} {'HOLIDAY' if is_holiday else 'normal'})"
        )
        query_embedding = self._embed([query])
        
        # --- 1. Brain Insights ---
        try:
            col_insights = self._get_collection(COL_INSIGHTS)
            results = col_insights.query(
                query_embeddings=query_embedding,
                n_results=min(top_k, 5),
                where={"$or": [
                    {"restaurant": str(res_code)},
                    {"restaurant": "ALL"},
                ]},
            )
            if results and results['documents'] and results['documents'][0]:
                context_parts.append("=== LEARNED PATTERNS (from past errors) ===")
                for doc in results['documents'][0]:
                    context_parts.append(f"• {doc}")
        except Exception as e:
            logger.debug(f"Insights retrieval error: {e}")
        
        # --- 2. Prediction Feedback ---
        try:
            col_feedback = self._get_collection(COL_FEEDBACK)
            # Query for similar situations (same restaurant, same weekday)
            fb_where = {"restaurant": str(res_code)}
            if weekday:
                fb_where = {"$and": [
                    {"restaurant": str(res_code)},
                    {"weekday": weekday},
                ]}
            
            results = col_feedback.query(
                query_embeddings=query_embedding,
                n_results=min(top_k, 5),
                where=fb_where,
            )
            if results and results['documents'] and results['documents'][0]:
                context_parts.append("\n=== PAST PREDICTION HISTORY ===")
                for doc in results['documents'][0]:
                    context_parts.append(f"• {doc}")
        except Exception as e:
            logger.debug(f"Feedback retrieval error: {e}")
        
        # --- 3. Restaurant Patterns ---
        try:
            col_patterns = self._get_collection(COL_PATTERNS)
            results = col_patterns.query(
                query_embeddings=query_embedding,
                n_results=3,
                where={"restaurant": str(res_code)},
            )
            if results and results['documents'] and results['documents'][0]:
                context_parts.append("\n=== RESTAURANT BASELINE PATTERNS ===")
                for doc in results['documents'][0]:
                    context_parts.append(f"• {doc}")
        except Exception as e:
            logger.debug(f"Patterns retrieval error: {e}")
        
        if not context_parts:
            return ""
        
        return "\n".join(context_parts)
    
    # ==========================================
    # SELF-LEARNING: Store New Learnings
    # ==========================================
    
    def learn_from_results(
        self,
        res_code: str,
        date_obj: datetime.date,
        predicted: float,
        actual: float,
        weekday: str,
        strategy: str,
        is_holiday: bool = False,
    ):
        """
        Store a single prediction result for future learning.
        Called after actuals become available.
        """
        error = predicted - actual
        error_pct = abs(error) / actual * 100 if actual > 0 else 0
        
        direction = "over-predicted" if error > 0 else "under-predicted"
        
        doc = (
            f"Restaurant {res_code} on {date_obj} ({weekday}): "
            f"Predicted {int(predicted)}, Actual {int(actual)}. "
            f"Error: {error:+.0f} ({error_pct:.0f}%). "
            f"Model {direction}. Strategy: {strategy}. "
        )
        if is_holiday:
            doc += "Holiday. "
        if error_pct > 30:
            doc += "SIGNIFICANT ERROR - needs correction. "
        elif error_pct < 10:
            doc += "GOOD prediction - this pattern works. "
        
        col = self._get_collection(COL_FEEDBACK)
        embedding = self._embed([doc])
        
        col.upsert(
            ids=[self._make_id('learn', res_code, str(date_obj))],
            embeddings=embedding,
            documents=[doc],
            metadatas=[{
                'type': 'learning',
                'restaurant': str(res_code),
                'date': str(date_obj),
                'weekday': weekday,
                'predicted': float(predicted),
                'actual': float(actual),
                'error_pct': float(error_pct),
                'strategy': strategy,
                'is_holiday': is_holiday,
            }],
        )
    
    # ==========================================
    # UTILITIES
    # ==========================================
    
    def get_stats(self) -> Dict:
        """Get statistics about knowledge store."""
        stats = {}
        for name in [COL_INSIGHTS, COL_FEEDBACK, COL_PATTERNS]:
            try:
                col = self._get_collection(name)
                stats[name] = col.count()
            except Exception:
                stats[name] = 0
        return stats
    
    def clear_all(self):
        """Clear all collections (use with caution!)."""
        client = self._get_client()
        for name in [COL_INSIGHTS, COL_FEEDBACK, COL_PATTERNS]:
            try:
                client.delete_collection(name)
                logger.info(f"Deleted collection: {name}")
            except Exception:
                pass
        self._collections = {}
