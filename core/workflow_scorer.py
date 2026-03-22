import json
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

class WorkflowScorer:
    """
    Tracks and persists the performance (quality scores) of various workflow templates.
    This allows the AI Director to learn which workflows produce the best results over time.
    """
    def __init__(self, data_dir: str):
        self.scores_file = Path(data_dir) / "workflow_scores.json"
        self.stats: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self):
        if self.scores_file.exists():
            try:
                with open(self.scores_file, "r", encoding="utf-8") as f:
                    self.stats = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load workflow scores from {self.scores_file}: {e}")
                self.stats = {}

    def save(self):
        try:
            self.scores_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.scores_file, "w", encoding="utf-8") as f:
                json.dump(self.stats, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save workflow scores: {e}")

    def record_score(self, template_name: str, score_data: Any):
        """
        Record a QualityScore for a specific template.
        score_data is expected to be a QualityScore object with .overall, .faces, etc.
        """
        if not template_name:
            return
            
        if template_name not in self.stats:
            self.stats[template_name] = {
                "runs": 0,
                "avg_overall": 0.0,
                "avg_faces": 0.0,
                "avg_artifacts": 0.0
            }
            
        stat = self.stats[template_name]
        runs = stat["runs"]
        
        # Incremental average calculation
        stat["avg_overall"] = ((stat["avg_overall"] * runs) + getattr(score_data, 'overall', 5.0)) / (runs + 1)
        
        if getattr(score_data, 'faces', None) is not None:
            stat["avg_faces"] = ((stat.get("avg_faces", 0.0) * runs) + score_data.faces) / (runs + 1)
            
        if getattr(score_data, 'artifacts', None) is not None:
            stat["avg_artifacts"] = ((stat.get("avg_artifacts", 0.0) * runs) + score_data.artifacts) / (runs + 1)
            
        stat["runs"] = runs + 1
        self.save()

    def get_template_stats_string(self, template_name: str) -> str:
        """Returns a summarized string of a template's historical performance."""
        if template_name not in self.stats or self.stats[template_name]["runs"] == 0:
            return "No historical data yet."
            
        stat = self.stats[template_name]
        return f"Avg Score: {stat['avg_overall']:.1f}/10 (Runs: {stat['runs']})"
