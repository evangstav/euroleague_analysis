"""Statistical view definitions."""

from .base import SQLView


class PositionMatchupView(SQLView):
    """Position matchup features view"""

    def __init__(self):
        super().__init__("position_matchup", "position_matchup_features.sql")


class RollingStatsView(SQLView):
    """Rolling statistics view"""

    def __init__(self):
        super().__init__("rolling_stats", "rolling_stats.sql")


class ShotPatternsView(SQLView):
    """Shot patterns view"""

    def __init__(self):
        super().__init__("shot_patterns", "shot_patterns.sql")


class GameContextView(SQLView):
    """Game context view"""

    def __init__(self):
        super().__init__("game_context", "game_context.sql")


class PlayByPlayView(SQLView):
    """Play-by-play features view"""

    def __init__(self):
        super().__init__("playbyplay", "playbyplay_features.sql")


class PlayerTiersView(SQLView):
    """Player tiers view"""

    def __init__(self):
        super().__init__("player_tiers", "player_tiers.sql")
