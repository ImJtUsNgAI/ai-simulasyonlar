# ============================================================
#  AI Football Simulator - Configuration File
#  Edit values here or use the in-game Settings menu (S key)
# ============================================================

# ── Window ───────────────────────────────────────────────────
WINDOW_W   = 1400
WINDOW_H   = 860
FPS        = 60
TITLE      = "AI Football Simulator"

# ── Field ────────────────────────────────────────────────────
FIELD_W    = 800
FIELD_H    = 560
GOAL_HALF  = 70   # half-height of each goal

# ── Isometric projection ─────────────────────────────────────
ISO_CX, ISO_CY       = 590, 430
ISO_SX, ISO_SY, ISO_Z = 0.68, 0.36, 0.52

# ── Simulation ───────────────────────────────────────────────
DEFAULT_SPEED      = 3      # sim steps per frame
DEFAULT_ALGORITHM  = "neuroevolution"

# ── Physics ──────────────────────────────────────────────────
BALL_FRICTION      = 0.970
BALL_BOUNCE        = 0.62
AGENT_FRICTION     = 0.80
AGENT_MAX_SPEED    = 7.5
AGENT_ACCEL        = 3.8
AGENT_PULL_RANGE   = 200   # ball attraction radius
AGENT_PULL_STR     = 0.8   # ball attraction strength
KICK_BASE          = 5.5
KICK_EXTRA         = 2.5
TOUCH_RADIUS       = 25    # pixel distance to count as touch

# ── Player (human-controlled) ────────────────────────────────
PLAYER_SPEED       = 5.5
PLAYER_KICK_STR    = 7.0

# ── Levels ───────────────────────────────────────────────────
LEVELS = {
    1: {
        "name":           "Level 1 — Touch the Ball",
        "desc":           "Just learn to touch the ball. Every tick without touching = penalty!",
        "n_blue":         1,
        "n_red":          0,
        "episode_s":      20,
        "touch_limit":    15,
        "has_goalkeeper": False,
        "match_mode":     False,
        "reward": {
            "touch":          20.0,
            "no_touch_tick":  -0.10,
            "ball_to_goal":   0.0,
            "goal":           0.0,
            "timeout":        -50.0,
            "proximity":      0.018,
            "dribble_max":    3.0,
        },
    },
    2: {
        "name":           "Level 2 — Shoot",
        "desc":           "Touch the ball and direct it toward goal. Score = big reward!",
        "n_blue":         1,
        "n_red":          0,
        "episode_s":      25,
        "touch_limit":    15,
        "has_goalkeeper": False,
        "match_mode":     False,
        "reward": {
            "touch":          10.0,
            "no_touch_tick":  -0.06,
            "ball_to_goal":   0.20,
            "goal":           400.0,
            "timeout":        -40.0,
            "proximity":      0.018,
            "dribble_max":    3.0,
        },
    },
    3: {
        "name":           "Level 3 — Goalkeeper",
        "desc":           "A goalkeeper blocks the angle. Dribble or shoot from distance!",
        "n_blue":         1,
        "n_red":          0,
        "episode_s":      30,
        "touch_limit":    15,
        "has_goalkeeper": True,
        "match_mode":     False,
        "reward": {
            "touch":          8.0,
            "no_touch_tick":  -0.05,
            "ball_to_goal":   0.22,
            "goal":           600.0,
            "timeout":        -40.0,
            "proximity":      0.018,
            "dribble_max":    3.0,
        },
    },
    4: {
        "name":           "Level 4 — 2v1 Pass",
        "desc":           "2 attackers vs 1 goalkeeper. Pass to find open space!",
        "n_blue":         2,
        "n_red":          0,
        "episode_s":      35,
        "touch_limit":    15,
        "has_goalkeeper": True,
        "match_mode":     False,
        "reward": {
            "touch":          6.0,
            "no_touch_tick":  -0.04,
            "ball_to_goal":   0.24,
            "goal":           700.0,
            "timeout":        -40.0,
            "proximity":      0.018,
            "dribble_max":    3.0,
        },
    },
    5: {
        "name":           "Level 5 — Full Match",
        "desc":           "2v2 full match! Each team defends its own goal.",
        "n_blue":         2,
        "n_red":          2,
        "episode_s":      90,
        "touch_limit":    20,
        "has_goalkeeper": False,
        "match_mode":     True,
        "reward": {
            "touch":          5.0,
            "no_touch_tick":  -0.03,
            "ball_to_goal":   0.26,
            "goal":           1000.0,
            "timeout":        -35.0,
            "proximity":      0.018,
            "dribble_max":    3.0,
        },
    },
}

# ── Objectives (shown in UI, used for bonus rewards) ─────────
OBJECTIVES = {
    "score_goal":     {"label": "Score a Goal",        "bonus": 0},
    "dribble_past":   {"label": "Dribble Past GK",     "bonus": 50},
    "consecutive":    {"label": "3 Goals in a Row",    "bonus": 200},
    "no_timeout":     {"label": "No Timeouts",         "bonus": 100},
    "team_pass":      {"label": "Team Pass & Goal",    "bonus": 300},
}

# ── Algorithms ───────────────────────────────────────────────
ALGORITHMS = {
    "neuroevolution": {"name": "Neuroevolution",        "short": "NE",   "desc": "Best brain becomes parent, all children mutate from it."},
    "tournament":     {"name": "Tournament Selection",  "short": "TS",   "desc": "3 random candidates per child, best one is parent."},
    "elitism":        {"name": "Elitism + Diversity",   "short": "EL",   "desc": "Top 2 copied directly, rest are their offspring."},
    "mentor":         {"name": "Mentor Teaching",       "short": "MT",   "desc": "Best agent blended into all others as a teacher."},
    "annealing":      {"name": "Simulated Annealing",   "short": "SA",   "desc": "High mutation early (explore), decreases over time (exploit)."},
    "ppo_lite":       {"name": "PPO-Lite",              "short": "PPO",  "desc": "Policy gradient update from experience replay buffer."},
    "dqn_lite":       {"name": "DQN-Lite",              "short": "DQN",  "desc": "Deep Q-Network: 9 discrete actions, epsilon-greedy."},
    "marl":           {"name": "MARL Cooperative",      "short": "MARL", "desc": "Team score shared: agents learn together."},
}
