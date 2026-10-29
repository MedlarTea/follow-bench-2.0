import random
from typing import Optional

import carla


WEATHER_PRESETS = {
    "clear":  carla.WeatherParameters(
        cloudiness=10.0, precipitation=0.0, precipitation_deposits=0.0,
        wind_intensity=10.0, sun_azimuth_angle=70.0, sun_altitude_angle=70.0,
        fog_density=0.0, fog_distance=0.0, wetness=0.0,
    ),
    "cloudy": carla.WeatherParameters(
        cloudiness=80.0, precipitation=0.0, precipitation_deposits=0.0,
        wind_intensity=20.0, sun_azimuth_angle=70.0, sun_altitude_angle=45.0,
        fog_density=0.0, fog_distance=0.0, wetness=0.0,
    ),
    "dusk":   carla.WeatherParameters(
        cloudiness=40.0, precipitation=0.0, precipitation_deposits=0.0,
        wind_intensity=10.0, sun_azimuth_angle=110.0, sun_altitude_angle=8.0,
        fog_density=0.0, fog_distance=0.0, wetness=0.0,
    ),
    "rain":   carla.WeatherParameters(
        cloudiness=90.0, precipitation=80.0, precipitation_deposits=60.0,
        wind_intensity=40.0, sun_azimuth_angle=70.0, sun_altitude_angle=30.0,
        fog_density=0.0, fog_distance=0.0, wetness=80.0,
    ),
    "fog":    carla.WeatherParameters(
        cloudiness=60.0, precipitation=0.0, precipitation_deposits=0.0,
        wind_intensity=5.0, sun_azimuth_angle=70.0, sun_altitude_angle=25.0,
        fog_density=80.0, fog_distance=8.0, fog_falloff=2.0, wetness=0.0,
    ),
}

WEATHER_LABELS_CN = {
    "clear": "晴朗", "cloudy": "阴天", "dusk": "下午傍晚", "rain": "雨天", "fog": "大雾",
}

CHOICES = list(WEATHER_PRESETS.keys())


LOW_VISIBILITY = ("dusk", "rain", "fog")

# Per-difficulty sampling weights.  easy/hard: uniform over all five presets;
# boss/legend: biased toward low-visibility conditions (80% dusk/rain/fog).
DIFFICULTY_WEIGHTS = {
    "easy":   {name: 0.20 for name in CHOICES},
    "hard":   {name: 0.20 for name in CHOICES},
    "boss":   {"clear": 0.10, "cloudy": 0.10, "dusk": 0.80 / 3, "rain": 0.80 / 3, "fog": 0.80 / 3},
    "legend": {"clear": 0.10, "cloudy": 0.10, "dusk": 0.80 / 3, "rain": 0.80 / 3, "fog": 0.80 / 3},
}


def pick_random(rng: Optional[random.Random] = None) -> str:
    rng = rng or random
    return rng.choice(CHOICES)


def sample_for_difficulty(difficulty: str, rng: Optional[random.Random] = None) -> str:
    """Sample a weather preset using the difficulty-dependent weights.

    Pass a seeded ``random.Random`` to make the draw reproducible (paired replay).
    """
    rng = rng or random
    weights = DIFFICULTY_WEIGHTS.get(str(difficulty).lower(), DIFFICULTY_WEIGHTS["easy"])
    return rng.choices(CHOICES, weights=[weights[n] for n in CHOICES], k=1)[0]


def seeded_rng(seed: Optional[int]) -> random.Random:
    """Weather RNG derived from the episode seed (independent stream from NPC draws)."""
    if seed is None:
        return random.Random()
    return random.Random(int(seed) * 7919 + 104729)


def apply(world, name: str) -> None:
    if name not in WEATHER_PRESETS:
        raise ValueError(f"unknown weather '{name}', choices={CHOICES}")
    world.set_weather(WEATHER_PRESETS[name])
