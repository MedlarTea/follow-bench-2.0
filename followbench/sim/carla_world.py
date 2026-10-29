"""Connect to the CARLA world for an episode, avoiding needless ``load_world``.

This CARLA build intermittently hangs its render thread inside
``client.load_world`` (the client then times out, and UE later dies with
Signal 11 once the render-fence timeout expires). A freshly booted server
already has its default map (Town10HD_Opt) loaded, so reloading it buys
nothing and is the only place the hang has been observed.

``FB_LOAD_MAP_POLICY`` controls the behaviour when ``--load-map`` is given:
  always     (default) always call ``load_world``;
  if_needed  skip it when the current map already matches and the world holds
             no leftover vehicles / walkers / sensors / controllers, i.e. the
             server is fresh or the previous episode cleaned up completely.
"""
from __future__ import annotations

import os
import time

import carla

_DYNAMIC_PREFIXES = ("vehicle.", "walker.", "sensor.", "controller.")


def _map_basename(name: str) -> str:
    return name.rsplit("/", 1)[-1]


def connect_world(client: carla.Client, map_name: str, load_map: bool) -> carla.World:
    if not load_map:
        return client.get_world()
    policy = os.environ.get("FB_LOAD_MAP_POLICY", "always").strip().lower()
    if policy == "if_needed":
        world = client.get_world()
        current = _map_basename(world.get_map().name)
        leftovers = [a for a in world.get_actors() if a.type_id.startswith(_DYNAMIC_PREFIXES)]
        if current == _map_basename(map_name) and not leftovers:
            print(f"[EP] map {current} already loaded on a clean world; skipping load_world")
            return world
        print(f"[EP] load_world needed (current={current} leftover_actors={len(leftovers)})")
    print(f"[EP] loading map: {map_name}")
    world = client.load_world(map_name)
    time.sleep(1.0)
    return world
