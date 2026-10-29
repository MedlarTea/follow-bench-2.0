from __future__ import annotations

import math

import carla
import numpy as np

from followbench.sim.robot_runtime import ScoutRobotRuntime

G1_ORIGIN_HEIGHT_AGL = 0.65


class G1RobotRuntime(ScoutRobotRuntime):
    """Unitree G1 humanoid follower — drop-in for ScoutRobotRuntime.

    Spawns the ``walker.pedestrian.g1_unitree`` actor. With ``use_physics`` (the
    default) it walks via ``WalkerControl`` so the legs actually animate, the way
    ``tools/teleop/keyboard_G1.py`` drives it: physics stays on, the
    planner's ``(v, w)`` becomes a walk direction + speed, and ``set_transform``
    only re-locks the yaw each tick (position comes from the walk). Set
    ``use_physics=False`` to fall back to the scout's pure-kinematic teleport.
    """

    BLUEPRINT_ID = "walker.pedestrian.g1_unitree"
    origin_height_agl = G1_ORIGIN_HEIGHT_AGL

    def __init__(self, world: carla.World, dt: float) -> None:
        super().__init__(world, dt)
        self.max_speed = 1.8  # humanoid walking cap (scout was 4.0)
        self.use_physics = True
        self._heading_rad = 0.0

    def spawn(self, spawn_tf: carla.Transform,
              roi_map=None, forbidden_xys=None, min_clearance_m: float = 0.7) -> None:
        bp_lib = self.world.get_blueprint_library()
        try:
            bp = bp_lib.find(self.BLUEPRINT_ID)
        except RuntimeError:
            matches = bp_lib.filter("walker.pedestrian.g1*")
            bp = matches[0] if matches else None
        if bp is None:
            raise RuntimeError(f"G1 humanoid blueprint not found ({self.BLUEPRINT_ID}).")
        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", "followbench_robot")
        if bp.has_attribute("is_invincible"):
            bp.set_attribute("is_invincible", "false")

        forbidden = list(forbidden_xys) if forbidden_xys else []
        clearance_sq = float(min_clearance_m) * float(min_clearance_m)
        base_x = float(spawn_tf.location.x)
        base_y = float(spawn_tf.location.y)
        base_z = float(spawn_tf.location.z)
        rot = spawn_tf.rotation

        def _xy_clear(x: float, y: float) -> bool:
            if roi_map is not None and not roi_map.is_world_walkable(x, y):
                return False
            for fx, fy in forbidden:
                dx = x - float(fx)
                dy = y - float(fy)
                if dx * dx + dy * dy < clearance_sq:
                    return False
            return True

        actor = None
        chosen_tf = spawn_tf
        for z_lift in (0.0, 0.4, 0.8, 1.2):
            cand = carla.Transform(
                carla.Location(x=base_x, y=base_y, z=base_z + z_lift), rot,
            )
            actor = self.world.try_spawn_actor(bp, cand)
            if actor is not None:
                chosen_tf = cand
                break

        if actor is None:
            for r in (0.5, 1.0, 1.5, 2.0, 3.0, 4.5):
                for theta in np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False):
                    x = base_x + r * math.cos(float(theta))
                    y = base_y + r * math.sin(float(theta))
                    if not _xy_clear(x, y):
                        continue
                    cand = carla.Transform(
                        carla.Location(x=x, y=y, z=base_z + 0.4), rot,
                    )
                    actor = self.world.try_spawn_actor(bp, cand)
                    if actor is not None:
                        chosen_tf = cand
                        break
                if actor is not None:
                    break

        if actor is None:
            raise RuntimeError(
                "Failed to spawn G1 humanoid: every z-lift and ring-search "
                "candidate collided. Inspect crowd density at spawn region."
            )
        self.actor = actor
        if (chosen_tf is not spawn_tf
                and (abs(chosen_tf.location.x - base_x) > 1e-3
                     or abs(chosen_tf.location.y - base_y) > 1e-3)):
            print(f"[ROBOT] G1 spawn fell back to ({chosen_tf.location.x:.2f},"
                  f"{chosen_tf.location.y:.2f}) — original ({base_x:.2f},{base_y:.2f}) was occupied")

        self._heading_rad = math.radians(float(spawn_tf.rotation.yaw))
        if not self.use_physics:
            try:
                self.actor.set_simulate_physics(False)
            except Exception:
                pass
        self.force_place(chosen_tf)

    def apply_velocity_command(self, v_mps: float, w_radps: float) -> None:
        if self.actor is None:
            return
        if not self.use_physics:
            super().apply_velocity_command(v_mps, w_radps)
            return
        # Keep the actor heading fixed while allowing signed longitudinal
        # commands. Negative speed moves opposite the heading, matching the
        # Scout teleop convention without turning the cameras around.
        v = float(np.clip(v_mps, -self.max_speed, self.max_speed))
        self._heading_rad += float(w_radps) * self.dt
        direction_sign = -1.0 if v < 0.0 else 1.0
        dx = direction_sign * math.cos(self._heading_rad)
        dy = direction_sign * math.sin(self._heading_rad)
        try:
            self.actor.apply_control(carla.WalkerControl(
                direction=carla.Vector3D(x=dx, y=dy, z=0.0),
                speed=abs(v),
                jump=False,
            ))
            tf = self.actor.get_transform()
            tf.rotation.yaw = math.degrees(self._heading_rad)
            self.actor.set_transform(tf)
        except Exception:
            pass

    def hold_still(self) -> None:
        if self.actor is None or not self.use_physics:
            return
        try:
            self.actor.apply_control(carla.WalkerControl(
                direction=carla.Vector3D(
                    x=math.cos(self._heading_rad), y=math.sin(self._heading_rad), z=0.0),
                speed=0.0, jump=False))
        except Exception:
            pass

    def force_place(self, target_tf: carla.Transform) -> None:
        if self.actor is None:
            return
        try:
            self.actor.set_simulate_physics(False)
            self.actor.set_transform(target_tf)
            self.actor.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
            self.actor.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
            self._heading_rad = math.radians(float(target_tf.rotation.yaw))
        except Exception:
            pass
        finally:
            if self.use_physics:
                try:
                    self.actor.set_simulate_physics(True)
                except Exception:
                    pass

    def rescue_reset(self, lift_z: float = 0.20) -> None:
        if self.actor is None:
            return
        tf = self.actor.get_transform()
        try:
            if not self.use_physics:
                self.actor.set_simulate_physics(False)
            self.actor.set_transform(carla.Transform(
                carla.Location(x=tf.location.x, y=tf.location.y, z=tf.location.z + float(lift_z)),
                carla.Rotation(pitch=0.0, yaw=tf.rotation.yaw, roll=0.0),
            ))
        except Exception:
            pass
