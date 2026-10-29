from __future__ import annotations

import math
import threading
from typing import Dict, Optional

import carla
import numpy as np

from followbench.sim.core_types import CalibrationDict, RobotState

DEFAULT_LIDAR_SCAN_HEIGHT_AGL = 0.50
SCOUT_ORIGIN_HEIGHT_AGL = 0.35


def _camera_intrinsics(width: int, height: int, fov_deg: float) -> Dict[str, float]:
    f = width / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    return {
        "fx": float(f),
        "fy": float(f),
        "cx": float(width / 2.0),
        "cy": float(height / 2.0),
        "width": int(width),
        "height": int(height),
        "fov_deg": float(fov_deg),
    }


class ScoutRobotRuntime:
    origin_height_agl = SCOUT_ORIGIN_HEIGHT_AGL

    def __init__(self, world: carla.World, dt: float) -> None:
        self.world = world
        self.dt = dt
        self.actor: Optional[carla.Actor] = None
        self.sensors: Dict[str, carla.Actor] = {}
        self._sensor_data: Dict[str, object] = {}
        self._locks: Dict[str, threading.Lock] = {}
        self._control = carla.VehicleControl()
        self._control.hand_brake = False
        self._control.manual_gear_shift = False
        self.max_speed = 4.0
        self.max_steer_rad = 0.7
        self.wheelbase = 1.8
        self._calibration: CalibrationDict = {}
        self._sensor_rel_tf: Dict[str, carla.Transform] = {}
        self.rescue_lift_z = 0.20

    def spawn(self, spawn_tf: carla.Transform,
              roi_map=None, forbidden_xys=None, min_clearance_m: float = 0.7) -> None:
        """Spawn the scout robot, trying to avoid colliding with existing actors.

        Stages:
          1. Try the requested (x, y) at progressive z-lifts (0, 0.4, 0.8, 1.2 m).
          2. Fall back to a ring search around (x, y) at radii 0.5..4.5 m, 24
             angular candidates, skipping non-walkable cells (if roi_map given)
             and any candidate within `min_clearance_m` of a forbidden xy
             (typically other walker positions).
        Whatever transform actually succeeds is used for the post-spawn
        ``force_place`` so the robot doesn't get yanked back into a collision.
        """
        bp_lib = self.world.get_blueprint_library()
        try:
            scout_bp = bp_lib.find("vehicle.scout.scout")
        except RuntimeError:
            matches = bp_lib.filter("vehicle.scout*")
            scout_bp = matches[0] if matches else None
        if scout_bp is None:
            raise RuntimeError("Scout blueprint not found.")
        if scout_bp.has_attribute("role_name"):
            scout_bp.set_attribute("role_name", "followbench_robot")

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

        # Stage 1: requested xy with progressive z-lift.
        for z_lift in (0.0, 0.4, 0.8, 1.2):
            cand = carla.Transform(
                carla.Location(x=base_x, y=base_y, z=base_z + z_lift), rot,
            )
            actor = self.world.try_spawn_actor(scout_bp, cand)
            if actor is not None:
                chosen_tf = cand
                break

        # Stage 2: ring search (perturbed xy) until something works.
        if actor is None:
            ring_radii = (0.5, 1.0, 1.5, 2.0, 3.0, 4.5)
            for r in ring_radii:
                for theta in np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False):
                    x = base_x + r * math.cos(float(theta))
                    y = base_y + r * math.sin(float(theta))
                    if not _xy_clear(x, y):
                        continue
                    cand = carla.Transform(
                        carla.Location(x=x, y=y, z=base_z + 0.4), rot,
                    )
                    actor = self.world.try_spawn_actor(scout_bp, cand)
                    if actor is not None:
                        chosen_tf = cand
                        break
                if actor is not None:
                    break

        if actor is None:
            raise RuntimeError(
                "Failed to spawn robot scout: every z-lift and ring-search "
                "candidate collided. Inspect crowd density at spawn region."
            )
        self.actor = actor
        if (chosen_tf is not spawn_tf
                and (abs(chosen_tf.location.x - base_x) > 1e-3
                     or abs(chosen_tf.location.y - base_y) > 1e-3)):
            print(f"[ROBOT] spawn fell back to ({chosen_tf.location.x:.2f},"
                  f"{chosen_tf.location.y:.2f}) — original ({base_x:.2f},{base_y:.2f}) was occupied")
        self._configure_vehicle_physics()
        self.force_place(chosen_tf)
        self.rescue_reset(lift_z=self.rescue_lift_z)
        # Switch to pure kinematic mode: physics off so set_transform sticks every tick.
        try:
            self.actor.set_simulate_physics(False)
        except Exception:
            pass
        # rescue_reset reads get_transform() before force_place materializes (CARLA needs
        # a tick), so it may have overwritten the intended spawn rotation with yaw=0.
        # Re-apply the spawn rotation using the z-lifted position rescue_reset established.
        try:
            cur = self.actor.get_transform()
            self.actor.set_transform(carla.Transform(cur.location, spawn_tf.rotation))
        except Exception:
            pass

    def lidar_mount_z(self, scan_height_agl: float = DEFAULT_LIDAR_SCAN_HEIGHT_AGL) -> float:
        return float(scan_height_agl) - float(self.origin_height_agl)

    def spawn_sensors(
        self,
        image_w: int = 800,
        image_h: int = 600,
        cam_fov: float = 90.0,
        lidar_range: float = 30.0,
        lidar_scan_height_agl: float = DEFAULT_LIDAR_SCAN_HEIGHT_AGL,
        enable_side_views: bool = True,
    ) -> None:
        if self.actor is None:
            raise RuntimeError("Spawn robot before sensors.")
        bp_lib = self.world.get_blueprint_library()

        camera_mount_z = 1.05
        lidar_mount_z = self.lidar_mount_z(lidar_scan_height_agl)
        forward_loc = carla.Location(x=0.30, y=0.0, z=camera_mount_z)
        # CARLA/UE yaw: +90° = robot's right, -90° = robot's left.
        rel = {
            "rgb": carla.Transform(forward_loc),
            "depth": carla.Transform(forward_loc),
            "instance": carla.Transform(forward_loc),
            "rgb_left": carla.Transform(forward_loc, carla.Rotation(yaw=-90.0)),
            "depth_left": carla.Transform(forward_loc, carla.Rotation(yaw=-90.0)),
            "instance_left": carla.Transform(forward_loc, carla.Rotation(yaw=-90.0)),
            "rgb_right": carla.Transform(forward_loc, carla.Rotation(yaw=90.0)),
            "depth_right": carla.Transform(forward_loc, carla.Rotation(yaw=90.0)),
            "instance_right": carla.Transform(forward_loc, carla.Rotation(yaw=90.0)),
            "lidar": carla.Transform(carla.Location(x=0.30, y=0.0, z=lidar_mount_z)),
        }
        self._sensor_rel_tf = rel

        def setup_cam(name: str, bp_name: str) -> None:
            bp = bp_lib.find(bp_name)
            bp.set_attribute("image_size_x", str(image_w))
            bp.set_attribute("image_size_y", str(image_h))
            bp.set_attribute("fov", str(cam_fov))
            s = self.world.spawn_actor(
                bp,
                rel[name],
                attach_to=self.actor,
                attachment_type=carla.AttachmentType.Rigid,
            )
            self.sensors[name] = s
            self._attach_listener(name, s)
            self._calibration[name] = {
                "type": "camera",
                "intrinsics": _camera_intrinsics(image_w, image_h, cam_fov),
                "extrinsics_robot_to_sensor": {
                    "x": rel[name].location.x,
                    "y": rel[name].location.y,
                    "z": rel[name].location.z,
                    "roll_deg": rel[name].rotation.roll,
                    "pitch_deg": rel[name].rotation.pitch,
                    "yaw_deg": rel[name].rotation.yaw,
                },
            }

        setup_cam("rgb", "sensor.camera.rgb")
        setup_cam("depth", "sensor.camera.depth")
        setup_cam("instance", "sensor.camera.instance_segmentation")
        # Side views are only spawned for side-follow modes; back-follow uses front only.
        if enable_side_views:
            setup_cam("rgb_left", "sensor.camera.rgb")
            setup_cam("depth_left", "sensor.camera.depth")
            setup_cam("instance_left", "sensor.camera.instance_segmentation")
            setup_cam("rgb_right", "sensor.camera.rgb")
            setup_cam("depth_right", "sensor.camera.depth")
            setup_cam("instance_right", "sensor.camera.instance_segmentation")

        lb = bp_lib.find("sensor.lidar.ray_cast")
        lb.set_attribute("range", str(lidar_range))
        # Single-line lidar for lightweight forward profile.
        lb.set_attribute("channels", "1")
        lb.set_attribute("points_per_second", "12000")
        lb.set_attribute("rotation_frequency", str(1.0 / self.dt))
        lb.set_attribute("upper_fov", "0")
        lb.set_attribute("lower_fov", "0")
        lidar = self.world.spawn_actor(
            lb,
            rel["lidar"],
            attach_to=self.actor,
            attachment_type=carla.AttachmentType.Rigid,
        )
        self.sensors["lidar"] = lidar
        self._attach_listener("lidar", lidar)
        self._calibration["lidar"] = {
            "type": "lidar",
            "params": {
                "range": lidar_range,
                "channels": 1,
                "points_per_second": 12000,
                "upper_fov": 0.0,
                "lower_fov": 0.0,
                "scan_mode": "single_line",
                "scan_height_agl": float(lidar_scan_height_agl),
                "origin_height_agl": float(self.origin_height_agl),
            },
            "extrinsics_robot_to_sensor": {
                "x": rel["lidar"].location.x,
                "y": rel["lidar"].location.y,
                "z": rel["lidar"].location.z,
                "roll_deg": rel["lidar"].rotation.roll,
                "pitch_deg": rel["lidar"].rotation.pitch,
                "yaw_deg": rel["lidar"].rotation.yaw,
            },
        }

    def _attach_listener(self, name: str, sensor: carla.Actor) -> None:
        self._locks[name] = threading.Lock()
        self._sensor_data[name] = None

        def cb(data, n=name):
            with self._locks[n]:
                self._sensor_data[n] = data

        sensor.listen(cb)

    def get_sensor_data(self, name: str):
        """Peek the latest sensor frame without consuming it.

        CARLA sensor listeners run on worker threads and are not guaranteed to
        have written the new frame by the time `world.tick()` returns on the
        main thread. The previous "read-and-clear" semantics caused intermittent
        `None` returns mid-episode (visible to downstream code as e.g.
        `rgb.shape` on a NoneType). We instead always return the most recent
        frame; the listener overwrites it whenever a new one arrives.
        """
        if name not in self._sensor_data:
            return None
        with self._locks[name]:
            return self._sensor_data[name]

    def get_calibration(self) -> CalibrationDict:
        return self._calibration

    def get_state(self) -> RobotState:
        if self.actor is None:
            raise RuntimeError("Robot not spawned.")
        loc = self.actor.get_location()
        vel = self.actor.get_velocity()
        tf = self.actor.get_transform()
        speed = float(np.hypot(vel.x, vel.y))
        return RobotState(
            x=float(loc.x),
            y=float(loc.y),
            z=float(loc.z),
            yaw_rad=float(np.deg2rad(tf.rotation.yaw)),
            speed=speed,
        )

    def apply_velocity_command(self, v_mps: float, w_radps: float) -> None:
        """
        Pure kinematic control via set_transform (same approach used by sfm_robot_follow).
        Computes the next pose from (v, w, dt) using the unicycle model and teleports
        the actor there each tick.  Bypasses CARLA vehicle physics entirely, which is
        necessary because:
          - VehicleControl (Ackermann) cannot rotate in place
          - set_target_velocity is overridden by vehicle damping each tick
        The actor's simulate_physics is kept OFF so the transform sticks.
        """
        if self.actor is None:
            return
        v = float(np.clip(v_mps, -self.max_speed, self.max_speed))
        w = float(w_radps)
        tf = self.actor.get_transform()
        yaw_rad = float(math.radians(tf.rotation.yaw))
        # Unicycle kinematic step
        new_yaw_rad = yaw_rad + w * self.dt
        new_x = tf.location.x + v * math.cos(yaw_rad) * self.dt
        new_y = tf.location.y + v * math.sin(yaw_rad) * self.dt
        new_tf = carla.Transform(
            carla.Location(x=new_x, y=new_y, z=tf.location.z),
            carla.Rotation(pitch=0.0, yaw=float(math.degrees(new_yaw_rad)), roll=0.0),
        )
        self.actor.set_transform(new_tf)

    def hold_still(self) -> None:
        if self.actor is None:
            return
        # Just don't move — no transform update needed.

    def _configure_vehicle_physics(self) -> None:
        if self.actor is None:
            return
        try:
            physics = self.actor.get_physics_control()
        except Exception:
            return
        extent = self.actor.bounding_box.extent
        physics.center_of_mass.x = 0.0
        physics.center_of_mass.y = 0.0
        physics.center_of_mass.z = -min(0.08, max(0.03, extent.z * 0.25))
        physics.mass = max(physics.mass, 80.0)
        for wheel in physics.wheels:
            wheel.damping_rate = max(wheel.damping_rate, 1.5)
            wheel.tire_friction = max(wheel.tire_friction, 2.5)
        try:
            self.actor.apply_physics_control(physics)
        except Exception:
            pass

    def rescue_reset(self, lift_z: float = 0.20) -> None:
        """R-key style rescue reset to prevent spawn sinking/tilting."""
        if self.actor is None:
            return
        tf = self.actor.get_transform()
        rescued_tf = carla.Transform(
            carla.Location(
                x=tf.location.x,
                y=tf.location.y,
                z=tf.location.z + float(lift_z),
            ),
            carla.Rotation(
                pitch=0.0,
                yaw=tf.rotation.yaw,
                roll=0.0,
            ),
        )
        try:
            self.actor.set_simulate_physics(False)
            self.actor.set_transform(rescued_tf)
            self.actor.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
            self.actor.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
            self.actor.apply_control(
                carla.VehicleControl(
                    throttle=0.0,
                    brake=1.0,
                    steer=0.0,
                    reverse=False,
                    hand_brake=False,
                )
            )
        finally:
            try:
                self.actor.set_simulate_physics(True)
            except Exception:
                pass

    def force_place(self, target_tf: carla.Transform) -> None:
        """Hard place actor to target transform, robust to spawn-at-origin issues."""
        if self.actor is None:
            return
        try:
            self.actor.set_simulate_physics(False)
            self.actor.set_transform(target_tf)
            self.actor.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
            self.actor.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
            self.actor.apply_control(
                carla.VehicleControl(
                    throttle=0.0,
                    brake=1.0,
                    steer=0.0,
                    reverse=False,
                    hand_brake=False,
                )
            )
        finally:
            try:
                self.actor.set_simulate_physics(True)
            except Exception:
                pass

    def destroy(self) -> None:
        for s in self.sensors.values():
            try:
                s.stop()
                s.destroy()
            except Exception:
                pass
        self.sensors.clear()
        if self.actor is not None:
            try:
                self.actor.destroy()
            except Exception:
                pass
            self.actor = None
