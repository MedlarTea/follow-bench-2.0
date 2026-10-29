import argparse
import math
import time
import sys
import weakref
import queue

import carla
import numpy as np
import pygame
from pygame.locals import K_a
from pygame.locals import K_d
from pygame.locals import K_ESCAPE
from pygame.locals import K_r
from pygame.locals import K_s
from pygame.locals import K_SPACE
from pygame.locals import K_TAB
from pygame.locals import K_w


WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
FIXED_DELTA_SECONDS = 0.02
TARGET_FPS = 60
CLIENT_TIMEOUT_SECONDS = 30.0

DEFAULT_BLUEPRINT_ID = "vehicle.scout.scout"
DEFAULT_ROLE_NAME = "scout"

FORWARD_SPEED_LIMIT_KMH = 30.0
REVERSE_SPEED_LIMIT_KMH = 15.0
RESCUE_LIFT_Z = 0.2

# Fixed spawn point
DEFAULT_ORIGIN_SPAWN_TRANSFORM = carla.Transform(
    carla.Location(x=-99.990, y=115.540, z=0.20),
    carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
)

# Main follow camera
MAIN_CAMERA_FOV = "95"

# Surround-view cameras
SV_CAMERA_FOV = "120"
SV_CAMERA_Z = 1.5
SV_CAMERA_X = 0.0
SV_CAMERA_Y = 0.0
SV_CAMERA_PITCH = 0.0
SV_CAMERA_ROLL = 0.0

SV_WIDTH = WINDOW_WIDTH // 2
SV_HEIGHT = WINDOW_HEIGHT // 2

VIEW_MAIN = 0
VIEW_SURROUND = 1


def clamp(value, low, high):
    return max(low, min(high, value))


def move_towards(current, target, rate, dt):
    if current < target:
        return min(current + rate * dt, target)
    return max(current - rate * dt, target)


def image_to_surface(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3][:, :, ::-1]
    return pygame.surfarray.make_surface(array.swapaxes(0, 1))


def make_camera_blueprint(world, width, height, fov):
    blueprint = world.get_blueprint_library().find("sensor.camera.rgb")
    blueprint.set_attribute("image_size_x", str(width))
    blueprint.set_attribute("image_size_y", str(height))
    blueprint.set_attribute("fov", str(fov))
    if blueprint.has_attribute("gamma"):
        blueprint.set_attribute("gamma", "2.2")
    return blueprint


class FollowCamera:
    def __init__(self, world, vehicle, width, height):
        self.world = world
        self.vehicle = vehicle
        self.width = width
        self.height = height
        self.sensor = None
        self.surface = None
        self._transform_index = 0
        self._camera_transforms = [
            carla.Transform(
                carla.Location(x=-4.0, y=0.0, z=1.5),
                carla.Rotation(pitch=-4.0),
            ),
            carla.Transform(
                carla.Location(x=0.0, y=0.0, z=5.5),
                carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0),
            ),
        ]
        self._spawn()

    def _is_top_view(self):
        return self._transform_index == 1

    def _get_top_view_transform(self):
        vehicle_transform = self.vehicle.get_transform()
        return carla.Transform(
            carla.Location(
                x=vehicle_transform.location.x,
                y=vehicle_transform.location.y,
                z=vehicle_transform.location.z + 5.5,
            ),
            carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0),
        )

    def _spawn(self):
        blueprint = make_camera_blueprint(self.world, self.width, self.height, MAIN_CAMERA_FOV)

        if self._is_top_view():
            self.sensor = self.world.spawn_actor(blueprint, self._get_top_view_transform())
        else:
            self.sensor = self.world.spawn_actor(
                blueprint,
                self._camera_transforms[self._transform_index],
                attach_to=self.vehicle,
                attachment_type=carla.AttachmentType.SpringArmGhost,
            )

        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: FollowCamera._on_image(weak_self, image))

    @staticmethod
    def _on_image(weak_self, image):
        self = weak_self()
        if self is None:
            return
        self.surface = image_to_surface(image)

    def tick(self):
        if self.sensor is not None and self._is_top_view():
            self.sensor.set_transform(self._get_top_view_transform())

    def next_view(self):
        self._transform_index = (self._transform_index + 1) % len(self._camera_transforms)
        self.destroy()
        self._spawn()

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    def destroy(self):
        if self.sensor is not None:
            self.sensor.destroy()
            self.sensor = None


class SurroundCameraRig:
    def __init__(self, world, vehicle):
        self.world = world
        self.vehicle = vehicle

        self.front_sensor = None
        self.rear_sensor = None
        self.left_sensor = None
        self.right_sensor = None

        self.front_queue = queue.Queue()
        self.rear_queue = queue.Queue()
        self.left_queue = queue.Queue()
        self.right_queue = queue.Queue()

        self.front_surface = None
        self.rear_surface = None
        self.left_surface = None
        self.right_surface = None

        self._spawn()

    def _spawn_one(self, yaw_deg):
        blueprint = make_camera_blueprint(self.world, SV_WIDTH, SV_HEIGHT, SV_CAMERA_FOV)
        transform = carla.Transform(
            carla.Location(x=SV_CAMERA_X, y=SV_CAMERA_Y, z=SV_CAMERA_Z),
            carla.Rotation(pitch=SV_CAMERA_PITCH, yaw=yaw_deg, roll=SV_CAMERA_ROLL),
        )
        return self.world.spawn_actor(
            blueprint,
            transform,
            attach_to=self.vehicle,
            attachment_type=carla.AttachmentType.Rigid,
        )

    def _spawn(self):
        self.front_sensor = self._spawn_one(0.0)
        self.rear_sensor = self._spawn_one(180.0)
        self.left_sensor = self._spawn_one(-90.0)
        self.right_sensor = self._spawn_one(90.0)

        self.front_sensor.listen(lambda image: self.front_queue.put(image))
        self.rear_sensor.listen(lambda image: self.rear_queue.put(image))
        self.left_sensor.listen(lambda image: self.left_queue.put(image))
        self.right_sensor.listen(lambda image: self.right_queue.put(image))

    def tick(self):
        while not self.front_queue.empty():
            self.front_surface = image_to_surface(self.front_queue.get())

        while not self.rear_queue.empty():
            self.rear_surface = image_to_surface(self.rear_queue.get())

        while not self.left_queue.empty():
            self.left_surface = image_to_surface(self.left_queue.get())

        while not self.right_queue.empty():
            self.right_surface = image_to_surface(self.right_queue.get())

    def render(self, display, font):
        display.fill((0, 0, 0))

        if self.front_surface is not None:
            display.blit(self.front_surface, (0, 0))
        if self.right_surface is not None:
            display.blit(self.right_surface, (SV_WIDTH, 0))
        if self.left_surface is not None:
            display.blit(self.left_surface, (0, SV_HEIGHT))
        if self.rear_surface is not None:
            display.blit(self.rear_surface, (SV_WIDTH, SV_HEIGHT))

        pygame.draw.line(display, (255, 255, 255), (SV_WIDTH, 0), (SV_WIDTH, WINDOW_HEIGHT), 2)
        pygame.draw.line(display, (255, 255, 255), (0, SV_HEIGHT), (WINDOW_WIDTH, SV_HEIGHT), 2)

        label_color = (255, 255, 0)
        display.blit(font.render("Front", True, label_color), (10, 10))
        display.blit(font.render("Right", True, label_color), (SV_WIDTH + 10, 10))
        display.blit(font.render("Left", True, label_color), (10, SV_HEIGHT + 10))
        display.blit(font.render("Rear", True, label_color), (SV_WIDTH + 10, SV_HEIGHT + 10))

    def destroy(self):
        for sensor in [self.front_sensor, self.rear_sensor, self.left_sensor, self.right_sensor]:
            if sensor is not None:
                sensor.destroy()


class SmoothKeyboardController:
    def __init__(self, vehicle):
        self.vehicle = vehicle
        self.control = carla.VehicleControl()
        self.brake_cmd = 0.0
        self.steer_cmd = 0.0

    def update(self, keys, dt):
        want_forward = keys[K_w]
        want_reverse = keys[K_s]
        want_left = keys[K_a]
        want_right = keys[K_d]
        want_handbrake = keys[K_SPACE]

        speed = self.vehicle.get_velocity()
        speed_mps = math.sqrt(speed.x ** 2 + speed.y ** 2 + speed.z ** 2)
        speed_kmh = speed_mps * 3.6

        target_forward = 0.0
        target_reverse = 0.0
        target_brake = 0.0

        if want_forward and not want_reverse:
            if speed_kmh < 8.0:
                target_forward = 1.0
            elif speed_kmh < FORWARD_SPEED_LIMIT_KMH:
                target_forward = 0.42
        elif want_reverse and not want_forward:
            if speed_kmh < 5.0:
                target_reverse = 0.9
            elif speed_kmh < REVERSE_SPEED_LIMIT_KMH:
                target_reverse = 0.28
        elif want_forward and want_reverse:
            target_brake = 1.0

        steer_target = 0.0
        if want_left:
            steer_target -= 0.45
        if want_right:
            steer_target += 0.45
        reverse_active = want_reverse and not want_forward
        if reverse_active:
            steer_target *= 0.75

        if want_forward and not want_reverse:
            throttle_value = target_forward
        elif want_reverse and not want_forward:
            throttle_value = target_reverse
        else:
            throttle_value = 0.0

        self.brake_cmd = move_towards(self.brake_cmd, target_brake, rate=8.0, dt=dt)
        self.steer_cmd = move_towards(self.steer_cmd, steer_target, rate=4.0, dt=dt)

        if abs(steer_target) < 1e-4:
            self.steer_cmd = move_towards(self.steer_cmd, 0.0, rate=5.0, dt=dt)

        self.control.throttle = clamp(throttle_value, 0.0, 1.0)
        self.control.brake = clamp(self.brake_cmd, 0.0, 1.0)
        self.control.steer = clamp(self.steer_cmd, -1.0, 1.0)
        self.control.hand_brake = want_handbrake
        self.control.reverse = reverse_active
        self.control.manual_gear_shift = False

        if self.control.brake > 0.05:
            self.control.throttle = min(self.control.throttle, 0.08)

        self.vehicle.apply_control(self.control)

        if reverse_active:
            self._apply_reverse_traction_assist(speed_mps)
        elif want_forward and self.control.throttle > 0.05:
            self._apply_forward_traction_assist(speed_mps)
        else:
            self._clear_longitudinal_push()

        return speed_kmh

    def _apply_reverse_traction_assist(self, speed_mps):
        speed_limit_mps = REVERSE_SPEED_LIMIT_KMH / 3.6
        if speed_mps > speed_limit_mps:
            return
        transform = self.vehicle.get_transform()
        forward = transform.get_forward_vector()
        launch_speed = min(speed_limit_mps, 3.2)
        cruise_speed = min(speed_limit_mps, max(1.8, speed_limit_mps * 0.55))
        assist_speed = launch_speed if speed_mps < 0.6 else cruise_speed
        self.vehicle.set_target_velocity(
            carla.Vector3D(
                x=-forward.x * assist_speed,
                y=-forward.y * assist_speed,
                z=0.0,
            )
        )

    def _apply_forward_traction_assist(self, speed_mps):
        speed_limit_mps = FORWARD_SPEED_LIMIT_KMH / 3.6
        if speed_mps > speed_limit_mps:
            return
        transform = self.vehicle.get_transform()
        forward = transform.get_forward_vector()
        launch_speed = min(speed_limit_mps, 6.0)
        cruise_speed = min(speed_limit_mps, max(3.0, speed_limit_mps * 0.55))
        assist_speed = launch_speed if speed_mps < 0.8 else cruise_speed
        self.vehicle.set_target_velocity(
            carla.Vector3D(
                x=forward.x * assist_speed,
                y=forward.y * assist_speed,
                z=0.0,
            )
        )

    def _clear_longitudinal_push(self):
        transform = self.vehicle.get_transform()
        forward = transform.get_forward_vector()
        velocity = self.vehicle.get_velocity()
        longitudinal_speed = velocity.x * forward.x + velocity.y * forward.y

        if abs(longitudinal_speed) < 0.05:
            target_speed = 0.0
        elif longitudinal_speed > 0.0:
            target_speed = max(0.0, longitudinal_speed - 0.25)
        else:
            target_speed = min(0.0, longitudinal_speed + 0.25)

        self.vehicle.set_target_velocity(
            carla.Vector3D(
                x=forward.x * target_speed,
                y=forward.y * target_speed,
                z=velocity.z,
            )
        )

    def reset_state(self):
        self.brake_cmd = 0.0
        self.steer_cmd = 0.0
        self.control = carla.VehicleControl()


def configure_vehicle_physics(vehicle):
    try:
        physics = vehicle.get_physics_control()
    except RuntimeError:
        return

    extent = vehicle.bounding_box.extent
    physics.center_of_mass.x = 0.0
    physics.center_of_mass.y = 0.0
    physics.center_of_mass.z = -min(0.08, max(0.03, extent.z * 0.25))
    physics.mass = max(physics.mass, 80.0)

    for wheel in physics.wheels:
        wheel.damping_rate = max(wheel.damping_rate, 1.5)
        wheel.tire_friction = max(wheel.tire_friction, 2.5)

    vehicle.apply_physics_control(physics)


def set_sync_mode(world):
    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
    settings.substepping = True
    settings.max_substep_delta_time = 0.01
    settings.max_substeps = 10
    world.apply_settings(settings)
    return original_settings


def connect_world(client, attempts=3, wait_seconds=2.0):
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            return client.get_world()
        except RuntimeError as error:
            last_error = error
            print(f"Connect attempt {attempt}/{attempts} failed: {error}")
            if attempt < attempts:
                time.sleep(wait_seconds)
    raise last_error


def resolve_blueprint(bp_library, blueprint_id):
    try:
        return bp_library.find(blueprint_id)
    except RuntimeError:
        matches = bp_library.filter(blueprint_id)
        if matches:
            return matches[0]

    keyword = blueprint_id.lower()
    for blueprint in bp_library:
        if keyword in blueprint.id.lower():
            return blueprint
    raise RuntimeError(f"Cannot find blueprint: {blueprint_id}")


def get_or_spawn_vehicle(world, blueprint_id, role_name, spawn_index, spawn_at_origin):
    blueprint = resolve_blueprint(world.get_blueprint_library(), blueprint_id)
    if blueprint.has_attribute("role_name"):
        blueprint.set_attribute("role_name", role_name)

    spawn_point = carla.Transform(
        carla.Location(
            x=DEFAULT_ORIGIN_SPAWN_TRANSFORM.location.x,
            y=DEFAULT_ORIGIN_SPAWN_TRANSFORM.location.y,
            z=DEFAULT_ORIGIN_SPAWN_TRANSFORM.location.z,
        ),
        carla.Rotation(
            pitch=DEFAULT_ORIGIN_SPAWN_TRANSFORM.rotation.pitch,
            yaw=DEFAULT_ORIGIN_SPAWN_TRANSFORM.rotation.yaw,
            roll=DEFAULT_ORIGIN_SPAWN_TRANSFORM.rotation.roll,
        ),
    )

    vehicle = world.try_spawn_actor(blueprint, spawn_point)
    if vehicle is None:
        raise RuntimeError("Failed to spawn vehicle at selected spawn point")

    print(f"Spawned vehicle: id={vehicle.id}, type={vehicle.type_id}")
    return vehicle


def reset_vehicle(vehicle, controller):
    controller.reset_state()
    current_transform = vehicle.get_transform()
    rescued_transform = carla.Transform(
        carla.Location(
            x=current_transform.location.x,
            y=current_transform.location.y,
            z=current_transform.location.z + RESCUE_LIFT_Z,
        ),
        carla.Rotation(
            pitch=0.0,
            yaw=current_transform.rotation.yaw,
            roll=0.0,
        ),
    )
    vehicle.set_simulate_physics(False)
    vehicle.set_transform(rescued_transform)
    vehicle.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
    vehicle.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
    vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=False))
    vehicle.set_simulate_physics(True)


def draw_hud(display, font_big, font_small, speed_kmh, control, camera_index, view_mode):
    overlay = pygame.Surface((600, 170), pygame.SRCALPHA)
    overlay.fill((10, 14, 18, 180))
    display.blit(overlay, (16, 16))

    line1 = f"Speed: {speed_kmh:5.1f} km/h"
    direction = "Reverse" if control.reverse else "Forward"
    line2 = f"Direction: {direction}   Throttle: {control.throttle:.2f}   Brake: {control.brake:.2f}"
    line3 = f"Steer: {control.steer:.2f}   Hand Brake: {'On' if control.hand_brake else 'Off'}"
    line4 = f"View: {'Main View' if view_mode == VIEW_MAIN else 'Surround View'}"
    line5 = "W forward   S reverse   A/D steer   Space hand brake"
    line6 = f"Tab switch view   R rescue   ESC quit   Main Camera: {camera_index + 1}"

    display.blit(font_big.render(line1, True, (245, 245, 245)), (32, 28))
    display.blit(font_small.render(line2, True, (220, 220, 220)), (32, 62))
    display.blit(font_small.render(line3, True, (220, 220, 220)), (32, 88))
    display.blit(font_small.render(line4, True, (255, 210, 120)), (32, 114))
    display.blit(font_small.render(line5, True, (120, 200, 255)), (32, 138))
    display.blit(font_small.render(line6, True, (255, 210, 120)), (32, 162))


def parse_args():
    parser = argparse.ArgumentParser(description="Keyboard control for a custom CARLA vehicle")
    parser.add_argument("--host", default="127.0.0.1", help="CARLA host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA port")
    parser.add_argument(
        "--blueprint",
        default=DEFAULT_BLUEPRINT_ID,
        help=f"Custom vehicle blueprint id or keyword (default: {DEFAULT_BLUEPRINT_ID})",
    )
    parser.add_argument(
        "--role-name",
        default=DEFAULT_ROLE_NAME,
        help=f"Role name used to reuse existing actor (default: {DEFAULT_ROLE_NAME})",
    )
    parser.add_argument(
        "--spawn-index",
        type=int,
        default=0,
        help="Spawn point index used when a new vehicle is created",
    )
    parser.add_argument(
        "--spawn-at-origin",
        action="store_true",
        help="Deprecated: fixed origin spawn is now the default behavior",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(CLIENT_TIMEOUT_SECONDS)
    world = connect_world(client)
    original_settings = set_sync_mode(world)

    pygame.init()
    pygame.font.init()
    display = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("My Robot Keyboard Control")
    clock = pygame.time.Clock()
    font_big = pygame.font.Font(None, 30)
    font_small = pygame.font.Font(None, 24)

    vehicle = None
    camera = None
    surround_rig = None
    current_view_mode = VIEW_MAIN

    try:
        vehicle = get_or_spawn_vehicle(
            world,
            args.blueprint,
            args.role_name,
            args.spawn_index,
            args.spawn_at_origin,
        )
        configure_vehicle_physics(vehicle)
        controller = SmoothKeyboardController(vehicle)
        camera = FollowCamera(world, vehicle, WINDOW_WIDTH, WINDOW_HEIGHT)
        surround_rig = SurroundCameraRig(world, vehicle)

        running = True
        while running:
            clock.tick(TARGET_FPS)
            world.tick()
            camera.tick()
            surround_rig.tick()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYUP and event.key == K_ESCAPE:
                    running = False
                elif event.type == pygame.KEYUP and event.key == K_TAB:
                    current_view_mode = VIEW_SURROUND if current_view_mode == VIEW_MAIN else VIEW_MAIN
                elif event.type == pygame.KEYUP and event.key == K_r:
                    reset_vehicle(vehicle, controller)

            dt = FIXED_DELTA_SECONDS
            keys = pygame.key.get_pressed()
            speed_kmh = controller.update(keys, dt)

            if current_view_mode == VIEW_MAIN:
                if camera.surface is not None:
                    camera.render(display)
                else:
                    display.fill((18, 18, 18))
            else:
                surround_rig.render(display, font_small)

            draw_hud(
                display,
                font_big,
                font_small,
                speed_kmh,
                controller.control,
                camera._transform_index,
                current_view_mode,
            )
            pygame.display.flip()

    finally:
        if vehicle is not None:
            try:
                vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
            except RuntimeError:
                pass
            try:
                vehicle.destroy()
            except RuntimeError:
                pass

        if camera is not None:
            camera.destroy()

        if surround_rig is not None:
            surround_rig.destroy()

        world.apply_settings(original_settings)
        pygame.quit()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)