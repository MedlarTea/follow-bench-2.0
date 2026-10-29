import carla
import pygame
import numpy as np
import math
import queue


HOST = "127.0.0.1"
PORT = 2000

BLUEPRINT_ID = "walker.pedestrian.g1_unitree"

SPAWN_X = -74.25
SPAWN_Y = 13.30
SPAWN_Z = 1
SPAWN_YAW = 0.0

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

INITIAL_SPEED = 1.0
MIN_SPEED = 0.0
MAX_SPEED = 5.0
SPEED_STEP = 0.2

TURN_RATE_DEG_PER_SEC = 60.0
TURN_BOOST_MULTIPLIER = 1.2
FPS = 30

# First-person / body-mounted camera
FP_CAMERA_X = 0.015
FP_CAMERA_Y = 0.0
FP_CAMERA_Z = 0.90
FP_CAMERA_PITCH = -28.0
FP_CAMERA_YAW = 0.0
FP_CAMERA_ROLL = 0.0

# Third-person follow camera
TP_CAMERA_X = -3.5
TP_CAMERA_Y = 0.0
TP_CAMERA_Z = 1.8
TP_CAMERA_PITCH = -10.0
TP_CAMERA_YAW = 0.0
TP_CAMERA_ROLL = 0.0

# Surround-view camera rig
SV_CAMERA_X = 0.0
SV_CAMERA_Y = 0.0
SV_CAMERA_Z = 1.2
SV_CAMERA_PITCH = -10.0
SV_CAMERA_ROLL = 0.0
SV_FOV = "120"

SV_WIDTH = WINDOW_WIDTH // 2
SV_HEIGHT = WINDOW_HEIGHT // 2

# Head-bob parameters for first-person view
BOB_FREQ_BASE = 1.6
BOB_FREQ_PER_SPEED = 1.2
BOB_Z_AMPLITUDE = 0.045
BOB_Y_AMPLITUDE = 0.02
BOB_PITCH_AMPLITUDE = 1.8
BOB_ROLL_AMPLITUDE = 1.5
BOB_YAW_AMPLITUDE = 0.8

VIEW_FIRST_PERSON = 0
VIEW_THIRD_PERSON = 1
VIEW_SURROUND = 2


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def make_surface_from_image(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    return surface


def make_camera(bp_lib, width, height, fov):
    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(width))
    cam_bp.set_attribute("image_size_y", str(height))
    cam_bp.set_attribute("fov", str(fov))
    return cam_bp


def draw_text_block(display, font, lines, x=10, y=10, color=(255, 255, 255)):
    cur_y = y
    for line in lines:
        text = font.render(line, True, color)
        display.blit(text, (x, cur_y))
        cur_y += 24


def main():
    pygame.init()
    pygame.font.init()

    display = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("CARLA Robot Walker Controller with Surround Cameras")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 20)

    client = carla.Client(HOST, PORT)
    client.set_timeout(10.0)
    world = client.get_world()
    bp_lib = world.get_blueprint_library()

    try:
        robot_bp = bp_lib.find(BLUEPRINT_ID)
    except RuntimeError:
        print(f"Blueprint not found: {BLUEPRINT_ID}")
        pygame.quit()
        return

    spawn_transform = carla.Transform(
        carla.Location(x=SPAWN_X, y=SPAWN_Y, z=SPAWN_Z),
        carla.Rotation(yaw=SPAWN_YAW)
    )

    robot = world.try_spawn_actor(robot_bp, spawn_transform)
    if robot is None:
        print("Robot spawn failed. Please check spawn position or collision.")
        pygame.quit()
        return

    print("robot spawned:", robot.type_id)

    # Queues and latest frames
    fp_queue = queue.Queue()
    fp_latest_surface = None

    sv_front_queue = queue.Queue()
    sv_rear_queue = queue.Queue()
    sv_left_queue = queue.Queue()
    sv_right_queue = queue.Queue()

    sv_front_surface = None
    sv_rear_surface = None
    sv_left_surface = None
    sv_right_surface = None

    fp_camera = None
    sv_front_camera = None
    sv_rear_camera = None
    sv_left_camera = None
    sv_right_camera = None

    # First-person / third-person shared main camera
    main_cam_bp = make_camera(bp_lib, WINDOW_WIDTH, WINDOW_HEIGHT, 120)
    fp_camera = world.spawn_actor(
        main_cam_bp,
        carla.Transform(
            carla.Location(x=FP_CAMERA_X, y=FP_CAMERA_Y, z=FP_CAMERA_Z),
            carla.Rotation(
                pitch=FP_CAMERA_PITCH,
                yaw=FP_CAMERA_YAW,
                roll=FP_CAMERA_ROLL
            )
        ),
        attach_to=robot
    )

    # Surround-view cameras
    sv_cam_bp = make_camera(bp_lib, SV_WIDTH, SV_HEIGHT, SV_FOV)

    sv_front_camera = world.spawn_actor(
        sv_cam_bp,
        carla.Transform(
            carla.Location(x=SV_CAMERA_X, y=SV_CAMERA_Y, z=SV_CAMERA_Z),
            carla.Rotation(pitch=SV_CAMERA_PITCH, yaw=0.0, roll=SV_CAMERA_ROLL)
        ),
        attach_to=robot
    )

    sv_rear_camera = world.spawn_actor(
        sv_cam_bp,
        carla.Transform(
            carla.Location(x=SV_CAMERA_X, y=SV_CAMERA_Y, z=SV_CAMERA_Z),
            carla.Rotation(pitch=SV_CAMERA_PITCH, yaw=180.0, roll=SV_CAMERA_ROLL)
        ),
        attach_to=robot
    )

    sv_left_camera = world.spawn_actor(
        sv_cam_bp,
        carla.Transform(
            carla.Location(x=SV_CAMERA_X, y=SV_CAMERA_Y, z=SV_CAMERA_Z),
            carla.Rotation(pitch=SV_CAMERA_PITCH, yaw=-90.0, roll=SV_CAMERA_ROLL)
        ),
        attach_to=robot
    )

    sv_right_camera = world.spawn_actor(
        sv_cam_bp,
        carla.Transform(
            carla.Location(x=SV_CAMERA_X, y=SV_CAMERA_Y, z=SV_CAMERA_Z),
            carla.Rotation(pitch=SV_CAMERA_PITCH, yaw=90.0, roll=SV_CAMERA_ROLL)
        ),
        attach_to=robot
    )

    def fp_callback(image):
        fp_queue.put(image)

    def sv_front_callback(image):
        sv_front_queue.put(image)

    def sv_rear_callback(image):
        sv_rear_queue.put(image)

    def sv_left_callback(image):
        sv_left_queue.put(image)

    def sv_right_callback(image):
        sv_right_queue.put(image)

    fp_camera.listen(fp_callback)
    sv_front_camera.listen(sv_front_callback)
    sv_rear_camera.listen(sv_rear_callback)
    sv_left_camera.listen(sv_left_callback)
    sv_right_camera.listen(sv_right_callback)

    current_speed = INITIAL_SPEED
    current_yaw = SPAWN_YAW
    sim_time = 0.0
    current_view_mode = VIEW_FIRST_PERSON

    try:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_TAB:
                        current_view_mode = (current_view_mode + 1) % 3

            keys = pygame.key.get_pressed()
            frame_dt = clock.get_time() / 1000.0
            if frame_dt <= 0.0:
                frame_dt = 1.0 / FPS
            sim_time += frame_dt

            # Q/E: decrease / increase speed
            if keys[pygame.K_q] and not keys[pygame.K_e]:
                current_speed = clamp(current_speed - SPEED_STEP, MIN_SPEED, MAX_SPEED)
            elif keys[pygame.K_e] and not keys[pygame.K_q]:
                current_speed = clamp(current_speed + SPEED_STEP, MIN_SPEED, MAX_SPEED)

            # A/D: turn left / turn right
            turn_rate = TURN_RATE_DEG_PER_SEC
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                turn_rate *= TURN_BOOST_MULTIPLIER

            if keys[pygame.K_a] and not keys[pygame.K_d]:
                current_yaw -= turn_rate * frame_dt
            elif keys[pygame.K_d] and not keys[pygame.K_a]:
                current_yaw += turn_rate * frame_dt

            current_yaw = (current_yaw + 180.0) % 360.0 - 180.0

            yaw_rad = math.radians(current_yaw)
            forward_x = math.cos(yaw_rad)
            forward_y = math.sin(yaw_rad)

            move_forward = 1.0 if keys[pygame.K_w] else 0.0
            if keys[pygame.K_SPACE]:
                move_forward = 0.0

            is_moving = move_forward > 0.0 and current_speed > 0.0

            control = carla.WalkerControl()
            control.jump = False

            if is_moving:
                control.direction = carla.Vector3D(x=forward_x, y=forward_y, z=0.0)
                control.speed = current_speed
            else:
                control.direction = carla.Vector3D(x=0.0, y=0.0, z=0.0)
                control.speed = 0.0

            robot.apply_control(control)

            tf = robot.get_transform()
            tf.rotation.yaw = current_yaw
            robot.set_transform(tf)

            # Update main shared camera
            if current_view_mode == VIEW_FIRST_PERSON:
                cam_x = FP_CAMERA_X
                cam_y = FP_CAMERA_Y
                cam_z = FP_CAMERA_Z
                cam_pitch = FP_CAMERA_PITCH
                cam_yaw = FP_CAMERA_YAW
                cam_roll = FP_CAMERA_ROLL

                if is_moving:
                    phase = sim_time * 2.0 * math.pi * (
                        BOB_FREQ_BASE + current_speed * BOB_FREQ_PER_SPEED
                    )
                    speed_alpha = current_speed / MAX_SPEED

                    cam_z += math.sin(phase) * BOB_Z_AMPLITUDE * speed_alpha
                    cam_y += math.sin(phase * 0.5) * BOB_Y_AMPLITUDE * speed_alpha
                    cam_pitch += math.sin(phase) * BOB_PITCH_AMPLITUDE * speed_alpha
                    cam_roll += math.sin(phase * 0.5) * BOB_ROLL_AMPLITUDE * speed_alpha
                    cam_yaw += math.sin(phase * 0.25) * BOB_YAW_AMPLITUDE * speed_alpha

                fp_camera.set_transform(carla.Transform(
                    carla.Location(x=cam_x, y=cam_y, z=cam_z),
                    carla.Rotation(
                        pitch=cam_pitch,
                        yaw=cam_yaw,
                        roll=cam_roll
                    )
                ))
            else:
                # Third-person follow view: no bobbing
                fp_camera.set_transform(carla.Transform(
                    carla.Location(x=TP_CAMERA_X, y=TP_CAMERA_Y, z=TP_CAMERA_Z),
                    carla.Rotation(
                        pitch=TP_CAMERA_PITCH,
                        yaw=TP_CAMERA_YAW,
                        roll=TP_CAMERA_ROLL
                    )
                ))

            # Update latest frames
            while not fp_queue.empty():
                fp_latest_surface = make_surface_from_image(fp_queue.get())

            while not sv_front_queue.empty():
                sv_front_surface = make_surface_from_image(sv_front_queue.get())

            while not sv_rear_queue.empty():
                sv_rear_surface = make_surface_from_image(sv_rear_queue.get())

            while not sv_left_queue.empty():
                sv_left_surface = make_surface_from_image(sv_left_queue.get())

            while not sv_right_queue.empty():
                sv_right_surface = make_surface_from_image(sv_right_queue.get())

            display.fill((0, 0, 0))

            # Render
            if current_view_mode in (VIEW_FIRST_PERSON, VIEW_THIRD_PERSON):
                if fp_latest_surface is not None:
                    display.blit(fp_latest_surface, (0, 0))
            else:
                # 2x2 surround view layout
                if sv_front_surface is not None:
                    display.blit(sv_front_surface, (0, 0))
                if sv_right_surface is not None:
                    display.blit(sv_right_surface, (SV_WIDTH, 0))
                if sv_left_surface is not None:
                    display.blit(sv_left_surface, (0, SV_HEIGHT))
                if sv_rear_surface is not None:
                    display.blit(sv_rear_surface, (SV_WIDTH, SV_HEIGHT))

                # separators
                pygame.draw.line(display, (255, 255, 255), (SV_WIDTH, 0), (SV_WIDTH, WINDOW_HEIGHT), 2)
                pygame.draw.line(display, (255, 255, 255), (0, SV_HEIGHT), (WINDOW_WIDTH, SV_HEIGHT), 2)

                label_color = (255, 255, 0)
                display.blit(font.render("Front", True, label_color), (10, 10))
                display.blit(font.render("Right", True, label_color), (SV_WIDTH + 10, 10))
                display.blit(font.render("Left", True, label_color), (10, SV_HEIGHT + 10))
                display.blit(font.render("Rear", True, label_color), (SV_WIDTH + 10, SV_HEIGHT + 10))

            if keys[pygame.K_SPACE]:
                mode_text = "Stop"
            elif is_moving:
                mode_text = "Forward"
            else:
                mode_text = "Idle"

            if current_view_mode == VIEW_FIRST_PERSON:
                view_text = "First-Person"
            elif current_view_mode == VIEW_THIRD_PERSON:
                view_text = "Third-Person"
            else:
                view_text = "Surround View"

            info_lines = [
                f"Blueprint: {robot.type_id}",
                f"Speed: {current_speed:.2f}",
                f"Yaw: {current_yaw:.1f}",
                f"Mode: {mode_text}",
                f"View: {view_text}",
                "W: Move Forward",
                "A/D: Turn Left / Turn Right",
                "Q/E: Decrease Speed / Increase Speed",
                "TAB: Switch View Mode",
                "SHIFT: Faster Turning",
                "SPACE: Stop",
                "ESC: Quit",
            ]

            # Draw HUD background
            hud_bg = pygame.Surface((420, 300), pygame.SRCALPHA)
            hud_bg.fill((0, 0, 0, 110))
            display.blit(hud_bg, (0, 0))
            draw_text_block(display, font, info_lines, x=10, y=10)

            pygame.display.flip()
            clock.tick(FPS)

    finally:
        print("cleaning up...")
        for cam in [fp_camera, sv_front_camera, sv_rear_camera, sv_left_camera, sv_right_camera]:
            try:
                if cam is not None:
                    cam.stop()
                    cam.destroy()
            except Exception:
                pass

        try:
            if robot is not None:
                robot.destroy()
        except Exception:
            pass

        pygame.quit()


if __name__ == "__main__":
    main()