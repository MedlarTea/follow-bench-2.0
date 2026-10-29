import carla
import pygame
import numpy as np
import math
import queue

HOST = "127.0.0.1"
PORT = 2000

BLUEPRINT_ID = "walker.pedestrian.dog"

SPAWN_X = -74.25
SPAWN_Y = 13.30
SPAWN_Z = 0.6
SPAWN_YAW = 0.0

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

INITIAL_SPEED = 1.0
MIN_SPEED = 0.0
MAX_SPEED = 5.0
SPEED_STEP = 0.2

TURN_RATE_DEG_PER_SEC = 180.0
TURN_BOOST_MULTIPLIER = 1.0
FPS = 30

# Main follow camera
CAMERA_X = -3.5
CAMERA_Y = 0.0
CAMERA_Z = 1.8
CAMERA_PITCH = -10.0
CAMERA_FOV = "90"

# Surround-view camera rig
SV_CAMERA_X = 0.0
SV_CAMERA_Y = 0.0
SV_CAMERA_Z = 0.8
SV_CAMERA_PITCH = 0.0
SV_CAMERA_ROLL = 0.0
SV_CAMERA_FOV = "120"

SV_WIDTH = WINDOW_WIDTH // 2
SV_HEIGHT = WINDOW_HEIGHT // 2

VIEW_MAIN = 0
VIEW_SURROUND = 1


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def make_surface_from_image(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]  # BGR -> RGB
    surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    return surface


def make_camera_blueprint(bp_lib, width, height, fov):
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
    pygame.display.set_caption("CARLA RobotDog Controller with Surround Cameras")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 20)

    client = carla.Client(HOST, PORT)
    client.set_timeout(10.0)
    world = client.get_world()

    bp_lib = world.get_blueprint_library()
    dog_bp = bp_lib.find(BLUEPRINT_ID)

    spawn_transform = carla.Transform(
        carla.Location(x=SPAWN_X, y=SPAWN_Y, z=SPAWN_Z),
        carla.Rotation(yaw=SPAWN_YAW)
    )

    dog = world.try_spawn_actor(dog_bp, spawn_transform)
    if dog is None:
        print("Dog spawn failed. Please check position or collision.")
        pygame.quit()
        return

    print("dog spawned:", dog.type_id)

    # Queues
    main_queue = queue.Queue()
    sv_front_queue = queue.Queue()
    sv_rear_queue = queue.Queue()
    sv_left_queue = queue.Queue()
    sv_right_queue = queue.Queue()

    # Latest surfaces
    main_surface = None
    sv_front_surface = None
    sv_rear_surface = None
    sv_left_surface = None
    sv_right_surface = None

    # Main camera
    main_cam_bp = make_camera_blueprint(bp_lib, WINDOW_WIDTH, WINDOW_HEIGHT, CAMERA_FOV)
    camera = world.spawn_actor(
        main_cam_bp,
        carla.Transform(
            carla.Location(x=CAMERA_X, y=CAMERA_Y, z=CAMERA_Z),
            carla.Rotation(pitch=CAMERA_PITCH)
        ),
        attach_to=dog
    )

    # Surround cameras
    sv_cam_bp = make_camera_blueprint(bp_lib, SV_WIDTH, SV_HEIGHT, SV_CAMERA_FOV)

    sv_front_camera = world.spawn_actor(
        sv_cam_bp,
        carla.Transform(
            carla.Location(x=SV_CAMERA_X, y=SV_CAMERA_Y, z=SV_CAMERA_Z),
            carla.Rotation(pitch=SV_CAMERA_PITCH, yaw=0.0, roll=SV_CAMERA_ROLL)
        ),
        attach_to=dog
    )

    sv_rear_camera = world.spawn_actor(
        sv_cam_bp,
        carla.Transform(
            carla.Location(x=SV_CAMERA_X, y=SV_CAMERA_Y, z=SV_CAMERA_Z),
            carla.Rotation(pitch=SV_CAMERA_PITCH, yaw=180.0, roll=SV_CAMERA_ROLL)
        ),
        attach_to=dog
    )

    sv_left_camera = world.spawn_actor(
        sv_cam_bp,
        carla.Transform(
            carla.Location(x=SV_CAMERA_X, y=SV_CAMERA_Y, z=SV_CAMERA_Z),
            carla.Rotation(pitch=SV_CAMERA_PITCH, yaw=-90.0, roll=SV_CAMERA_ROLL)
        ),
        attach_to=dog
    )

    sv_right_camera = world.spawn_actor(
        sv_cam_bp,
        carla.Transform(
            carla.Location(x=SV_CAMERA_X, y=SV_CAMERA_Y, z=SV_CAMERA_Z),
            carla.Rotation(pitch=SV_CAMERA_PITCH, yaw=90.0, roll=SV_CAMERA_ROLL)
        ),
        attach_to=dog
    )

    def main_callback(image):
        main_queue.put(image)

    def sv_front_callback(image):
        sv_front_queue.put(image)

    def sv_rear_callback(image):
        sv_rear_queue.put(image)

    def sv_left_callback(image):
        sv_left_queue.put(image)

    def sv_right_callback(image):
        sv_right_queue.put(image)

    camera.listen(main_callback)
    sv_front_camera.listen(sv_front_callback)
    sv_rear_camera.listen(sv_rear_callback)
    sv_left_camera.listen(sv_left_callback)
    sv_right_camera.listen(sv_right_callback)

    current_speed = INITIAL_SPEED
    current_yaw = SPAWN_YAW
    move_mode = 0  # 1 forward, -1 backward, 0 stop
    current_view_mode = VIEW_MAIN

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
                        if current_view_mode == VIEW_MAIN:
                            current_view_mode = VIEW_SURROUND
                        else:
                            current_view_mode = VIEW_MAIN
                    elif event.key == pygame.K_q:
                        current_speed = clamp(current_speed - SPEED_STEP, MIN_SPEED, MAX_SPEED)
                    elif event.key == pygame.K_e:
                        current_speed = clamp(current_speed + SPEED_STEP, MIN_SPEED, MAX_SPEED)

            keys = pygame.key.get_pressed()
            frame_dt = clock.get_time() / 1000.0
            if frame_dt <= 0.0:
                frame_dt = 1.0 / FPS

            turn_rate = TURN_RATE_DEG_PER_SEC
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                turn_rate *= TURN_BOOST_MULTIPLIER

            if keys[pygame.K_a] and not keys[pygame.K_d]:
                current_yaw -= turn_rate * frame_dt
            elif keys[pygame.K_d] and not keys[pygame.K_a]:
                current_yaw += turn_rate * frame_dt

            current_yaw = (current_yaw + 180.0) % 360.0 - 180.0

            if keys[pygame.K_SPACE]:
                move_mode = 0
            elif keys[pygame.K_w] and not keys[pygame.K_s]:
                move_mode = 1
            elif keys[pygame.K_s] and not keys[pygame.K_w]:
                move_mode = -1
            else:
                move_mode = 0

            yaw_rad = math.radians(current_yaw)
            dir_x = math.cos(yaw_rad)
            dir_y = math.sin(yaw_rad)

            control = carla.WalkerControl()
            control.jump = False

            if move_mode == 1:
                control.direction = carla.Vector3D(x=dir_x, y=dir_y, z=0.0)
                control.speed = current_speed
                dog.apply_control(control)

                tf = dog.get_transform()
                tf.rotation.yaw = current_yaw
                dog.set_transform(tf)

            elif move_mode == -1:
                control.direction = carla.Vector3D(x=0.0, y=0.0, z=0.0)
                control.speed = 0.0
                dog.apply_control(control)

                tf = dog.get_transform()
                step = current_speed * frame_dt
                tf.location.x -= dir_x * step
                tf.location.y -= dir_y * step
                tf.rotation.yaw = current_yaw
                dog.set_transform(tf)

            else:
                control.direction = carla.Vector3D(x=0.0, y=0.0, z=0.0)
                control.speed = 0.0
                dog.apply_control(control)

                tf = dog.get_transform()
                tf.rotation.yaw = current_yaw
                dog.set_transform(tf)

            # Get latest frames
            while not main_queue.empty():
                main_surface = make_surface_from_image(main_queue.get())

            while not sv_front_queue.empty():
                sv_front_surface = make_surface_from_image(sv_front_queue.get())

            while not sv_rear_queue.empty():
                sv_rear_surface = make_surface_from_image(sv_rear_queue.get())

            while not sv_left_queue.empty():
                sv_left_surface = make_surface_from_image(sv_left_queue.get())

            while not sv_right_queue.empty():
                sv_right_surface = make_surface_from_image(sv_right_queue.get())

            display.fill((0, 0, 0))

            if current_view_mode == VIEW_MAIN:
                if main_surface is not None:
                    display.blit(main_surface, (0, 0))
            else:
                if sv_front_surface is not None:
                    display.blit(sv_front_surface, (0, 0))
                if sv_right_surface is not None:
                    display.blit(sv_right_surface, (SV_WIDTH, 0))
                if sv_left_surface is not None:
                    display.blit(sv_left_surface, (0, SV_HEIGHT))
                if sv_rear_surface is not None:
                    display.blit(sv_rear_surface, (SV_WIDTH, SV_HEIGHT))

                pygame.draw.line(display, (255, 255, 255), (SV_WIDTH, 0), (SV_WIDTH, WINDOW_HEIGHT), 2)
                pygame.draw.line(display, (255, 255, 255), (0, SV_HEIGHT), (WINDOW_WIDTH, SV_HEIGHT), 2)

                label_color = (255, 255, 0)
                display.blit(font.render("Front", True, label_color), (10, 10))
                display.blit(font.render("Right", True, label_color), (SV_WIDTH + 10, 10))
                display.blit(font.render("Left", True, label_color), (10, SV_HEIGHT + 10))
                display.blit(font.render("Rear", True, label_color), (SV_WIDTH + 10, SV_HEIGHT + 10))

            if current_view_mode == VIEW_MAIN:
                view_text = "Main View"
            else:
                view_text = "Surround View"

            info_lines = [
                f"Blueprint: {dog.type_id}",
                f"Speed: {current_speed:.2f}",
                f"Yaw: {current_yaw:.1f}",
                f"Mode: {'Forward' if move_mode == 1 else 'Backward' if move_mode == -1 else 'Stop'}",
                f"View: {view_text}",
                "W/S: Hold Forward / Backward",
                "A/D: Hold Turn Left / Turn Right",
                "SHIFT: Faster Turning",
                "Q/E: Decrease / Increase Speed",
                "TAB: Switch View Mode",
                "SPACE: Hold Stop",
                "ESC: Quit",
            ]

            hud_bg = pygame.Surface((430, 300), pygame.SRCALPHA)
            hud_bg.fill((0, 0, 0, 110))
            display.blit(hud_bg, (0, 0))
            draw_text_block(display, font, info_lines, x=10, y=10)

            pygame.display.flip()
            clock.tick(FPS)

    finally:
        print("cleaning up...")
        for cam in [camera, sv_front_camera, sv_rear_camera, sv_left_camera, sv_right_camera]:
            try:
                if cam is not None:
                    cam.stop()
                    cam.destroy()
            except Exception:
                pass

        try:
            if dog is not None:
                dog.destroy()
        except Exception:
            pass

        pygame.quit()


if __name__ == "__main__":
    main()