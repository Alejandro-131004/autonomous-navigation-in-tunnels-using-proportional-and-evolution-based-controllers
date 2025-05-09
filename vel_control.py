try:
    from controllers.utils import cmd_vel
except ImportError:
    print("Warning: controllers.utils.cmd_vel not found. Using dummy function.")
    def cmd_vel(supervisor, lv, av):
        # Dummy implementation for testing without the utility file
        left_motor = supervisor.getDevice("left wheel motor")
        right_motor = supervisor.getDevice("right wheel motor")
        wheel_radius = 0.02 # Assuming a standard wheel radius
        axle_track = 0.0565 # Assuming a standard axle track
        left_velocity = (lv - av * axle_track / 2.0) / wheel_radius
        right_velocity = (lv + av * axle_track / 2.0) / wheel_radius
        left_motor.setVelocity(left_velocity)
        right_motor.setVelocity(right_velocity)