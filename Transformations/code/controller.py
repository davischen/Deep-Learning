import pystk
import numpy as np
import math

def control(aim_point, current_vel, steer_gain=2, skid_thresh=0.5, target_vel=25):
    """
    Set the Action for the low-level controller
    :param aim_point: Aim point, in screen coordinate frame [-1..1]
    :param current_vel: Current velocity of the kart
    :return: a pystk.Action (set acceleration, brake, steer, drift)
    """
    action = pystk.Action()

    """
    Your code here
    Hint: Use action.acceleration (0..1) to change the velocity. Try targeting a target_velocity (e.g. 20).
    Hint: Use action.brake to True/False to brake (optionally)
    Hint: Use action.steer to turn the kart towards the aim_point, clip the steer angle to -1..1
    Hint: You may want to use action.drift=True for wide turns (it will turn faster)


    Hint: Skid if the steering angle is too large.

    Hint: Target a constant velocity.

    Hint: Steering and relative aim point use different units. Use the aim point and a tuned scaling factor to select the amount of normalized steering.
    """

    #brake_angle = 110 * math.pi / 180 #剎車角度 = 110 * math.pi / 180
    #drift_angle = 120 * math.pi / 180 #漂移角度 = 120 * math.pi / 180
    #steer_angle = 165 * math.pi / 180 #轉向角度 = 165 * math.pi / 180


    angle = aim_point[0] #math.atan2(aim_point[0], aim_point[1])
    
    steer_angle = steer_gain * angle 
    #brake
    action.brake = True if current_vel >=  target_vel else False
    # Compute accelerate
    # target_vel/current_vel#
    action.acceleration = 1.0 if current_vel < target_vel else 0.0

    # Compute steering
    action.steer = np.clip(steer_angle * steer_gain, -1, 1)
    # Compute skidding #轉向角度
    if abs(steer_angle) > skid_thresh:
        action.drift = True
    else:
        action.drift = False
    action.nitro = True

    return action


if __name__ == '__main__':
    from .utils import PyTux
    from argparse import ArgumentParser

    def test_controller(args):
        import numpy as np
        pytux = PyTux()
        for t in args.track:
            steps, how_far = pytux.rollout(t, control, max_frames=1000, verbose=args.verbose)
            print(steps, how_far)
        pytux.close()


    parser = ArgumentParser()
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    test_controller(args)
