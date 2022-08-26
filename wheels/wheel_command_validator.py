class WheelCmdValidator:
    def validate(self, vel_left, vel_right, access_granted):
        if not access_granted:
            vel_left = min(vel_left, 0)
            vel_right = min(vel_right, 0)

        return vel_left, vel_right
