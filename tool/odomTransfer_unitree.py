import asyncio
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber, ChannelPublisher
from unitree_sdk2py.idl.std_msgs.msg.dds_ import Header_
from unitree_sdk2py.idl.builtin_interfaces.msg.dds_ import Time_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import TimeSpec_
from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import Quaternion_, Pose_, Point_, Twist_, Vector3_, PoseWithCovariance_, TwistWithCovariance_
from unitree_sdk2py.idl.nav_msgs.msg.dds_ import Odometry_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_


class TopicTransfer:
    def __init__(self):
        self.sportstate_subscriber = ChannelSubscriber("rt/lf/sportmodestate", SportModeState_)
        self.sportstate_subscriber.Init(self.SportModeStateMessageHandler, 50)

        self.odom_publisher = ChannelPublisher("rt/robot_odom", Odometry_)
        self.odom_publisher.Init()

        asyncio.create_task(self.odom_publish())
        
        # Create a default message instance during initialization to ensure that latest_odom_msg exists
        self.latest_odom_msg = self._create_default_odom()

    def _create_default_odom(self):
        """Create a default Odometry_ instance, containing all required parameters"""
        return Odometry_(
            header=Header_(
                stamp=Time_(sec=0, nanosec=0),
                frame_id="odom"
            ),
            child_frame_id="base_link",
            pose=PoseWithCovariance_(
                pose=Pose_(
                    position=Point_(x=0.0, y=0.0, z=0.0),
                    orientation=Quaternion_(x=0.0, y=0.0, z=0.0, w=1.0)  
                ),
                covariance=[0.0] * 36
            ),
            twist=TwistWithCovariance_(
                twist=Twist_(
                    linear=Vector3_(x=0.0, y=0.0, z=0.0),  # Vector3_ need x、y、z
                    angular=Vector3_(x=0.0, y=0.0, z=0.0)  # Vector3_ need x、y、z
                ),
                covariance=[0.0] * 36
            )
        )

    async def odom_publish(self):
        while True:
            try:
                if self.latest_odom_msg is not None:
                    self.odom_publisher.Write(self.latest_odom_msg)
                else:
                    print("Odom publish warning: latest_odom_msg is None, use default value")
            except TypeError as e:
                print(f"Odom publish error: {e}")
            await asyncio.sleep(0.02)

    def SportModeStateMessageHandler(self, msg: SportModeState_):
        # print("Received the sportmodestate msg")

        # Process the timestamp
        sec = msg.stamp.sec if isinstance(msg.stamp, TimeSpec_) else 0
        nanosec = msg.stamp.nanosec if isinstance(msg.stamp, TimeSpec_) else 0
        
        # Create the Odometry_ instance, ensuring all struct parameters are complete
        odom_msg = Odometry_(
            header=Header_(
                stamp=Time_(sec=sec, nanosec=nanosec),
                frame_id="odom"
            ),
            child_frame_id="base_link",
            pose=PoseWithCovariance_(
                pose=Pose_(
                    position=Point_(
                        x=msg.position[0] if hasattr(msg, 'position') else 0.0,
                        y=msg.position[1] if hasattr(msg, 'position') else 0.0,
                        z=msg.position[2] if hasattr(msg, 'position') else 0.0
                    ),
                    orientation=Quaternion_(
                        x=msg.imu_state.quaternion[1] if hasattr(msg.imu_state, 'quaternion') else 0.0,
                        y=msg.imu_state.quaternion[2] if hasattr(msg.imu_state, 'quaternion') else 0.0,
                        z=msg.imu_state.quaternion[3] if hasattr(msg.imu_state, 'quaternion') else 0.0,
                        w=msg.imu_state.quaternion[0] if hasattr(msg.imu_state, 'quaternion') else 1.0
                    )
                ),
                covariance=[0.0] * 36
            ),
            twist=TwistWithCovariance_(
                twist=Twist_(
                    linear=Vector3_(
                        x=msg.velocity[0] if hasattr(msg, 'velocity') else 0.0,
                        y=msg.velocity[1] if hasattr(msg, 'velocity') else 0.0,
                        z=msg.velocity[2] if hasattr(msg, 'velocity') else 0.0
                    ),
                    angular=Vector3_(
                        x=msg.imu_state.rpy[0] if hasattr(msg.imu_state, 'rpy') else 0.0,
                        y=msg.imu_state.rpy[1] if hasattr(msg.imu_state, 'rpy') else 0.0,
                        z=msg.imu_state.rpy[2] if hasattr(msg.imu_state, 'rpy') else 0.0
                    )
                ),
                covariance=[0.0] * 36
            )
        )
        
        self.latest_odom_msg = odom_msg
        # print("Updated the latest odom msg")


async def main():
    ChannelFactoryInitialize(0, "enP8p1s0")
    node = TopicTransfer()
    await asyncio.Future()

if __name__ == '__main__':
    asyncio.run(main())

