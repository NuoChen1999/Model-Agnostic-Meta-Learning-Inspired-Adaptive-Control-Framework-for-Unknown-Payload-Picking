import pybullet as p
import pybullet_data
import time
import numpy as np
import matplotlib.pyplot as plt

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

plane_id = p.loadURDF("plane.urdf")
robot_id = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

end_effector_index = 11

num_joints = p.getNumJoints(robot_id)
active_joints = [i for i in range(num_joints) if p.getJointInfo(robot_id, i)[2] != p.JOINT_FIXED][:7]

center_x, center_y, center_z = 0.5, 0, 0.5
radius = 0.3
num_points = 3000
time_steps = np.linspace(0, 2 * np.pi, num_points)

target_positions = np.array([
    [center_x + radius * np.cos(t), center_y + radius * np.sin(t), center_z]
    for t in time_steps
])

actual_positions = []

for i in active_joints:
    p.setJointMotorControl2(robot_id, i, controlMode=p.VELOCITY_CONTROL, force=0)

dt = 1 / 240.0
for i in range(num_points):
    target_position = target_positions[i]
    target_orientation = p.getQuaternionFromEuler([0, np.pi, 0])

    joint_angles = p.calculateInverseKinematics(robot_id, end_effector_index, target_position, target_orientation)

    joint_angles = joint_angles[:7]

    p.setJointMotorControlArray(
        bodyUniqueId=robot_id,
        jointIndices=active_joints,
        controlMode=p.POSITION_CONTROL,
        targetPositions=joint_angles
    )

    actual_position = p.getLinkState(robot_id, end_effector_index)[0]
    actual_positions.append(actual_position)

    p.stepSimulation()
    time.sleep(dt)

p.disconnect()

actual_positions = np.array(actual_positions)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

ax.plot(target_positions[:, 0], target_positions[:, 1], target_positions[:, 2], 'r--', label="Target Trajectory")

ax.plot(actual_positions[:, 0], actual_positions[:, 1], actual_positions[:, 2], 'b-', label="Actual Trajectory")

# error = np.linalg.norm(target_positions - actual_positions, axis=1)
# mean_error = np.mean(error)
# print(f"Mean tracking error: {mean_error:.4f} m")

ax.set_xlabel("X Position (m)")
ax.set_ylabel("Y Position (m)")
ax.set_zlabel("Z Position (m)")
ax.set_title("End Effector 3D Trajectory Tracking")
ax.legend()
plt.show()
