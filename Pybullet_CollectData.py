import pybullet as p
import pybullet_data
import time
import numpy as np
import matplotlib.pyplot as plt

def setup_simulation():
    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    table_position = [0, 0, 0]
    table = p.loadURDF("table/table.urdf", table_position, globalScaling=1.0)
    panda_base_position = [-0.3, 0, 0.625]
    panda = p.loadURDF("franka_panda/panda.urdf", basePosition=panda_base_position, useFixedBase=True)
    block_position = [0.49, 0.08, 0.65]
    block = p.loadURDF("cube_small.urdf", block_position, globalScaling=1)
    p.changeDynamics(block, -1, mass= 2.8, lateralFriction=10, rollingFriction=0.2, spinningFriction=0.2)
    return panda, block

def get_joint_indices(robot):
    num_joints = p.getNumJoints(robot)
    joint_indices = [i for i in range(num_joints) if p.getJointInfo(robot, i)[2] != p.JOINT_FIXED]
    return joint_indices

def move_arm_to_target(robot, target_position, target_orientation):
    joint_indices = get_joint_indices(robot)
    joint_positions = p.calculateInverseKinematics(robot, 11, target_position, target_orientation)
    for j, joint_index in enumerate(joint_indices[:7]):
        p.setJointMotorControl2(robot, joint_index, p.POSITION_CONTROL, joint_positions[j])

def grasp_object(robot, close):
    finger_joints = [9, 10]
    grip_position = 0.02 if close else 0.1
    for joint in finger_joints:
        p.setJointMotorControl2(robot, joint, p.POSITION_CONTROL, grip_position, force=100)


def main():
    panda, block = setup_simulation()
    # ==================================================================== #
    #                           Pick up the block                          #
    # ==================================================================== #
    pickup_position = [0.6, 0.1, 0.75]
    pickup_orientation = p.getQuaternionFromEuler([0, np.pi, 0])
    grasp_object(panda, close=False) # Open the gripper
    for _ in range(500):
        p.stepSimulation()
        time.sleep(1 / 240)
    move_arm_to_target(panda, pickup_position, pickup_orientation) # Move to the top of the block
    for _ in range(200):
        p.stepSimulation()
        time.sleep(1 / 240)
    pickup_position[2] = 0.63
    move_arm_to_target(panda, pickup_position, pickup_orientation) # Move to the center of the block
    for _ in range(200):
        p.stepSimulation()
        time.sleep(1 / 240)
    grasp_object(panda, close=True) # Close the gripper
    for _ in range(200):
        p.stepSimulation()
        time.sleep(1 / 240)
    pickup_position = [0.4, 0, 1.3]
    interpolated_positions = np.linspace([0.6, 0.1, 0.63], pickup_position, 300)
    for i in range(300): # Go to the initial position of the trajectory
        move_arm_to_target(panda, interpolated_positions[i], pickup_orientation)
        p.stepSimulation()
        time.sleep(1 / 240)
    for i in range(300): # Stay
        move_arm_to_target(panda, pickup_position, pickup_orientation)
        p.stepSimulation()
        time.sleep(1 / 240)
    # ==================================================================== #
    #                         Define the trajectory                        #
    # ==================================================================== #
    center_x, center_y, center_z = 0.3, 0, 1.3

    # Trajectory 1
    num_points = 1000
    time_steps = np.linspace(0, 2 * np.pi, num_points)
    target_positions = np.array([
        [center_x + 0.1 * np.cos(t), center_y + 0.3 * np.sin(t), center_z]
        for t in time_steps
    ])
    target_orientations = np.array([
        [0, np.pi, 0]
        for t in time_steps
    ])

    # # Trajectory 2
    # num_points = 500
    # time_steps = np.linspace(0, 2 * np.pi, num_points)
    # target_positions = np.array([
    #     [center_x + 0.1 * np.cos(t), center_y + 0.3 * np.sin(t), center_z + 0.1 * np.sin(t)]
    #     for t in time_steps
    # ])
    # target_orientations = np.array([
    #     [0, np.pi * (1 + np.cos(t)), 0]
    #     for t in time_steps
    # ])

    actual_positions = []
    MCG_minus_u_history = []
    q_history = []
    q_dot_history = []
    torque_history = []
    joint_indices = get_joint_indices(panda)
    print(joint_indices)
    dt = 1 / 240.0
    for i in range(num_points):
        target_position = target_positions[i]
        target_orientation = target_orientations[i]
        move_arm_to_target(panda, target_position, target_orientation)

        joint_states = p.getJointStates(panda, joint_indices)
        q = [state[0] for state in joint_states]
        q_dot = [state[1] for state in joint_states]
        # q_ddot = (q_dot - q_dot_history[-1]) / dt
        real_torque = [state[3] for state in joint_states]
        MCG = p.calculateInverseDynamics(panda, q, q_dot, [0] * len(q))
        MCG = np.array(MCG)
        MCG_minus_u_history.append(MCG - real_torque)
        q = np.array(q)
        q_dot = np.array(q_dot)
        real_torque = np.array(real_torque)
        q_history.append(q)
        q_dot_history.append(q_dot)
        torque_history.append(real_torque)
        actual_position = p.getLinkState(panda, 11)[0]
        actual_positions.append(actual_position)

        p.stepSimulation()
        time.sleep(dt)

    # ==================================================================== #
    #                                 Plot                                 #
    # ==================================================================== #
    actual_positions = np.array(actual_positions)
    MCG_minus_u_history = np.array(MCG_minus_u_history)
    q_history = np.array(q_history)
    q_dot_history = np.array(q_dot_history)
    torque_history = np.array(torque_history)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(target_positions[:, 0], target_positions[:, 1], target_positions[:, 2], 'r--', label="Target Trajectory")
    ax.plot(actual_positions[:, 0], actual_positions[:, 1], actual_positions[:, 2], 'b-', label="Actual Trajectory")

    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_zlabel("Z Position (m)")
    # ax.set_xlim(0, 0.7)
    # ax.set_ylim(-0.3, 0.3)
    # ax.set_zlim(0.5, 1)
    ax.set_title("End Effector 3D Trajectory Tracking")
    ax.legend()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(q_history[:, 0], label='joint 1')
    plt.plot(q_history[:, 1], label='joint 2')
    plt.plot(q_history[:, 2], label='joint 3')
    plt.plot(q_history[:, 3], label='joint 4')
    plt.plot(q_history[:, 4], label='joint 5')
    plt.plot(q_history[:, 5], label='joint 6')
    plt.plot(q_history[:, 6], label='joint 7')
    plt.legend()
    plt.show()
    train_x_history = np.hstack((q_history[:, :7], q_dot_history[:, :7], torque_history[:, :7]))
    train_y_history = MCG_minus_u_history[:, :7]
    # np.savetxt('Test_x_2.csv', train_x_history, delimiter=',', comments='', fmt='%.5f')
    # np.savetxt('Test_y_2.csv', train_y_history, delimiter=',', comments='', fmt='%.5f')
    # np.savetxt('Test_4_MCG_minus_u.csv', MCG_minus_u_history, delimiter=',', comments='', fmt='%.5f')
    # np.savetxt('Test_4_q.csv', q_history, delimiter=',', comments='', fmt='%.5f')
    # np.savetxt('Test_4_q_dot.csv', q_dot_history, delimiter=',', comments='', fmt='%.5f')
    # np.savetxt('Test_4_torque.csv', torque_history, delimiter=',', comments='', fmt='%.5f')


if __name__ == "__main__":
    main()
