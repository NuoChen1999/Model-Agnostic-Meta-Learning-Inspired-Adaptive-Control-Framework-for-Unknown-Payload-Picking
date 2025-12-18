import pybullet as p
import pybullet_data
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

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

class SimpleNN(nn.Module):
    def __init__(self, input_dim=21, hidden_dim=40):
        super(SimpleNN, self).__init__()
        # self.fc1 = nn.Linear(input_dim, hidden_dim)
        # self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # self.fc3 = nn.Linear(hidden_dim, 7)
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 7)
        self.relu = nn.ReLU()

    def forward(self, x, params=None):
        x = x.reshape(x.shape[0], -1)
        if params is None:
            x = torch.tanh(self.fc1(x))
            x = torch.tanh(self.fc2(x))
            x = self.fc3(x)
        else:
            x = torch.tanh(F.linear(x, params['fc1.weight'], params['fc1.bias']))
            x = torch.tanh(F.linear(x, params['fc2.weight'], params['fc2.bias']))
            x = F.linear(x, params['fc3.weight'], params['fc3.bias'])
        return x


def main():
    panda, block = setup_simulation()
    # ==================================================================== #
    #                           Pick up the block                          #
    # ==================================================================== #
    pickup_position = [0.6, 0.1, 0.75]
    pickup_orientation = p.getQuaternionFromEuler([0, np.pi, 0])
    grasp_object(panda, close=False) # Open the gripper
    for _ in range(200):
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
    desired_data = pd.read_csv('Test_x_2.csv')
    desired_q = desired_data.iloc[:, :7].to_numpy()
    desired_qdot = desired_data.iloc[:, 7:14].to_numpy()
    desired_u = desired_data.iloc[:, 14:21].to_numpy()
    num_points = len(desired_data) - 1
    dt = 1 / 240.0
    Kp = np.array([600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0])  # 比例增益
    # Kd = np.array([5.0, 5.0, 5.0, 5.0, 3.0, 2.5, 1.5])  # 微分增益
    Kd = np.array([15, 50, 5, 30, 2, 2, 1])
    num_joints = p.getNumJoints(panda)
    # joint_indices = np.array([0, 1, 2, 3, 4, 5, 6])
    joint_indices = [i for i in range(num_joints) if p.getJointInfo(panda, i)[2] != p.JOINT_FIXED]
    p.setJointMotorControlArray(panda, joint_indices, controlMode=p.VELOCITY_CONTROL, forces=[0] * len(joint_indices))
    joint_indices = np.array([0, 1, 2, 3, 4, 5, 6])
    error = []
    model1 = SimpleNN()
    model1.load_state_dict(torch.load("model_2.pth", weights_only=True))
    a = torch.tensor(np.array([0.1001175 ,  0.27657417, -0.06205704, -0.92647467,  0.11273581,
        0.18845111, -0.05525976]), dtype=torch.float32)
    for i in range(num_points):
        joint_states = p.getJointStates(panda, joint_indices)
        q = np.array([state[0] for state in joint_states])
        zeros = np.zeros((1, 2))
        q_1 = np.hstack((q, zeros[0]))
        q_dot = np.array([state[1] for state in joint_states])
        q_dot_1 = np.hstack((q_dot, zeros[0]))
        real_torque = np.array([state[3] for state in joint_states])

        M_matrix = p.calculateMassMatrix(panda, q_1.tolist())

        M_matrix = np.array(M_matrix)
        print(M_matrix)

        gravity_torques = p.calculateInverseDynamics(panda, q_1.tolist(), q_dot_1.tolist(), [0] * len(q_1))
        input = desired_q[i]
        input = np.hstack((input, desired_qdot[i]))
        input = np.hstack((input, desired_u[i]))
        input = torch.tensor(input.reshape(1,21), dtype=torch.float32)
        y_pred = model1(input)
        a = torch.tensor(a, dtype=torch.float32)
        control_torques = y_pred * a


        pd_torques = Kp * (desired_q[i] - q) + Kd * (desired_qdot[i] - q_dot)
        error.append(q - desired_q[i])
        final_torques = np.array(gravity_torques)[:7] + pd_torques

        grasp_object(panda, close=True)

        p.setJointMotorControlArray(
            bodyUniqueId=panda,
            jointIndices=joint_indices,
            controlMode=p.TORQUE_CONTROL,
            forces=final_torques.tolist()
        )
        real_torque = np.array([state[3] for state in joint_states])
        # print(final_torques)
        p.stepSimulation()
        time.sleep(1.0 / 240.0)
    error = np.array(error)
    plt.figure(figsize=(13, 9))
    plt.plot(error[:, 0], linewidth=4)
    plt.plot(error[:, 1], linewidth=4)
    plt.plot(error[:, 2], linewidth=4)
    plt.plot(error[:, 3], linewidth=4)
    plt.plot(error[:, 4], linewidth=4)
    plt.plot(error[:, 5], linewidth=4)
    plt.plot(error[:, 6], linewidth=4)
    plt.xlabel('Time Step', fontsize='25', fontweight='bold')
    plt.ylabel('Angle Error (rad)', fontsize='25', fontweight='bold')
    plt.xticks([0, 125, 250, 375, 500], fontsize='25')
    plt.yticks([-0.1, 0, 0.1], fontsize='25')
    plt.xlim(0, 500)
    # plt.ylim(-0.06, 0.07)
    plt.grid()
    plt.title('PD Controller, Trajectory 2', fontsize='30', fontweight='bold')
    plt.show()
    error_sum1 = np.sum(np.abs(error[:, 0])) / num_points
    error_sum2 = np.sum(np.abs(error[:, 1])) / num_points
    error_sum3 = np.sum(np.abs(error[:, 2])) / num_points
    error_sum4 = np.sum(np.abs(error[:, 3])) / num_points
    error_sum5 = np.sum(np.abs(error[:, 4])) / num_points
    error_sum6 = np.sum(np.abs(error[:, 5])) / num_points
    error_sum7 = np.sum(np.abs(error[:, 6])) / num_points
    print(error_sum1,error_sum2,error_sum3,error_sum4,error_sum5,error_sum6,error_sum7)


if __name__ == "__main__":
    main()
