from coppeliasim_zmqremoteapi_client import RemoteAPIClient

client = RemoteAPIClient()
sim    = client.require('sim')
simIK  = client.require('simIK')

sim.startSimulation()

# Handles
joint_handles = [sim.getObject(f'/UR5/joint[{i}]') for i in range(6)]
tip_handle    = sim.getObject('/UR5/UR5_tip')
target_handle = sim.getObject('/UR5/UR5_target')

# Set your T_sd (column-major, 12 elements)
T_sd = [0, 0, -1,  1, 0, 0,  0, -1, 0,  -0.5, 0.1, 0.1]
sim.setObjectMatrix(target_handle, -1, T_sd)

# Get initial guess via CoppeliaSim's IK solver
ik_env   = simIK.createEnvironment()
ik_group = simIK.createGroup(ik_env)
simIK.addElementFromScene(
    ik_env, ik_group,
    sim.getObject('/UR5'),
    tip_handle, target_handle,
    simIK.constraint_pose
)

q0, *_ = simIK.findConfig(ik_env, ik_group, joint_handles)
print("Initial guess q0:", q0)

# → Feed q0 into your Newton-Raphson solver

sim.stopSimulation()