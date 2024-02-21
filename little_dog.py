from functools import partial

import numpy as np
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    JointIndex,
    MeshcatVisualizer,
    Parser,
    PidController,
    Simulator,
    StartMeshcat,
    namedview,
)

from underactuated import ConfigureParser

def set_home(plant, context):
    PositionView = namedview("Positions", plant.GetPositionNames(
        plant.GetModelInstanceByName("LittleDog"), always_add_suffix=False)
    )
    
    hip_roll = 0.1
    hip_pitch = 1
    knee = 1.55
    q0 = PositionView(np.zeros(plant.num_positions()))
    q0.front_right_hip_roll = -hip_roll
    q0.front_right_hip_pitch = hip_pitch
    
    q0.front_right_knee = -knee
    
    q0.front_left_hip_roll = hip_roll
    q0.front_left_hip_pitch = hip_pitch
    
    q0.front_left_knee = -knee
    
    q0.back_right_hip_roll = -hip_roll
    q0.back_right_hip_pitch = -hip_pitch
    q0.back_right_knee = knee
    
    q0.back_left_hip_roll = hip_roll
    q0.back_left_hip_pitch = -hip_pitch
    
    q0.back_left_knee = knee
    
    q0.body_qw = 1.0 #unit quaternion "w" component
    q0.body_z = 0.146 #height of the body
    
    plant.SetPositions(context, q0[:])
    
def run_pid_control(meshcat):
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=1e-3)
    parser = Parser(plant)
    ConfigureParser(parser) #wtf does this do?
    parser.AddModelsFromUrl("package://underactuated/models/littledog/LittleDog.urdf")
    parser.AddModelsFromUrl("package://underactuated/models/littledog/ground.urdf")
    plant.Finalize()
    
    #12 actuators
    kp = 1*np.ones(12)
    ki = 0.0*np.ones(12)
    kd = 0.1*np.ones(12)
    S = np.zeros((24,37)) # 37 joints stuf in total. we only care about 24 of them.
    #only 12 joints in total we care about. 4 hip joints * 2 joint dir + 4 knee joints * 1 joint dir.
    
    num_q = plant.num_positions()
    j = 0
    for i in range(plant.num_joints()):
        joint = plant.get_joint(JointIndex(i))
        if joint.num_positions() != 1:
            # if it is not a hip joint or knee joint move on
            continue
        # S selects joint position and velocities for anything that's not a hip or knee joint.
        S[j,joint.position_start()] = 1
        S[12+j, num_q + joint.velocity_start()] = 1
        
        #if a knee joint use lower kd
        if "knee" in joint.name():
            kd[j] = 0.1
        j = j + 1 # NOTE: j doesn't increment if it's a hip joint.
    
    control = builder.AddSystem(PidController(kp=kp, ki=ki, kd=kd, state_projection=S,
                                              output_projection=plant.MakeActuationMatrix()[6:,:].T))
    
    builder.Connect(plant.get_state_output_port(), control.get_input_port_estimated_state())
    builder.Connect(control.get_output_port_control(), plant.get_actuation_input_port())
    
    visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    
    diagram = builder.Build()
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()
    plant_context = plant.GetMyMutableContextFromRoot(context)
    set_home(plant, plant_context)
    x0 = S @ plant.get_state_output_port().Eval(plant_context)
    control.get_input_port_desired_state().FixValue(control.GetMyContextFromRoot(context), x0)
    
    simulator.set_target_realtime_rate(1.0)
    visualizer.StartRecording()
    simulator.AdvanceTo(30.0)
    visualizer.PublishRecording()

if __name__ == '__main__':
    meshcat = StartMeshcat()
    run_pid_control(meshcat)