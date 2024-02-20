from pydrake.visualization import AddDefaultVisualization
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.parsing import Parser
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.analysis import Simulator
from pydrake.geometry import StartMeshcat
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.primitives import ConstantVectorSource
import numpy as np
from env_util import AddGround

# ATLAS_PATH = "package://drake_models/atlas/atlas_convex_hull.urdf"
LOCAL_ATLAS = "assets/atlas/atlas_convex_hull.urdf"


if __name__ == '__main__':
    meshcat = StartMeshcat()
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=1e-3)
    parser = Parser(plant)
    # parser.AddModelsFromUrl(ATLAS_PATH)
    parser.AddModels(LOCAL_ATLAS)
    AddGround(plant)
    plant.Finalize()
    
    print(plant.num_actuated_dofs())
    tau = builder.AddSystem(ConstantVectorSource(np.zeros(plant.num_actuated_dofs())))
    
    builder.Connect(tau.get_output_port(0), plant.get_actuation_input_port())
    plant.mutable_gravity_field().set_gravity_vector(np.array([0, 0, -9.81]))
    
    pelvis = plant.GetBodyByName("pelvis")
    
    AddDefaultVisualization(builder, meshcat)
    
    
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyMutableContextFromRoot(context)
    
    plant.SetFreeBodyPose(plant_context, pelvis, RigidTransform(RollPitchYaw(0, 0, 0), [0, 0, 1]))
    
    simulator = Simulator(diagram, context)
    simulator.set_target_realtime_rate(1e-2)
    simulator.AdvanceTo(1.0)