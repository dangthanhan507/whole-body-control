from pydrake.multibody.plant import MultibodyPlant
from pydrake.geometry import HalfSpace, ProximityProperties
from pydrake.multibody.tree import RigidBody
from pydrake.math import RigidTransform
from contact_lib import AddContactModel

def RegisterShape(plant: MultibodyPlant, name:str, body: RigidBody, 
                  shape, prop: ProximityProperties, color=[1,0,0,1], rt=RigidTransform()):
    
    if plant.geometry_source_is_registered():
        plant.RegisterCollisionGeometry(
            body, rt, shape, name, prop
        )
        plant.RegisterVisualGeometry(
            body, rt, shape, name, color
        )

def AddGround(plant: MultibodyPlant):
    ground_color = [0.5, 1.0, 0.5, 1.0]
    ground_prop = AddContactModel(plant, halfspace_slab=0.5, hydro_mod = 1e4, dissip=100.0, mu_static=1.0, res_hint=1)
    RegisterShape(plant, "GroundVisualGeometry", plant.world_body(), HalfSpace(), ground_prop, ground_color)