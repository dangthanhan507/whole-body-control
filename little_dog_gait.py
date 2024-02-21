from functools import partial

import numpy as np
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    AddUnitQuaternionConstraintOnPlant,
    AutoDiffXd,
    DiagramBuilder,
    ExtractGradient,
    ExtractValue,
    InitializeAutoDiff,
    JacobianWrtVariable,
    JointIndex,
    MathematicalProgram,
    MeshcatVisualizer,
    OrientationConstraint,
    Parser,
    PidController,
    PiecewisePolynomial,
    PositionConstraint,
    RotationMatrix,
    Simulator,
    SnoptSolver,
    Solve,
    StartMeshcat,
    eq,
    namedview,
)
from underactuated import ConfigureParser
import time

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
    
def autoDiffArrayEqual(a, b):
    return np.array_equal(a, b) and np.array_equal(
        ExtractGradient(a), ExtractGradient(b)
    )   

def gait_optimization(meshcat, gait="walking_trot"):
    start_timer = time.time()
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=1e-3)
    parser = Parser(plant)
    ConfigureParser(parser) #wtf does this do?
    little_dog = parser.AddModelsFromUrl("package://underactuated/models/littledog/LittleDog.urdf")[0]
    ground = parser.AddModelsFromUrl("package://underactuated/models/littledog/ground.urdf")[0]
    
    plant.Finalize()
    visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)
    set_home(plant, plant_context)
    diagram.ForcedPublish(context)
    
    q0 = plant.GetPositions(plant_context)
    body_frame = plant.GetFrameByName("body")
    
    PositionView = namedview("Positions", plant.GetPositionNames(little_dog, always_add_suffix=False))
    VelocityView = namedview("Velocities", plant.GetVelocityNames(little_dog, always_add_suffix=False))
    
    # rubber on rubber
    mu = 1
    total_mass = plant.CalcTotalMass(plant_context, [little_dog])
    gravity = plant.gravity_field().gravity_vector()
    
    #number of joints that matter for us
    nq = 12
    foot_frame = [
        plant.GetFrameByName("front_left_foot_center"),
        plant.GetFrameByName("front_right_foot_center"),
        plant.GetFrameByName("back_left_foot_center"),
        plant.GetFrameByName("back_right_foot_center"),
    ]
    
    #setup gait
    is_laterally_symmetric = False
    check_self_collision = False
    
    '''
        This pre-determines the contact sequences for the feet.
        So this will be a fixed sequence hybrid trajectory optimization problem.
    '''
    if gait == "running_trot":
        N = 21
        in_stance = np.zeros((4,N)) # label 0 if leg is in air, 1 if leg is in stance
        in_stance[1,  3:17] = 1 # front right in stance
        in_stance[2,  3:17] = 1 # front left in stance
        speed = 0.9
        stride_length = 0.5
        is_laterally_symmetric = True
    elif gait == "walking_trot":
        N = 21
        in_stance = np.zeros((4,N))
        in_stance[0, :11] = 1 # back right in stance
        in_stance[1, 8:N] = 1 # front right in stance
        in_stance[2, 8:N] = 1 # front left in stance
        in_stance[3, :11] = 1 # back left in stance
        speed = 0.4
        stride_length = 0.25
        is_laterally_symmetric = True
    elif gait == "rotary_gallop":
        N = 41
        in_stance = np.zeros((4,N))
        in_stance[0, 7:19] = 1
        in_stance[1, 3:15] = 1
        in_stance[2, 24:35] = 1
        in_stance[3, 26:38] = 1
        speed = 1
        stride_length = 0.65
        check_self_collision = True
    elif gait == "bound":
        N = 41
        in_stance = np.zeros((4, N))
        in_stance[0, 6:18] = 1
        in_stance[1, 6:18] = 1
        in_stance[2, 21:32] = 1
        in_stance[3, 21:32] = 1
        speed = 1.2
        stride_length = 0.55
        check_self_collision = True
    else:
        raise RuntimeError("Unknown gait: " + gait)

    T = stride_length / speed
    
    if is_laterally_symmetric:
        T = T / 2.0
    
    # now for the good stuff
    prog = MathematicalProgram()
    
    # Time steps
    # apparently time steps we can optimize as a free variable (will be part of CoM timestepping)
    h = prog.NewContinuousVariables(N-1, "h")
    prog.AddBoundingBoxConstraint(0.5*T / N, 2.0*T / N, h)
    prog.AddLinearConstraint(sum(h) >= 0.9 * T)
    prog.AddLinearConstraint(sum(h) <= 1.1 * T)
    prog.SetInitialGuess(h, np.full(N-1, T))
    
    # Create one context per time step (to maximize cache hits)
    # NOTE: WTF?
    context = [plant.CreateDefaultContext() for _ in range(N)]
    ad_plant = plant.ToAutoDiffXd()
    
    # Joint pos and vel
    nq = plant.num_positions()
    nv = plant.num_velocities()
    q = prog.NewContinuousVariables(nq, N, "q")
    v = prog.NewContinuousVariables(nv, N, "v")
    q_view = PositionView(q)
    v_view = VelocityView(v)
    q0_view = PositionView(q0)
    
    #joint costs
    q_cost = PositionView([1]*nq)
    v_cost = VelocityView([1]*nv)
    
    # do not punish the movement of the CoM (body) in the cost function
    q_cost.body_x = 0
    q_cost.body_y = 0
    q_cost.body_qx = 0
    q_cost.body_qy = 0
    q_cost.body_qz = 0
    q_cost.body_qw = 0
    
    # only punish vz,vy for robot to stay upright and move laterally in the cost function
    v_cost.body_vx = 0
    v_cost.body_wx = 0
    v_cost.body_wy = 0
    v_cost.body_wz = 0
    
    # punish the movement of hip roll
    q_cost.front_left_hip_roll = 5
    q_cost.front_right_hip_roll = 5
    q_cost.back_left_hip_roll = 5
    q_cost.back_right_hip_roll = 5
    
    for n in range(N):
        #joint limits
        prog.AddBoundingBoxConstraint(
            plant.GetPositionLowerLimits(),
            plant.GetPositionUpperLimits(),
            q[:,n]
        )
        #joint vel limits
        prog.AddBoundingBoxConstraint(
            plant.GetVelocityLowerLimits(),
            plant.GetVelocityUpperLimits(),
            v[:,n]
        )
        
        #make sure quaternions in the joints stay normalized
        AddUnitQuaternionConstraintOnPlant(plant, q[:,n], prog)
        
        #body orientation
        prog.AddConstraint(
            OrientationConstraint(
                plant, 
                body_frame, 
                RotationMatrix(), 
                plant.world_frame(), 
                RotationMatrix(), 
                0.1,
                context[n]
            ),
            q[:,n]
        ) #make sure at each time step the joints follow the same body orientation
        #NOTE: before there was no cost on body, but it's because we threw that into the constraints
        
        
        #setup initial guess
        prog.SetInitialGuess(q[:,n], q0)
        
        #running costs
        ##################3
        #make sure we stay as close to intial position as possible
        prog.AddQuadraticErrorCost(np.diag(q_cost), q0, q[:,n]) 
        prog.AddQuadraticErrorCost(np.diag(v_cost), [0] * nv, v[:,n]) #no punish velocity
        ##################3
    
    #make autodiff context for constraint (maximize cache hits)
    #NOTE: still don't know wtf this means.    
    ad_velocity_dynamics_context = [ad_plant.CreateDefaultContext() for _ in range(N)]
    
    
    def velocity_dynamics_constraint(vars, context_index):
        h, q, v, qn = np.split(vars, [1, 1+ nq, 1 + nq + nv])
        if isinstance(vars[0], AutoDiffXd):
            if not autoDiffArrayEqual(
                q,
                ad_plant.GetPositions(ad_velocity_dynamics_context[context_index])
            ):
                ad_plant.SetPositions(ad_velocity_dynamics_context[context_index], q)
                
                ## END OF IF STATEMENT
            
            v_from_qdot = ad_plant.MapQDotToVelocity(
                ad_velocity_dynamics_context[context_index], (qn - q) / h
            )
        else:
            if not np.array_equal(q, plant.GetPositions(context[context_index])):
                # if context is not the same as the current q, set the context to the current q
                plant.SetPositions(context[context_index], q)
            v_from_qdot = plant.MapQDotToVelocity(
                context[context_index], (qn - q) / h
            )
        
        return v - v_from_qdot
    
    for n in range(N-1):
        prog.AddConstraint(
            partial(velocity_dynamics_constraint, context_index=n),
            lb = [0]*nv,
            ub = [0]*nv,
            vars = np.concatenate(([h[n]], q[:,n], v[:,n], q[:,n+1]))
        )
        
    contact_force = [
        prog.NewContinuousVariables(3, N-1, f"foot{foot}_contact_force")
        for foot in range(4)
    ]
    
    for n in range(N-1):
        for foot in range(4):
            # Linear friction cone
            
            # square pyramid constraint
            prog.AddLinearConstraint(
                contact_force[foot][0,n] <= mu * contact_force[foot][2,n]
            )
            
            prog.AddLinearConstraint(
                -contact_force[foot][0,n] <= mu * contact_force[foot][2,n]
            )
            
            prog.AddLinearConstraint(
                contact_force[foot][1,n] <= mu * contact_force[foot][2,n]
            )
            
            prog.AddLinearConstraint(
                -contact_force[foot][1,n] <= mu * contact_force[foot][2,n]
            )
            
            #normal force >= 0 if in stance, normal_force == 0, if not in stance
            prog.AddBoundingBoxConstraint(
                0,
                in_stance[foot,n] * 4 * 9.81 * total_mass,
                contact_force[foot][2,n]
            ) # mass * grav * 4 if should be in stance
            
    # Center of mass 
    com = prog.NewContinuousVariables(3, N, "com")
    comdot = prog.NewContinuousVariables(3, N, "comdot")
    comddot = prog.NewContinuousVariables(3, N-1, "comddot")
    
    # Initial CoM x,y pos == 0
    prog.AddBoundingBoxConstraint(0, 0, com[:2, 0])
    
    # Intiial CoM z vel == 0
    prog.AddBoundingBoxConstraint(0, 0, comdot[2, 0])
    
    # CoM height bound to be z > 0.125 (stance height)
    prog.AddBoundingBoxConstraint(0.125, np.inf, com[2, :])
    
    # CoM x velocity >= 0
    prog.AddBoundingBoxConstraint(0, np.inf, comdot[0, :])
    
    # CoM final x position    
    if is_laterally_symmetric:
        prog.AddBoundingBoxConstraint(
            stride_length / 2.0, stride_length / 2.0, com[0, -1]
        )
    else:
        prog.AddBoundingBoxConstraint(stride_length, stride_length, com[0, -1])
    
    # CoM dynamics constraint
    for n in range(N-1):
        # discretized steps
        
        # this is super simple CoM forward euler integration
        prog.AddConstraint(eq(com[:,n+1], com[:,n] + h[n] * comdot[:,n]))
        prog.AddConstraint(eq(comdot[:,n+1], comdot[:,n] + h[n] * comddot[:,n]))
        
        # enforce that the CoM and contact forces cancel out    
        prog.AddConstraint(
            eq(
                total_mass * comddot[:,n],
                sum(contact_force[i][:,n] for i in range(4)) + total_mass * gravity
            )
        )
    
    # Angular Momentum
    H = prog.NewContinuousVariables(3, N, "H")
    Hdot = prog.NewContinuousVariables(3, N-1, "Hdot")
    prog.SetInitialGuess(H, np.zeros((3, N)))
    prog.SetInitialGuess(Hdot, np.zeros((3, N-1)))
    
    # Hdot = sum_i_cross(p_FootiW-com, contact_force_i)
    # this is just a way of calculating the angular momentum based on contact force
    # and the distance of leg from the CoM
    
    def angular_momentum_constraint(vars, context_index):
        q, com, Hdot, contact_force = np.split(vars, [nq, nq+3, nq+6])
        contact_force = contact_force.reshape((3, 4), order='F')
        
        if isinstance(vars[0], AutoDiffXd):
            dq = ExtractGradient(q)
            q = ExtractValue(q)
            if not np.array_equal(q, plant.GetPositions(context[context_index])):
                plant.SetPositions(context[context_index], q)
            torque = np.zeros(3)
            for i in range(4):
                p_WF = plant.CalcPointsPositions(
                    context[context_index],
                    foot_frame[i],
                    [0, 0, 0],
                    plant.world_frame()
                )
                Jq_WF = plant.CalcJacobianTranslationalVelocity(
                    context[context_index],
                    JacobianWrtVariable.kQDot,
                    foot_frame[i],
                    [0, 0, 0],
                    plant.world_frame(),
                    plant.world_frame()
                )
                
                #returns autodiff matrix of p_WF
                ad_p_WF = InitializeAutoDiff(p_WF, Jq_WF @ dq) 
                
                # angular momentum equation listed above
                torque = torque + np.cross(
                    ad_p_WF.reshape(3) - com, contact_force[:,i]
                )
            
        else:
            if not np.array_equal(q, plant.GetPositions(context[context_index])):
                plant.SetPositions(context[context_index], q)
            torque = np.zeros(3)
            for i in range(4):
                p_WF = plant.CalcPointsPositions(
                    context[context_index],
                    foot_frame[i],
                    [0, 0, 0],
                    plant.world_frame()
                )
                torque += np.cross(p_WF.reshape(3) - com, contact_force[:,i])
                
        return Hdot - torque

    for n in range(N-1):
        # default forward euler integration
        prog.AddConstraint(eq(H[:,n+1], H[:,n] + h[n] * Hdot[:,n]))
        
        # 
        Fn = np.concatenate([contact_force[i][:,n] for i in range(4)])
        prog.AddConstraint(
            partial(angular_momentum_constraint, context_index=n),
            lb=np.zeros(3),
            ub=np.zeros(3),
            vars = np.concatenate([q[:,n], com[:,n], Hdot[:,n], Fn])
        ) # NOTE: this is the constraint that makes sure the angular momentum is conserved
    
    com_constraint_context = [ad_plant.CreateDefaultContext() for _ in range(N)]
    
    # Bruh how much constraints do we have to add.
    # constraint:
    # com == CenterOfMass(q), H = SpatialMomentumInWOrldAboutPoint(q,v,com)
    
    # this just makes sure that our CoM when we update all of it is
    # consistent with the LittleDog model
    def com_constraint(vars, context_index):
        qv, com, H = np.split(vars, [nq + nv, nv + nq+3])
        if isinstance(vars[0], AutoDiffXd):
            if not autoDiffArrayEqual(
                qv,
                ad_plant.GetPositionsAndVelocities(com_constraint_context[context_index])
            ):
                ad_plant.SetPositionsAndVelocities(com_constraint_context[context_index], qv)
            com_q = ad_plant.CalcCenterOfMassPositionInWorld(
                com_constraint_context[context_index], [little_dog]
            )
            H_qv = ad_plant.CalcSpatialMomentumInWorldAboutPoint(
                com_constraint_context[context_index], [little_dog], com
            ).rotational() #angular momentum
        else:
            if not np.array_equal(
                qv, plant.GetPositionsAndVelocities(context[context_index])
            ):
                plant.SetPositionsAndVelocities(context[context_index], qv)
            
            com_q = plant.CalcCenterOfMassPositionInWorld(
                context[context_index], [little_dog]
            )
            
            H_qv = plant.CalcSpatialMomentumInWorldAboutPoint(
                context[context_index], [little_dog], com
            ).rotational()
        
        return np.concatenate((com_q - com, H_qv - H))

    for n in range(N):
        prog.AddConstraint(
            partial(com_constraint, context_index=n),
            lb=np.zeros(6),
            ub=np.zeros(6),
            vars=np.concatenate((q[:,n], v[:,n], com[:,n], H[:,n]))
        )
    
    #Kinematic Constraints
    # Basically we have control over joint space
    # However, want to have control over the end effector space
    # This is regarding Foot position and orientation
    def fixed_position_constraint(vars, context_index, frame):
        q, qn = np.split(vars, [nq])
        if not np.array_equal(q, plant.GetPositions(context[context_index])):
            plant.SetPositions(context[context_index], q)
        if not np.array_equal(qn, plant.GetPositions(context[context_index+1])):
            plant.SetPositions(context[context_index+1], qn)
        p_WF = plant.CalcPointsPositions(
            context[context_index],
            frame,
            [0, 0, 0],
            plant.world_frame()
        )
        pn_WF = plant.CalcPointsPositions(
            context[context_index+1],
            frame,
            [0, 0, 0],
            plant.world_frame()
        )
        
        if isinstance(vars[0], AutoDiffXd):
            J_WF = plant.CalcJacobianTranslationalVelocity(
                context[context_index],
                JacobianWrtVariable.kQDot,
                frame,
                [0, 0, 0],
                plant.world_frame(),
                plant.world_frame()
            )
            
            Jn_WF = plant.CalcJacobianTranslationalVelocity(
                context[context_index+1],
                JacobianWrtVariable.kQDot,
                frame,
                [0, 0, 0],
                plant.world_frame(),
                plant.world_frame()
            )
            
            return InitializeAutoDiff(pn_WF - p_WF, Jn_WF @ ExtractGradient(qn) - J_WF @ ExtractGradient(q))
        else:
            return pn_WF - p_WF

    # just add in constraints where if it's in stance, the foot should not move
    # if it's not in stance, the foot should move but be above the ground by a set 
    # clearance.
    for i in range(4):
        for n in range(N):
            # in stance constraint
            if in_stance[i,n]:
                prog.AddConstraint(
                    PositionConstraint(
                        plant,
                        plant.world_frame(),
                        [-np.inf, -np.inf, 0],
                        [np.inf, np.inf, 0],
                        foot_frame[i],
                        [0, 0, 0],
                        context[n]
                    ),
                    q[:,n]
                )
                if n > 0 and in_stance[i,n-1]:
                    # feet should not move when in stance
                    prog.AddConstraint(
                        partial(
                            fixed_position_constraint,
                            context_index=n-1,
                            frame=foot_frame[i]
                        ),
                        lb=np.zeros(3),
                        ub=np.zeros(3),
                        vars=np.concatenate((q[:,n-1], q[:,n]))
                    )
            else:
                min_clearance = 0.01
                prog.AddConstraint(
                    PositionConstraint(
                        plant,
                        plant.world_frame(),
                        [-np.inf, -np.inf, min_clearance],
                        [np.inf, np.inf, np.inf],
                        foot_frame[i],
                        [0, 0, 0],
                        context[n]
                    ),
                    q[:,n]
                )
    
    #Periodicity Constraint
    if is_laterally_symmetric:
        # Joints
        def AddAntiSymmetricPair(a,b): #skew sym pair
            prog.AddLinearEqualityConstraint(a[0] == -b[-1])
            prog.AddLinearEqualityConstraint(a[-1] == -b[0])
        
        def AddSymmetricPair(a,b): # sym pair
            prog.AddLinearEqualityConstraint(a[0] == b[-1])
            prog.AddLinearEqualityConstraint(a[-1] == b[0])
            
        AddAntiSymmetricPair(q_view.front_left_hip_roll, q_view.front_right_hip_roll) # left and right roll should be opposite
        AddSymmetricPair(q_view.front_left_hip_pitch, q_view.front_right_hip_pitch) # left and right pitch should be same
        
        AddSymmetricPair(q_view.front_left_knee, q_view.front_right_knee) # left and right knee should be same
        
        AddAntiSymmetricPair(q_view.back_left_hip_roll, q_view.back_right_hip_roll) # left and right roll should be opposite
        AddSymmetricPair(q_view.back_left_hip_pitch, q_view.back_right_hip_pitch) # left and right pitch should be same
        
        AddSymmetricPair(q_view.back_left_knee, q_view.back_right_knee) # left and right knee should be same
        
        prog.AddLinearEqualityConstraint(q_view.body_y[0] == -q_view.body_y[-1])
        prog.AddLinearEqualityConstraint(q_view.body_z[0] == q_view.body_z[-1])
        
        #body orientation must be in xz plane
        prog.AddBoundingBoxConstraint(0,0, q_view.body_qx[[0,-1]])
        prog.AddBoundingBoxConstraint(0,0, q_view.body_qz[[0,-1]])
        
        #floating base vel
        prog.AddLinearEqualityConstraint(v_view.body_vx[0] == v_view.body_vx[-1])
        prog.AddLinearEqualityConstraint(v_view.body_vy[0] == -v_view.body_vy[-1])
        prog.AddLinearEqualityConstraint(v_view.body_vz[0] == v_view.body_vz[-1])
        
        # CoM velocity
        prog.AddLinearEqualityConstraint(comdot[0,0] == comdot[0,-1])
        prog.AddLinearEqualityConstraint(comdot[1,0] == -comdot[1,-1])
        prog.AddLinearEqualityConstraint(comdot[2,0] == comdot[2,-1])
        
    else:
        q_selector = PositionView([True]* nq)
        q_selector.body_x = False #everything but body_x is periodic
        
        prog.AddLinearConstraint(eq(q[q_selector, 0], q[q_selector, -1]))
        prog.AddLinearConstraint(eq(v[:, 0], v[:, -1]))
    
    snopt = SnoptSolver().solver_id()
    prog.SetSolverOption(snopt, "Iterations Limit", 1e5)
    prog.SetSolverOption(snopt, "Major Iterations Limit", 200)
    prog.SetSolverOption(snopt, "Major Feasibility Tolerance", 5e-6)
    prog.SetSolverOption(snopt, "Major Optimality Tolerance", 1e-4)
    prog.SetSolverOption(snopt, "Superbasics limit", 2000)
    prog.SetSolverOption(snopt, "Linesearch tolerance", 0.9)
    
    #solve using snopt
    
    result = Solve(prog)
    print(result.get_solver_id().name())
    print(f"Time it took: {time.time() - start_timer}")
    input()
    
    def HalfStrideToFullStride(a):
        b = PositionView(np.copy(a))

        b.body_y = -a.body_y
        # Mirror quaternion so that roll=-roll, yaw=-yaw
        b.body_qx = -a.body_qx
        b.body_qz = -a.body_qz

        b.front_left_hip_roll = -a.front_right_hip_roll
        b.front_right_hip_roll = -a.front_left_hip_roll
        b.back_left_hip_roll = -a.back_right_hip_roll
        b.back_right_hip_roll = -a.back_left_hip_roll

        b.front_left_hip_pitch = a.front_right_hip_pitch
        b.front_right_hip_pitch = a.front_left_hip_pitch
        b.back_left_hip_pitch = a.back_right_hip_pitch
        b.back_right_hip_pitch = a.back_left_hip_pitch

        b.front_left_knee = a.front_right_knee
        b.front_right_knee = a.front_left_knee
        b.back_left_knee = a.back_right_knee
        b.back_right_knee = a.back_left_knee

        return b
    
    #Animate trajectory
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)
    t_sol = np.cumsum(np.concatenate(([0], result.GetSolution(h))))
    q_sol = PiecewisePolynomial.FirstOrderHold(t_sol, result.GetSolution(q))
    visualizer.StartRecording()
    num_strides = 4
    t0 = t_sol[0]
    tf = t_sol[-1]
    T = tf * num_strides * (2.0 if is_laterally_symmetric else 1.0)
    for t in np.hstack((np.arange(t0, T, 1.0/32.0), T)):
        context.SetTime(t)
        stride = (t - t0) // (tf - t0)
        ts = (t - t0) % (tf - t0)
        qt = PositionView(q_sol.value(ts))
        if is_laterally_symmetric:
            if stride % 2 == 1:
                qt = HalfStrideToFullStride(qt)
                qt.body_x += stride_length / 2.0
            stride = stride // 2
        qt.body_x += stride * stride_length
        plant.SetPositions(plant_context, qt[:])
        diagram.ForcedPublish(context)
    visualizer.StopRecording()
    visualizer.PublishRecording()

if __name__ == "__main__":
    meshcat = StartMeshcat()
    gait_optimization(meshcat, "walking_trot")
        
    