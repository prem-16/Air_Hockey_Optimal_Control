import threading
import time

import numpy as np
import matplotlib as mpl
from scipy.interpolate import CubicSpline
from air_hockey_challenge.utils.kinematics import inverse_kinematics, jacobian, forward_kinematics
from air_hockey_challenge.framework.agent_base import AgentBase
from baseline.baseline_agent import BezierPlanner, TrajectoryOptimizer, PuckTracker
from casadi import SX, sin, Function, inf, vertcat, nlpsol, qpsol, sumsqr
import copy


def build_agent(env_info, **kwargs):
    """
    Function where an Agent that controls the environments should be returned.
    The Agent should inherit from the mushroom_rl Agent base env.

    Args:
        env_info (dict): The environment information
        kwargs (any): Additionally setting from agent_config.yml
    Returns:
         (AgentBase) An instance of the Agent
    """
    return DefendingAgent(env_info, **kwargs)


class DefendingAgent(AgentBase):
    def __init__(self, env_info, agent_id=1, **kwargs):
        super(DefendingAgent, self).__init__(env_info, agent_id, **kwargs)
        self.last_cmd = None
        self.joint_trajectory = None
        self.restart = True
        self.optimization_failed = False
        self.tactic_finished = False
        self.dt = 1 / self.env_info['robot']['control_frequency']
        self.ee_height = self.env_info['robot']["ee_desired_height"]

        self.bound_points = np.array([[-(self.env_info['table']['length'] / 2 - 0.05),
                                       -(self.env_info['table']['width'] / 2 - 0.05)],
                                      [-(self.env_info['table']['length'] / 2 - 0.05),
                                       (self.env_info['table']['width'] / 2 - 0.05)],
                                      [-0.3, (self.env_info['table']['width'] / 2 - 0.05)],
                                      [-0.3, -(self.env_info['table']['width'] / 2 - 0.05)]])
        self.bound_points = self.bound_points + np.tile([1.51, 0.], (4, 1))
        self.boundary_idx = np.array([[0, 1], [1, 2], [0, 3]])

        table_bounds = np.array([[self.bound_points[0], self.bound_points[1]],
                                 [self.bound_points[1], self.bound_points[2]],
                                 [self.bound_points[2], self.bound_points[3]],
                                 [self.bound_points[3], self.bound_points[0]]])
        self.bezier_planner = BezierPlanner(table_bounds, self.dt)
        self.optimizer = TrajectoryOptimizer(self.env_info)
        if self.env_info['robot']['n_joints'] == 3:
            self.joint_anchor_pos = np.array([-0.9273, 0.9273, np.pi / 2])
        else:
            self.joint_anchor_pos = np.array([6.28479822e-11, 7.13520517e-01, -2.96302903e-11, -5.02477487e-01,
                                              -7.67250279e-11, 1.92566224e+00, -2.34645597e-11])

        self.puck_tracker = PuckTracker(env_info, agent_id=agent_id)
        self._obs = None

        self.agent_params = {
            'hit_range': [0.8, 1.3],
            'max_plan_steps': 10,
        }

    def reset(self):
        self.last_cmd = None
        self.joint_trajectory = []
        self.restart = True
        self.tactic_finished = False
        self.optimization_failed = False

        self._obs = None
        self.puck_pos_list = []
        self.plan_thread = threading.Thread(target=self._plan_trajectory_thread)
        # self.plan_thread.start()

    def draw_action(self, obs):
        if self.restart:
            self.restart = False
            self.puck_tracker.reset(self.get_puck_pos(obs))
            self.last_cmd = np.vstack([self.get_joint_pos(obs), self.get_joint_vel(obs)])
            self.joint_trajectory = np.array([self.last_cmd])

        self.puck_tracker.step(self.get_puck_pos(obs))
        self._obs = obs.copy()
        opt_trial = 0
        t_predict = 1.0
        defend_line = 0.8
        state, P, t_predict = self.puck_tracker.get_prediction(t_predict, defend_line)
        puck_pos = state[[0, 1, 4]]
        self.puck_pos_list.append(puck_pos)
        des_pos = puck_pos

        q_0 = obs[6:13]
        jac = jacobian(self.robot_model, self.robot_data, q_0)[:3, :7]
        x0 = list(forward_kinematics(self.robot_model, self.robot_data, q_0)[0])
        q, _ = self.solve_casadi(x0, des_pos, jac)
        if not _:
            opt_trial += 1
            self.optimization_failed = True
            action = self.last_cmd[:7]
            print("fail")
        else:
            next_q = q_0 + q * self.env_info['dt']
            action = next_q
            self.last_cmd = action

        return action

    def calculate_positions(self, joint_velocities, initial_position, time_step):
        positions = [initial_position]

        for velocity in joint_velocities:
            current_position = np.array(positions[-1])
            new_position = current_position + np.array(velocity) * time_step
            positions.append(new_position.tolist())

        return positions

    def _plan_trajectory_thread(self):
        while not self.tactic_finished:
            time.sleep(0.01)
            opt_trial = 0
            t_predict = 1.0
            defend_line = 0.8
            state, P, t_predict = self.puck_tracker.get_prediction(t_predict, defend_line)

            if len(self.joint_trajectory) < self.agent_params['max_plan_steps']:
                if np.linalg.det(P[:2, :2]) < 1e-3:
                    joint_pos = self.get_joint_pos(self._obs)
                    joint_vel = self.get_joint_vel(self._obs)
                    puck_pos = state[[0, 1, 4]]
                    jac = jacobian(self.robot_model, self.robot_data, joint_vel)[:3, :7]
                    ee_pos, _ = self.get_ee_pose(self._obs)

                    q, _ = self.solve_casadi(ee_pos.tolist(), puck_pos.tolist(), jac)
                    _ = self.calculate_positions(q, joint_pos, self.env_info['dt'])

                    joint_pos_traj = joint_pos + q * self.env_info['dt']
                    ee_traj, switch_idx = self.plan_ee_trajectory(puck_pos, ee_pos, t_predict)
                    _, joint_pos_traj = self.optimizer.optimize_trajectory(ee_traj, joint_pos, joint_vel, None)

                    if len(joint_pos_traj) > 0:
                        if len(self.joint_trajectory) > 0:
                            self.joint_trajectory = np.vstack([self.joint_trajectory,
                                                               self.cubic_spline_interpolation(joint_pos_traj)])
                        else:
                            self.joint_trajectory = self.cubic_spline_interpolation(joint_pos_traj)
                        self.tactic_finished = True
                    else:
                        opt_trial += 1
                        self.optimization_failed = True
                        self.joint_trajectory = np.array([])

            if opt_trial >= 5:
                self.tactic_finished = True
                break

    def integrate_RK4(self, s_expr, a_expr, sdot_expr, dt, N_steps=25):
        '''RK4 integrator.

        s_expr, a_expr: casadi expression that have been used to define the dynamics sdot_expr
        sdot_expr:      casadi expr defining the rhs of the ode
        dt:             integration interval
        N_steps:        number of integration steps per integration interval, default:1
        '''
        dt = self.env_info['dt']
        h = dt / N_steps

        s_end = s_expr

        sdot_fun = Function('xdot', [s_expr, a_expr], [sdot_expr])

        for _ in range(N_steps):
            # FILL IN YOUR CODE HERE
            v_1 = sdot_fun(s_end, a_expr)
            v_2 = sdot_fun(s_end + 0.5 * h * v_1, a_expr)
            v_3 = sdot_fun(s_end + 0.5 * h * v_2, a_expr)
            v_4 = sdot_fun(s_end + v_3 * h, a_expr)
            s_end += (1 / 6) * (v_1 + 2 * v_2 + 2 * v_3 + v_4) * h

        F_expr = s_end

        return F_expr

    def solve_casadi(self, x0_bar, x_des, jac):
        # continuous model dynamics
        n_s = 3  # number of states
        n_a = 7  # number of actions

        x = SX.sym('x')
        y = SX.sym('y')
        z = SX.sym('z')

        omega = SX.sym('omega', 7)

        s = vertcat(x, y, z)
        # q_0 = policy.robot_data.qpos.copy()
        # jac = jacobian(policy.robot_model, policy.robot_data,q_0)[:3, :7]
        s_dot = vertcat(jac @ omega)
        # Define number of steps in the control horizon and discretization step
        # print(s_dot)
        N = 5
        delta_t = 1 / 50
        # Define RK4 integrator function and initial state x0_bar
        F_rk4 = Function("F_rk4", [s, omega], [self.integrate_RK4(s, omega, s_dot, delta_t)])

        # Start with an empty NLP
        w = []
        w0 = []
        lbw = []
        ubw = []
        J = 0
        g = []
        lbg = []
        ubg = []

        # "Lift" initial conditions
        Xk = SX.sym('X0', 3)
        w += [Xk]
        lbw += x0_bar  # set initial state
        ubw += x0_bar  # set initial state
        w0 += x0_bar  # set initial state

        # Formulate the NLP
        for k in range(N):
            # New NLP variable for the control
            Uk = SX.sym('U_' + str(k), 7)
            w += [Uk]
            lbw += [-1.48352986, -1.48352986, -1.74532925, -1.30899694, -2.26892803,
                    -2.35619449, -2.35619449]
            ubw += [1.48352986, 1.48352986, 1.74532925, 1.30899694, 2.26892803,
                    2.35619449, 2.35619449]
            w0 += [0, 0, 0, 0, 0, 0, 0]

            # Integrate till the end of the interval
            Xk_end = F_rk4(Xk, Uk)
            # J = J + delta_t *(sumsqr((Xk-x_des).T @ Q )+ sumsqr(R@Uk)) # Complete with the stage cost
            J = J + (sumsqr((Xk - x_des)))  # Complete with the stage cost

            # New NLP variable for state at end of interval
            Xk = SX.sym(f'X_{k + 1}', 3)
            w += [Xk]
            lbw += [.5, -.5, 0.165]
            ubw += [1.5, .5, 0.165]
            w0 += [0, 0, 0]

            # Add equality constraint to "close the gap" for multiple shooting
            g += [Xk_end - Xk]
            lbg += [0, 0, 0]
            ubg += [0, 0, 0]
        J = J + sumsqr((Xk - x_des))  # Complete with the terminal cost (NOTE it should be weighted by delta_t)

        # Create an NLP solver
        prob = {'f': J, 'x': vertcat(*w), 'g': vertcat(*g)}
        solver = nlpsol('solver', 'ipopt', prob, {'ipopt': {'print_level': 0}, 'print_time': False})

        # Solve the NLP
        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        w_opt = sol['x'].full().flatten()

        # return np.array([w_opt[3::10],w_opt[4::10],w_opt[5::10],w_opt[6::10],w_opt[7::10],w_opt[8::10],w_opt[9::10]])

        return np.array(w_opt[3:10]), solver.stats()['success']

    def plan_ee_trajectory(self, puck_pos, ee_pos, t_plan):
        hit_dir_2d = np.array([0., np.sign(puck_pos[1])])

        hit_pos_2d = puck_pos[:2] - hit_dir_2d * (self.env_info['mallet']['radius'])

        start_pos_2d = ee_pos[:2]

        hit_vel = 0
        self.bezier_planner.compute_control_point(start_pos_2d, np.zeros(2), hit_pos_2d, hit_dir_2d * hit_vel, t_plan)

        res = np.array([self.bezier_planner.get_point(t_i) for t_i in np.arange(0, self.bezier_planner.t_final + 1e-6,
                                                                                self.dt)])
        p = res[1:, 0].squeeze()
        dp = res[1:, 1].squeeze()
        ddp = res[1:, 2].squeeze()

        p = np.hstack([p, np.ones((p.shape[0], 1)) * self.ee_height])
        dp = np.hstack([dp, np.zeros((p.shape[0], 1))])
        ddp = np.hstack([ddp, np.zeros((p.shape[0], 1))])

        hit_traj = np.hstack([p, dp, ddp])

        last_point_2d = hit_traj[-1, :2]
        last_vel_2d = hit_traj[-1, 3:5]

        # Plan Return Trajectory
        stop_point = np.array([0.65, 0.])
        self.bezier_planner.compute_control_point(last_point_2d, last_vel_2d, stop_point, np.zeros(2), 1.5)

        res = np.array([self.bezier_planner.get_point(t_i) for t_i in np.arange(0, self.bezier_planner.t_final + 1e-6,
                                                                                self.dt)])
        p = res[1:, 0].squeeze()
        dp = res[1:, 1].squeeze()
        ddp = res[1:, 2].squeeze()

        p = np.hstack([p, np.ones((p.shape[0], 1)) * self.ee_height])
        dp = np.hstack([dp, np.zeros((p.shape[0], 1))])
        ddp = np.hstack([ddp, np.zeros((p.shape[0], 1))])
        return_traj = np.hstack([p, dp, ddp])

        ee_traj = np.vstack([hit_traj, return_traj])
        return ee_traj, len(hit_traj)

    def cubic_spline_interpolation(self, joint_pos_traj):
        joint_pos_traj = np.array(joint_pos_traj)
        t = np.linspace(1, joint_pos_traj.shape[0], joint_pos_traj.shape[0]) * 0.02

        f = CubicSpline(t, joint_pos_traj, axis=0)
        df = f.derivative(1)
        return np.stack([f(t), df(t)]).swapaxes(0, 1)


def main():
    from air_hockey_challenge.framework.air_hockey_challenge_wrapper import AirHockeyChallengeWrapper
    import matplotlib.pyplot as plt
    import matplotlib
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12

    plt.rc('font', size=BIGGER_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)
    matplotlib.use("tkAgg")

    env = AirHockeyChallengeWrapper(env="7dof-defend", interpolation_order=1, debug=True)

    agent = DefendingAgent(env.base_env.env_info)

    obs = env.reset()
    agent.reset()

    steps = 0
    puck_positions = []
    ee_positions = []
    all_dq = []
    all_ee_pos = []
    while True:
        steps += 1
        action = agent.draw_action(obs)
        obs, reward, done, info = env.step(action)
        puck_pos, puck_dir = env.base_env.get_puck(obs)
        # ee_pos, _ = env.base_env.get_ee_pose(obs)
        # print(puck_pos)
        puck_positions.append(puck_pos)
        # ee_positions.append(ee_pos)
        env.render()
        env_info = env.base_env.env_info
        dq = obs[env_info['joint_vel_ids']]
        q = obs[env_info['joint_pos_ids']]
        ee_pos = list(forward_kinematics(agent.robot_model, agent.robot_data, q)[0])
        all_ee_pos.append(ee_pos)
        all_dq.append(dq)
        print(agent.puck_pos_list)
        if done or steps > env.info.horizon / 2:
            nq = env.base_env.env_info['robot']['n_joints']
            if env.base_env.debug:
                x_pos = [puck_positions[i][0] for i in range(len(puck_positions))]
                y_pos = [puck_positions[i][1] for i in range(len(puck_positions))]
                x_pred = [agent.puck_pos_list[i][0] for i in range(len(agent.puck_pos_list))]
                y_pred = [agent.puck_pos_list[i][0] for i in range(len(agent.puck_pos_list))]
                # z_pos = [puck_positions[i][2] for i in range(len(puck_positions))]
                ee_pos_x = [all_ee_pos[i][0] for i in range(len(all_ee_pos))]
                ee_pos_y = [all_ee_pos[i][1] for i in range(len(all_ee_pos))]
                # ee_pos_z = [ee_positions[i][2] for i in range(len(ee_positions))]
                # ax = plt.axes(projection="3d")
                plt.scatter(x_pos[0], y_pos[0], marker='X', c='g', zorder=3,
                            label='Start position of puck', s=50)
                plt.scatter(x_pos[-1], y_pos[-1], marker='X', c='r', zorder=3, label='End position of puck', s=50)
                plt.scatter(ee_pos_x[0], ee_pos_y[0], marker='X', zorder=3, c='#653700',
                            label='Start position of robot end-effector', s=50)
                plt.scatter(ee_pos_x[-1], ee_pos_y[-1], marker='X', zorder=3, c='#ED0DD9',
                            label='End position of robot end-effector', s=50)
                # plt.arrow(0, x_pos[0], 1, x_pos[1] - x_pos[0], shape='full', lw=0, length_includes_head=True, head_width=.05)
                plt.xlabel("X position in the air hockey table ")
                plt.ylabel("Y position in the air hockey table ")
                lgd = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                                 fancybox=True, shadow=True, ncol=2)
                plt.plot(x_pos, y_pos, linestyle='dashed', linewidth=3.0)

                # plt.plot(x_pred, y_pred, linestyle= ':')
                plt.plot(ee_pos_x, ee_pos_y, linestyle='solid', linewidth=3.0)
                # plt.xlim(0.5, 1.5)
                # plt.ylim(-0.5, 0.5)

                plt.savefig("trajectory.jpeg", dpi=300, bbox_extra_artists=(lgd,), bbox_inches='tight')
                puck_positions = []
                all_ee_pos = []
                agent.puck_pos_list = []
                # plt.plot(ee_pos_x,ee_pos_y, 'bo')

                # puck_postions = []
                trajectory_record = np.array(env.base_env.controller_record)
                fig, axes = plt.subplots(2, 4)
                nq_total = nq * env.base_env.n_agents
                # fig.set_figheight(12)
                # fig.set_figwidth(12)
                env_info = env.base_env.env_info
                constraints = env_info['constraints'].get('joint_pos_constr')

                # print(constraints.fun(q, dq))
                lower_vel_limit = [-1.48352986, -1.48352986, -1.74532925, -1.30899694, -2.26892803,
                                   -2.35619449, -2.35619449]
                upper_vel_limit = [1.48352986, 1.48352986, 1.74532925, 1.30899694, 2.26892803,
                                   2.35619449, 2.35619449]
                fig.text(0.5, 0.01, 'Time (sec)', ha='center', va='center')
                fig.text(0.001, 0.5, 'Joint Velocity(rad/s)', ha='center', va='center', rotation='vertical')
                fig.tight_layout()
                fig.delaxes(axes[1][3])
                time_ = [i * 0.02 for i in range(len(all_dq))]
                for j in range(nq):
                    # axes[0, j].plot(trajectory_record[:, j])
                    # axes[0, j].plot(trajectory_record[:, j + nq_total]) # current joint position
                    # axes[1, j].plot(trajectory_record[:, j + 2 * nq_total])
                    # axes[j].axis('equal')
                    # axes[j].set_aspect('equal')

                    q_ = [all_dq[i][j] for i in range(len(all_dq))]
                    index_1 = j // 4
                    index_2 = j % 4
                    axes[index_1][index_2].axhline(upper_vel_limit[j], color='r', linestyle='-')
                    axes[index_1][index_2].axhline(lower_vel_limit[j], color='r', linestyle='-')
                    axes[index_1][index_2].plot(time_, q_)  # current joint velocity
                plt.savefig("joint_velocity_constraint.jpeg", dpi=300, bbox_inches='tight')
                plt.show()

            all_dq = []

            steps = 0
            obs = env.reset()
            agent.reset()


if __name__ == '__main__':
    main()
