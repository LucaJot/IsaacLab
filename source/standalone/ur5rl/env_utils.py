from omni.isaac.lab.utils.io.pkl import dump_pickle
from omni.isaac.lab.utils.io.yaml import dump_yaml
from rsl_rl.runners import OnPolicyRunner
from agents.rsl_rl_ppo_cfg import (
    Ur5RLPPORunnerCfg,
)
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
import os
from datetime import datetime
import gymnasium as gym
import torch


def set_learning_config():
    # Get learning configuration
    agent_cfg: RslRlOnPolicyRunnerCfg = Ur5RLPPORunnerCfg()

    # specify directory for logging experiments --------------------------
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(
        log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint
    )
    log_dir = os.path.dirname(resume_path)
    # --------------------------------------------------------------------
    return agent_cfg, log_dir, resume_path


def load_most_recent_model(
    env: gym.Env, log_dir, resume_path, agent_cfg: RslRlOnPolicyRunnerCfg
):
    """Load the most recent model from the log directory."""
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(
        env,  # type: ignore
        agent_cfg.to_dict(),  # type: ignore
        log_dir=None,
        device=agent_cfg.device,
    )
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device="cuda:0")

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(
        ppo_runner.alg.actor_critic,
        ppo_runner.obs_normalizer,
        path=export_model_dir,
        filename="policy.pt",
    )
    export_policy_as_onnx(
        ppo_runner.alg.actor_critic,
        normalizer=ppo_runner.obs_normalizer,
        path=export_model_dir,
        filename="policy.onnx",
    )
    return policy


def train_rsl_rl_agent(
    env,
    env_cfg,
    agent_cfg,
    resume=True,
):
    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    if resume:
        # save resume path before creating a new log_dir
        resume_path = get_checkpoint_path(
            log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint
        )

    # create runner from rsl-rl
    runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device
    )
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint

    if resume:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training
    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True
    )


def train_rsl_rl_agent_init(env, env_cfg, agent_cfg, CL: int, resume=True):
    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_CL_iteration{CL}"
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    if resume:
        # save resume path before creating a new log_dir
        resume_path = get_checkpoint_path(
            log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint
        )

    # create runner from rsl-rl
    runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device
    )
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint

    if resume:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    print(f"Starting CL iteration {CL}")
    env.unwrapped.set_CL_state(CL)  # type: ignore
    # run training
    log_results = runner.learn(
        num_learning_iterations=agent_cfg.max_iterations,
        init_at_random_ep_len=True,
    )
    return log_results


def run_task_in_sim(
    env: RslRlVecEnvWrapper,
    log_dir: str,
    resume_path: str,
    agent_cfg: RslRlOnPolicyRunnerCfg,
    simulation_app,
):
    """Play with RSL-RL agent."""

    policy = load_most_recent_model(
        env=env,
        log_dir=log_dir,
        resume_path=resume_path,
        agent_cfg=agent_cfg,
    )

    # reset environment
    obs, _ = env.reset()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, reward, dones, info = env.step(actions)  # type: ignore
            # print_dict(info)
            if dones[0]:  # type: ignore
                if info["time_outs"][0]:  # type: ignore
                    print("Time Out!")
                    return False, False, obs, policy
                else:
                    print("Interrupt detected!")
                    if info["observations"]["torque_limit_exeeded"][0]:
                        print("Torque Limit Exceeded!")
                    last_obs = obs.clone()
                    torch.save(last_obs, os.path.join(log_dir, "last_obs.pt"))
                    return False, True, obs, policy

            if info["observations"]["grasp_success"][0]:  # type: ignore
                print("Grasp Successful!")
                return True, False, obs, policy
