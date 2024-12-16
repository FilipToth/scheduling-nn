from ray import tune
from ray import train
from ppo.ppo import PPOTrainer, PPOParams

def eval_callback(eval_reward, cumulative_reward, eval_step_count, eval_slowdown):
    pass


def step_callback(num_el, reward, step_count, lr):
    pass


def train_ppo(config):
    lr = config["lr"]
    gamma = config["gamma"]
    lmbda = config["lmbda"]
    clip_eps = config["clip_eps"]
    entropy_eps = config["entropy_eps"]
    max_grad_norm = config["max_grad_norm"]

    params = PPOParams(
        lr=lr,
        gamma=gamma,
        lmbda=lmbda,
        clip_eps=clip_eps,
        entropy_eps=entropy_eps,
        max_grad_norm=max_grad_norm
    )

    trainer = PPOTrainer(params, eval_callback, step_callback)

    num_steps = config["training_steps"]
    for _ in range(num_steps):
        reward = trainer.train_step()
        if reward == None:
            break

        train.report({ "reward": reward })


search_space = {
    "lr": tune.loguniform(1e-6, 1e-3),
    "gamma": tune.uniform(0.9, 0.999),
    "lmbda": tune.uniform(0.9, 1.0),
    "clip_eps": tune.uniform(0.1, 0.3),
    "entropy_eps": tune.loguniform(1e-5, 1e-3),
    "max_grad_norm": tune.uniform(0.5, 2.0),
    "training_steps": tune.randint(100, 1000),
}

config = tune.TuneConfig(
    num_samples=50,
    metric="reward",
    mode="max"
)

tuner = tune.Tuner(
    train_ppo,
    param_space=search_space,
    tune_config=config
)

results = tuner.fit()
results_df = results.get_dataframe()
results_df.to_csv("../out/raytune.csv")

best = results.get_best_result(metric="reward", mode="max")
print(f"best: {best}")
