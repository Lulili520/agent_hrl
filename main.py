import numpy as np
from alfworld.agents.environment import get_environment
import alfworld.agents.modules.generic as generic

from models.HLP import HLP_net
from models.LLP import LLP_net

from prompt import High_prompt_template, Low_prompt_template


# while True:
#     # get random actions from admissible 'valid' commands (not available for AlfredThorEnv)
#     admissible_commands = list(info['admissible_commands']) # note: BUTLER generates commands word-by-word without using admissible_commands
#     random_actions = [np.random.choice(admissible_commands[0])]

#     # step
#     obs, scores, dones, infos = env.step(random_actions)
#     print("Action: {}, Obs: {}".format(random_actions[0], obs[0]))


def main():
    # load config
    config = generic.load_config()
    env_type = config['env']['type'] # 'AlfredTWEnv' or 'AlfredThorEnv' or 'AlfredHybrid'

    # setup environment
    env = get_environment(env_type)(config, train_eval='train')
    env = env.init_env(batch_size=1)

    # interact
    obs, info = env.reset()
    admissible_commands = list(info['admissible_commands'])

    for turn in range(5):
        step = 0
        history = High_prompt_template.format(observation=obs, option1="locate", option2="", option3=, option4=)
        while step <= 200:
            

    print(admissible_commands)

if __name__ == "__main__":
    main()
