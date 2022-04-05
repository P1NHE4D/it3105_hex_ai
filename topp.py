from yaml import YAMLError, safe_load

from game.hex import Hex
from game.nim import Nim
from rl.topp import anet_tournament
from copy import deepcopy
from glob import glob


def main():
    config = dict()
    with open("topp_config.yaml", "r") as stream:
        try:
            config = safe_load(stream)
        except YAMLError as exc:
            print(exc)

    game_config = config["game"]
    if game_config["name"] == "hex":
        game = Hex(**game_config["params"])
    elif game_config["name"] == "nim":
        game = Nim(**game_config["params"])
    else:
        raise ValueError(f'unknown game {game_config["name"]}')
    agent_config = config.get("agent", {})

    topp_config = config["topp"]
    topp_num_sample_games = topp_config["num_games_per_series"]
    topp_include_uniform = topp_config["include_uniform"]
    path = topp_config["weight_path"]
    weight_files = glob(path.strip("/") + "/*.index")
    configs = []
    for weight_file in weight_files:
        cpy = deepcopy(agent_config)
        cpy['anet']['weight_file'] = weight_file
        configs.append(cpy)
    anet_tournament(game, configs, topp_num_sample_games, topp_include_uniform)


if __name__ == '__main__':
    main()
