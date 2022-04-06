from yaml import safe_load, YAMLError
from oht.client import OHTClient


def main():
    config = dict()
    with open("oht_config.yaml", "r") as stream:
        try:
            config = safe_load(stream)
        except YAMLError as exc:
            print(exc)
    client = OHTClient(config=config)
    try:
        league = config["oht"]["league"]
    except KeyError:
        league = False
    client.run(mode="league" if league else "qualifiers")


if __name__ == '__main__':
    main()
