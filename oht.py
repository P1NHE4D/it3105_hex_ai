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
    client.run()


if __name__ == '__main__':
    main()
