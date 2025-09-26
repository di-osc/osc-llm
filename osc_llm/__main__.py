from jsonargparse import CLI


commands = {}


def run_cli():
    CLI(components=commands)


if __name__ == "__main__":
    run_cli()
