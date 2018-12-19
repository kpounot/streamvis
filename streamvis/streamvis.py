import logging
import os
import subprocess
import sys

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """
    Execute the "streamvis" command line program.

    This is a wrapper around 'bokeh serve' command which provides a user interface to launch
    applications bundled with the streamvis package.

    For more information, see:
    https://bokeh.pydata.org/en/latest/docs/reference/command/subcommands/serve.html
    """

    apps_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'apps')

    # TODO: generalize streamvis parsing after python/3.7 release
    # due to an issue with 'argparse' (https://bugs.python.org/issue14191),
    # which is supposed to be fixed in python/3.7, keep parsing unflexible, but very simple
    _, app_name, *app_args = sys.argv

    command = ['bokeh', 'serve', os.path.join(apps_path, app_name), *app_args]
    logger.info(' '.join(command))
    subprocess.run(command)

if __name__ == "__main__":
    main()
