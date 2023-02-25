import os
import sys
import click
import subprocess



def configure(python_cmd):
	subprocess.Popen([python_cmd, "./src/configure.py", *sys.argv[1:]], stdout=sys.stdout, stderr=sys.stderr).wait()

def render(python_cmd):
	with open(os.devnull, "w") as devnull:
		subprocess.Popen(["blender", "--background", "--python", "./src/render.py"], stdout=devnull, stderr=sys.stderr).wait()

def merge(python_cmd):
	subprocess.Popen([python_cmd, "./src/merge.py", *sys.argv[1:]], stdout=sys.stdout, stderr=sys.stderr).wait()


@click.command(context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
))
@click.option("--target", type=click.Choice(["all","configure","render","merge"]), default="all")

def main(target):

	python_cmd = os.path.realpath(sys.executable)

	if target in ["all", "configure"]:
		configure(python_cmd)

	if target in ["all", "render"]:
		render(python_cmd)

	if target in ["all", "merge"]:
		merge(python_cmd)

if __name__ == "__main__":
	main()