.PHONY: all

all: clean configure render merge

configure:
	python3.10 src/configure.py

render:
	blender  --background --python src/render.py > /dev/null 2> &1
merge:
	python3.10 src/merge.py

clean:
	rm -rf /data/intermediate/
#	rm -rf /data/output/
