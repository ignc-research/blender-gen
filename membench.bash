script="blender --background --python main.py"
name="$2"

if [ -z "$name" ]
	then
	name="$(git log -1 --pretty=%B)"
fi

echo "benching '${script}' for ${1} times"

echo -n ""

avg_memory="0"
avg_time="0"


for i in $(seq 1 $1);
do
	echo -ne "\r$i/$1 "
	rm -r ./data
	fulldata="$(/usr/bin/time -f "\mytime-%e\nmymem-%M\n" ${script} 2>&1)"

	now_time="$(echo "${fulldata}" | grep 'mytime' | sed -e s/.*-//)"
	now_memory="$(echo "${fulldata}" | grep 'mymem' | sed -e s/.*-//)"

	avg_memory="$(echo "$avg_memory + $now_memory" | bc)"
	avg_time="$(echo "$avg_time + $now_time" | bc)"

	echo -n "Last: $(echo "$now_memory / 1000" | bc)MB / ${now_time}s"
done

avg_memory="$(echo "100 * $avg_memory / $1 * .01" | bc)"
avg_time="$(echo "100 * $avg_time / $1 * .01" | bc)"

echo ""

echo "mean peak memory usage: $(echo "$avg_memory / 1000" | bc)MB"
echo "mean execution time: ${avg_time}s"

echo "$name,$1,$avg_memory,$avg_time" >> bench.csv
