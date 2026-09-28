
default:
    @just --list

[working-directory('build/sample')]
leakcheck NUMCALLERS='20':
    valgrind --leak-check=yes --num-callers={{ NUMCALLERS }} ./neat_sample_xor 126 2>&1 | tee lcoutput.txt
    less lcoutput.txt

[working-directory('build/sample')]
profile:
    valgrind --tool=cachegrind ./neat_sample_xor 126
    ls -Atr | tail -n 1 | xargs -I{} cg_annotate {} | less

[working-directory('build/sample')]
perf-stat-xor:
    perf stat --big-num --event=cache-references,cache-misses,cycles,instructions,faults,migrations --repeat=10 ./neat_sample_xor 126

[working-directory('build/sample')]
time-snake:
    time ./neat_sample_snakesdl --seed 126 --generations 100 --headless --population 300

[working-directory('build/sample')]
perf-report:
    perf record -B -e cache-misses,cache-references ./neat_sample_xor 126
    perf report -Mintel 
