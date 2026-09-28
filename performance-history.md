ae6e7a8 20260926 18:36 - refactor: seedable runs

perf stat --big-num --event=cache-references,cache-misses,cycles,instructions,faults,migrations --repeat=10 ./neat_sample_xor 126

Performance counter stats for './neat_sample_xor 126' (10 runs):

         4,873,340      cache-references:u                                                      ( +-  0.76% )
           150,896      cache-misses:u                                                          ( +-  8.28% )
       449,371,532      cycles:u                                                                ( +-  0.26% )
       786,661,587      instructions:u                                                          ( +-  0.00% )
               217      faults:u                                                                ( +-  0.17% )
                 0      migrations:u

       0.118719894 +- 0.000335120 seconds time elapsed  ( +-  0.28% )

14bd459 20260926 21:57 - refactor!: non-static injected InnovationHistory

perf stat --big-num --event=cache-references,cache-misses,cycles,instructions,faults,migrations --repeat=10 ./neat_sample_xor 126

Performance counter stats for './neat_sample_xor 126' (10 runs):

         4,957,917      cache-references:u                                                      ( +-  0.90% )
           151,968      cache-misses:u                                                          ( +- 11.06% )
       448,477,726      cycles:u                                                                ( +-  0.21% )
       786,736,559      instructions:u                                                          ( +-  0.00% )
               217      faults:u                                                                ( +-  0.21% )
                 0      migrations:u

       0.118183718 +- 0.000280175 seconds time elapsed  ( +-  0.24% )

4edd4da 20260926 22:16 - refactor!: type changes in Neuron

perf stat --big-num --event=cache-references,cache-misses,cycles,instructions,faults,migrations --repeat=10 ./neat_sample_xor 126

Performance counter stats for './neat_sample_xor 126' (10 runs):

         3,871,524      cache-references:u                                                      ( +-  1.38% )
            88,293      cache-misses:u                                                          ( +- 11.07% )
       440,142,163      cycles:u                                                                ( +-  0.22% )
       786,977,220      instructions:u                                                          ( +-  0.00% )
               182      faults:u                                                                ( +-  0.20% )
                 0      migrations:u

       0.115875108 +- 0.000277010 seconds time elapsed  ( +-  0.24% )

27/09/2026 17:58 - 9e8cc92 refactor: snake test args, optional native build

perf stat --big-num --event=cache-references,cache-misses,cycles,instructions,faults,migrations --repeat=10 ./neat_sample_xor 126

Performance counter stats for './neat_sample_xor 126' (10 runs):

         3,880,905      cache-references:u                                                      ( +-  1.26% )
           117,780      cache-misses:u                                                          ( +- 11.90% )
       440,983,778      cycles:u                                                                ( +-  0.20% )
       786,977,233      instructions:u                                                          ( +-  0.00% )
               183      faults:u                                                                ( +-  0.29% )
                 0      migrations:u

       0.116177510 +- 0.000280519 seconds time elapsed  ( +-  0.24% )

time ./neat_sample_snakesdl 126 100 headless

Exiting normally: true | Time: 11210ms

real    0m11.221s
user    0m11.089s
sys     0m0.080s
