import numpy as np
import tracemalloc
tracemalloc.start()

n = 10
l = []

for i in range(n):
    ss1 = tracemalloc.take_snapshot()

    array_l = np.random.choice(1000, size=(700, 700))
    array_s = array_l[:5, :5]
    l.append(array_s)

    ss2 = tracemalloc.take_snapshot()
    top_stats = ss2.compare_to(ss1, 'lineno')
    stat = top_stats[0]
    print("%s memory blocks: %.1f KiB" % (stat.count, stat.size / 1024))

    for line in stat.traceback.format():
        print(line)
    