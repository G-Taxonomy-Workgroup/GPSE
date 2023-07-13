## Data format

The `.txt` graph files under Abboud2020 follows the format used in
https://github.com/hadarser/ProvablyPowerfulGraphNetworks/tree/master.
The format readme can be found
[here](https://www.dropbox.com/s/vjd6wy5nemg2gh6/benchmark_graphs.zip?file_subpath=%2Fbenchmark_graphs%2FREADME.md),
which is also summarized below:

```txt
1st line: N number of graphs; then the following N blocks describe the graphs

For each block of text:
A line contains n l, where n is number of nodes in the current graph, and l is
the graph label (0 or 1 in the case of EXP and CEXP graphs from Abboud2020)

Following n lines:
the ith line describes the information of ith node (0 based), which starts with
t m, where t is the tag of current node, and m is the number of neighbors of
current node;
following m numbers indicate the neighbor indices (starting from 0).
following d numbers (if any) indicate the continuous node features (attributes)
```

## Data sources

- Abboud2020: Supplemental materials under https://openreview.net/forum?id=L7Irrt5sMQa
