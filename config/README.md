## Configuration File

You can specify the number of nodes and the GPU allocation as follows:

```
{
  ....,
  "port": "1234",
  "n_nodes": 25,
  "node0": {
    "node_id": 0,
    "cuda": "cuda:0"
  },
  "node1": {
    "node_id": 1,
    "cuda": "cuda:1
  }, ...
}
```
