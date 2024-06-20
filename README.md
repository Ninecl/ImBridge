# README
Source code of "Inductive Message Passing for Knowledge Graph Completion over Relational Hypergraph"

# Run Example
Take the dataset FB-U30 as example, to train the model, run:
```python
python train.py -d FB-U30 -e ImBridge
```

To evaluate the model:
```python
python test_rank.py -d FB-U30 -e ImBridge
```

To evaluate the model on bridging links with seen relations:
```python
python test_rank.py -d FB-U30  -e ImBridge -t bri_seenR
```

To evaluate the model on bridging links with unseen relations:
```python
python test_rank.py -d FB-U30  -e ImBridge -t bri_unseenR
```

To evaluate the model on enclosing links with seen relations:
```python
python test_rank.py -d FB-U30  -e ImBridge -t enc_seenR
```

To evaluate the model on enclosing links with unseen relations:
```python
python test_rank.py -d FB-U30  -e ImBridge -t enc_unseenR
```

