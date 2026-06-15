# regtree

Regression Tree implementation in C++ with Python bindings.


## Installation
```bash
pip install git+https://github.com/piotr-kosakowski/regtree.git
```

## Basic Usage
```python
import numpy as np
from regtree import DecisionTreeRegressor

X = np.random.rand(100, 5)
y = np.random.rand(100)

model = DecisionTreeRegressor(max_depth=5)
model.fit(X, y)
preds = model.predict(X)
```

## Documentation
The full documentation and C++ API reference are available at the [github pages](https://piotr-kosakowski.github.io/regtree/).


