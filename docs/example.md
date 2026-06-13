# Usage Example

Below is a basic example of how to use DecisionTreeRegressor from the  `regtree` package.

```python
import numpy as np
from regtree import DecisionTreeRegressor

X = np.random.rand(100, 5)
y = np.ones(100)
model = DecisionTreeRegressor(max_depth=3, min_samples_split=2)
model.fit(X, y)
predictions = model.predict(X)
```
