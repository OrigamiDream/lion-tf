# Lion
[[arXiv Paper]](https://arxiv.org/abs/2302.06675)

A new optimizer discovered by Google Brain team, which is more memory-efficient than Adam. 

This implementation uses the new Optimizer API released in TensorFlow 2.11+

The new optimizer is way simpler than Adam, it only takes 2 lines of code for dense gradients.
```python
# Dense gradients.
variable.assign_sub(lr * tf.math.sign(m * self.beta_1 + gradient * (1 - self.beta_1)))
m.assign(m * self.beta_2 + gradient * (1 - self.beta_2))
```

I've never tested the implementation yet.

### Usage
```python
import tensorflow as tf
from tensorflow.keras import layers, models

from lion_tf import Lion

# model definition
model = models.Sequential([
    layers.Dense(3, input_shape=(2,)),
    layers.Dense(1)
])

optimizer = Lion()

# feedforward and backpropagation
with tf.GradientTape() as tape:
    loss = model(tf.zeros((1, 2)), training=True)
grads = tape.gradient(loss, model.trainable_variables)
optimizer.apply_gradients(zip(grads, model.trainable_variables))
```
