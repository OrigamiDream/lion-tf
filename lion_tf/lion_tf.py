import tensorflow as tf

from tensorflow.keras.optimizers import Optimizer


class Lion(Optimizer):

    def __init__(self,
                 learning_rate=0.0001,
                 beta_1=0.9,
                 beta_2=0.999,
                 weight_decay=None,
                 clipnorm=None,
                 clipvalue=None,
                 global_clipnorm=None,
                 use_ema=False,
                 ema_momentum=0.99,
                 ema_overwrite_frequency=None,
                 jit_compile=True,
                 name='Lion',
                 **kwargs):
        super(Lion, self).__init__(
            name=name,
            weight_decay=weight_decay,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            global_clipnorm=global_clipnorm,
            use_ema=use_ema,
            ema_momentum=ema_momentum,
            ema_overwrite_frequency=ema_overwrite_frequency,
            jit_compile=jit_compile,
            **kwargs
        )
        self._learning_rate = self._build_learning_rate(learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self._built = False
        self._momentums = []

    def build(self, var_list):
        super(Lion, self).build(var_list)
        if hasattr(self, '_built') and self._built:
            return
        self._built = True
        self._momentums = []
        for var in var_list:
            self._momentums.append(self.add_variable_from_reference(
                model_variable=var, variable_name='m'
            ))

    def update_step(self, gradient, variable):
        lr = tf.cast(self.learning_rate, variable.dtype)
        var_key = self._var_key(variable)
        m = self._momentums[self._index_dict[var_key]]

        if isinstance(gradient, tf.IndexedSlices):
            # Sparse gradients.
            m.assign(m * self.beta_1)
            m_scaled_g_values = gradient.values * (1 - self.beta_1)
            m_t = m.scatter_add(
                tf.IndexedSlices(
                    m_scaled_g_values, gradient.indices
                )
            )
            variable.assign_sub(lr * tf.math.sign(m_t))

            m_t = m_t.scatter_add(
                tf.IndexedSlices(
                    -m_scaled_g_values, gradient.indices
                )
            )
            m_t = m_t.assign(m_t * self.beta_2 / self.beta_1)
            m_scaled_g_values = gradient.values * (1 - self.beta_2)
            m_t.scatter_add(
                tf.IndexedSlices(
                    m_scaled_g_values, gradient.indices
                )
            )
        else:
            # Dense gradients.
            m_t = m * self.beta_1 + gradient * (1 - self.beta_1)
            variable.assign_sub(
                lr * tf.math.sign(m_t)
            )
            m.assign(m * self.beta_2 + gradient * (1 - self.beta_2))

    def get_config(self):
        config = super(Lion, self).get_config()

        config.update({
            'learning_rate': self._serialize_hyperparameter(self._learning_rate),
            'beta_1': self.beta_1,
            'beta_2': self.beta_2
        })
        return config
