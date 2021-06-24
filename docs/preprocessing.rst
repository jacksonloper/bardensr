.. bardensr public api preprocessing

Preprocessing
====================================

Numpy-based API for preprocessing
---------------------------------

The functions in this module focus on "imagestacks," i.e.,
in the context of this package, arrays of the form ``N x M0 x M1 x M2``.
They are designed to give relatively simple interfaces to
the functionality needed for preprocessing such imagestacks.

.. automodule:: bardensr.preprocessing
    :autosummary:
    :members:


Lower-level tensorflow-based API for preprocessing
--------------------------------------------------

The functions in this module accept general tensorflow tensors
as arguments and return them as outputs.  This has some advantages...

- Using these can save time, because tensorflow tensors can live on the
  GPU, which means you don't have to copy them back and
  forth to the CPU.
- These functions can also be included inside of tf.function blocks
  and thus compiled for more speed
  and memory efficiency.
- You can differentiate through them with tensorflow.

.. automodule:: bardensr.preprocessing.preprocessing_tf
    :autosummary:
    :members:
