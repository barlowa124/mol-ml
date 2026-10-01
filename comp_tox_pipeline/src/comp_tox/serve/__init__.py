"""Serving and monitoring layer for the trained Tox21 models.

`reference` freezes a compact profile of the training distribution
(fingerprint matrix, bit frequencies, conformal qhat, AD baseline) that
`predictor`, `drift`, and the FastAPI `app` share at runtime.
"""
