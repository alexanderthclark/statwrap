Quickstart
==========

This page gives a short walkthrough for getting started with StatWrap.

Installation
------------

Install the package from PyPI using ``pip``:

.. code-block:: bash

   pip install statwrap

Using the package
-----------------

In a Jupyter or Colab notebook import ``statwrap`` and load one of the
convention modules. ``%use_fpp`` follows the style of *Statistics* by
Freedman, Pisani and Purves while ``%use_sheets`` mimics Google Sheets.

.. code-block:: python

   import statwrap
   %use_fpp  # or %use_sheets

A few convenience functions then become available. For example:

.. code-block:: python

   average(1, 2, 3)
   box_model(1, 2, 3, draws=2, random_seed=0)

See :doc:`python_primer` if you need a refresher on Python basics.
