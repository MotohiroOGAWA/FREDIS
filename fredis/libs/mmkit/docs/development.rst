Development
===========

This section describes how to use ``mmkit`` in development environments.

Using as a Git Submodule
------------------------

You can integrate ``mmkit`` into an existing project as a Git submodule.

.. code-block:: bash

   # At the root directory of your project
   git submodule add https://github.com/MotohiroOGAWA/mmkit.git ./mmkit
   git commit -m "Add mmkit as submodule"

Notes
-----

Using a Git submodule is useful for:

- Reproducible research environments
- Managing dependencies in HPC or cluster environments
- Integrating ``mmkit`` into larger pipelines