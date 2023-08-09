# Pyloric network simulator

The pyloric circuit model described by Prinz at al.[^model-def] continues to be studied today,
both as an instructive model of an interesting biological process[^prinz-research] and also as a case study of functional diversity in computational models.[^goncalves2022]
The authors’ [also provide](https://biology.emory.edu/research/Prinz/database-sensors/) their C/C++ implementation of the model.

This repository is a reimplementation of the model in pure Python, using JAX to achieve comparable execution speeds as C/C++.

## Overview

### Main features

- Easy to install: `pip install pyloric-network-simulator "jax[cpu]"`
- Easy to use: `from pyloric_network_simulator import prinz2004`
  - No need to learn about C linkers or HDF5 file formats.
  - Model thermalizations are computed as needed.
    - In the original implementation, an initial sweep over 20 million models for individual neurons
- Easy to understand:
  - Python is easier to read than C++ and familiar to more users (especially to scientists).
  - Code structure closely follows how the model is defined in the papers, making it easier to understand each component.
  - The entire model fits in a single code file.
  - The code file can be opened in a Jupyter Notebook,[^jupytext] providing structure and formatted documentation explaining each code section in detail.  
    ![Screenshot: Conductance model](docs/inlined-docs-1.png)  ![Screenshot: Constants](docs/inlined-docs-1.png)
- Easy to modify:
  - Users have full control over the circuit topology: number, size and type of populations, as well as the synaptic conductivities, are all specified when creating the model:

    Standard pyloric circuit with 4 populations
    
        model = Prinz2004(pop_sizes = {"PD": 2, "AB": 1, "LP": 1, "PY": 5},      # Pop. sizes
                          gs      = [ [    0  ,     0  ,     3  ,     0 ],       # Syn. conductivities
                                      [    0  ,     0  ,     3  ,     3 ],
                                      [    3  ,     3  ,     0  ,     3 ],
                                      [    3  ,     3  ,     3  ,     0 ] ],
                          g_ion = neuron_models.loc[["AB/PD 3", "AB/PD 3", "LP 1", "PY 5"]]   # Neuron types
                          )

     Reduced pyloric circuit with 2 populations
    
        model = Prinz2004(pop_sizes = {"AB": 1, "LP": 1},
                          gs      = [ [    0  ,     3 ],
                                      [    3  ,     0 ] ],
                          g_ion = neuron_models.loc[["AB/PD 3", "LP 1"]]
                          )
- Fully documented: The original specification of the pyloric circuit model is spread across at least three resources[^model-def].
  The included inlined documentation collects all definitions in one place, is fully referenced and uses standardized notation.
- Fast: Special care was taken to use vectorized operations wherever possible, and JAX’s JIT capability is used to compile
  the model to C at runtime. This should provide speeds comparable with a plain C/C++ implementation.
  Use of JAX also opens the possibility of using GPU acceleration for models with many neurons.

### Limitation

The principal and very important disadvantage of this implementation is that currently it was only used for one example in one paper,
in contrast to the original C/C++ implementation which has received many years of focussed attention.
This implementaiton has *not* been exhaustively tested for consistency with the original.

- Basic qualitative comparisons suggests that the single neuron conductance models closely reproduce the results reported by Prinz et al.
- Some differences in the simulated activity (wrt to the original implementation) do seem to occur when neuron models are combined into a circuit.

### Code structure

The code uses [MyST Markdown](https://mystmd.org/) and [Jupytext](https://jupytext.readthedocs.io/), which enables a [literate programming](https://texfaq.org/FAQ-lit) style by embedding rich Markdown comments in Python comments.
The result is documentation and code which as near as possible.
This is especially useful for scientific programming, where the code itself may not be especially complicated, but may be the result of
complex arguments which need mathematics or figures to express intelligibly.

Literate code files like `prinz2004.py` are multi-purpose:
- Import them as normal Python modules.
- Run them as normal Python modules on the command line.
- When opened in Jupyter Notebook or VS Code, all markdown comments are rendered inline, placing math and figures right there alongside the code.
- Export to HTML book format (this is how we produce the documentation).

## Installation

Since the documentation is inlined with the code, we recommend including the source code in your project

1. Download the [most recent release](https://github.com/alcrene/pyloric-network-simulator/releases).  
   Alternatively, you can use [`git-subrepo`](https://github.com/ingydotnet/git-subrepo) to clone this repository into a subdirectory of your project.

2. Unpack into your project directory, so it looks something like

       <my-project>
       ├─ ...
       └─ pyloric_simulator
          ├─ config/
          ├─ prinz2004.py
          ├─ requirements.txt
          └─ ...

   or

       <my-project>
       ├─ ...
       └─ lib
          └─ pyloric_simulator
             ├─ config/
             ├─ prinz2004.py
             ├─ requirements.txt
             └─ ...

3. Install the requirements

   pip install -r ./pyloric_simulator/requirements.txt

4. Add the contents of `requirements.txt` to your own dependencies.

To use the `prinz2004.py` module, just import it as you would any of your own modules.

### Alternative: Standalone package

If you want to develop the simulator, you may prefer to clone the repository and make it a dependency to your project.

1. Choose a location for the repo

       cd ~/code
   
2. Clone the repo

       git clone git@github.com:alcrene/pyloric-network-simulator.git

3. Install in [development mode](https://setuptools.pypa.io/en/latest/userguide/development_mode.html)

       pip install -e ./pyloric-network-simulator

## Usage

See the documentation.

## Building the documentation

Ensure the *.md* and *.py* files are synchronized

    jupytext --sync pyloric_simulator/prinz2004.md

[Build](https://jupyterbook.org/en/stable/start/your-first-book.html) the documentation using

    jb build .

[Push](https://jupyterbook.org/en/stable/publish/gh-pages.html#option-2-automatically-push-your-build-files-with-ghp-import) to GitHub Pages

    ghp-import -n -p -f _build/html


[^model-def]:
    • Prinz, A. A., Bucher, D. & Marder, E. *Similar network activity from disparate circuit parameters.* Nature Neuroscience 7, 1345–1352 (2004). [doi:10.1038/nn1352](https://doi.org/10.1038/nn1352)  
    • Prinz, A. A., Billimoria, C. P. & Marder, E. *Alternative to Hand-Tuning Conductance-Based Models: Construction and Analysis of Databases of Model Neurons.*
      Journal of Neurophysiology 90, 3998–4015 (2003). [doi:10.1152/jn.00641.2003](https://doi.org/10.1152/jn.00641.2003)  
    • Marder, E. & Abbott, L. F. *Modeling small networks.* in Methods in neuronal modeling: from ions to networks (eds. Koch, C. & Segev, I.) (MIT Press, 1998).

[^prinz-research]: https://biology.emory.edu/research/Prinz/research.html
[^goncalves2022]: Gonçalves, P. J. et al. *Training deep neural density estimators to identify mechanistic models of neural dynamics.* eLife 9, e56261 (2020). [doi:10.7554/eLife.56261](https://doi.org/10.7554/eLife.56261)
[^jupytext]: Presuming the [Jupytext](https://jupytext.readthedocs.io/) extension is installed.
