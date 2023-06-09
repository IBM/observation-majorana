## Supplementary code for "Observing and braiding topological Majorana modes on programmable quantum simulators"

**Authors: Nikhil Harle\*, Oles Shtanko\**, Ramis Movassagh**

\* nikhil.calvin@gmail.com\
\** oles.shtanko@ibm.com

Publication: [Nat Commun 14, 2286 (2023)](https://www.nature.com/articles/s41467-023-37725-0)\
Preprint: [https://arxiv.org/abs/2203.15083](https://arxiv.org/abs/2203.15083)

**Abstract:**
> Despite its great promise of fault tolerance, the simplest demonstration of topological quantum computation remains elusive. Majorana modes are the primitive building blocks and their experimental realization on various platforms is yet to be confirmed. This work presents an experimental and theoretical framework for the space-resolved detection and exchange of Majorana modes on programmable (noisy) quantum hardware. We have implemented our framework by performing a series of measurements on a driven Ising-type quantum spin model with tunable interactions, which confirm the existence of the topological Majorana modes and distinguishes them from trivial modes. Lastly, we propose and demonstrate a novel technique for braiding the Majorana modes which results in the correct statistics but decreases the magnitude of the signal. The present work may be seen as the first reliable observation of Majorana modes on existing quantum hardware.


The code contains data and reproductions of three experiments:

1. **Wavefunction extraction**
  * We generate dynamics that exhibit boundary-localized Majorana modes $\Gamma_s$; $s \in [L,R]$. We consider an ansatz in which they are linear combinations of physical Majorana fermion operators $\gamma_{\mu}$:

$$ \Gamma_s = \sum_{\mu=0}^{2N}  \psi^s_{\mu} \gamma_{\mu} $$

  * We obtain the *wavefunction* $\psi^s_{\mu}$.

2. **Verification of topological nature**
  * Since trivial modes can also produce Majorana-like signatures, we devise an experiment to distinguish trivial modes from topological Majorana modes.
  * We measure the *two-point correlation function* 


$$ T_{\mu,\nu} \simeq \frac{1}{D} \sum_{n=0}^{D-1} \bra{\psi_n} \gamma_{\mu} \gamma_{\nu} \ket{\psi_n} $$


  * Specifically, we measure the function $T_{1,2x}$ for two scenarios: a topological Majorana phase, and a trivial phase, where we remove the gates connecting the edge qubits to the rest. 
  * For the topological modes we expect $T_{1,2x}$ to have a peak when x = $N$, but for the trivial modes we expect $T_{1,2x}$ to have a peak for both $x =0$ and $x=N$.

3. **Braiding**
  * Braiding is an exchange of particle positions that can tell us about the statistics of the particles and be used for topological quantum computing. 
  * The traditional adiabatic braiding method is too slow (requires deep circuits) and difficult to implement in 1D.
  * We propose a method that can be implemented in 1D and is fast at the expense of amplitude damping.

### Contents
`tutorial.ipynb` contains the code that allows to
1. **Load Data
   * Load the experimental data used to generate the paper figures.
2. **Reproduce the experiment
  * using the Qiskit `qasm_simulator` with or without noise (noisy simulation *requires an IBM Quantum account*)
  * using actual IBM hardware (*requires an IBM Quantum account*).

`paper_data` is a folder containing the raw data used to generate the paper figures.
