v1.0.0 9/16/2025 (first release on github)
  First functioning version on pypi. Single control volume of gases 
  and a single control volume of brine. Connected to a single pipe and single 
  radial ground heat exchanger. Gas mixtures are available through REFPROP 
  HEOS and other CoolProps fluids for single and static mixtures.
  Verified against one set of simulation results and one set of experimental
  results with a report pending. Newton solver with implicit steps an convergence forcasting.
  Generalized components that serve as an abstract class to make 
  adding and combining components more straightforward. HydDown
  code no longer exists in UH2SC
v0.0.0 3/01/2024 (never released in github)
  Original code with explicit time steps and no portability of components
  Heavily dependent on HydDown.
