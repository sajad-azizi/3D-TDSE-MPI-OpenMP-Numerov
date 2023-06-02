# Time-dependent Schrödinger equation
Time-independent SE is solved using the Numerov method, in which we calculate eigenvalues and eigenstates for a specific spherical potential for each angulare momentum $\ell$.




Our energy box supports bound states and continuum states up to $E_{\rm max}$. The continuum state is caculated in a box, which is called ``box" solution. The boundary condition for a box solution is that a state should reach zero at the end of the box.




Then we caculate the dipole matrix element $d_{n\ell ,  n' \ell^{\prime}} = \langle \psi_{n' \ell^{\prime}} |r| \psi_{n\ell } \rangle $


Having a Hamiltonian in the velocity gauge
$$H_{n\ell ,  n' \ell^{\prime}} = E_{n\ell} \delta_{n\ell ,  n' \ell^{\prime}} + \mathrm{i} A(t) (E_{n\ell} - E_{ n' \ell^{\prime}}) d_{n\ell ,  n' \ell^{\prime}}$$ where $A(t)$ is the vector potential, we can propagate it in time.
$$a(t + dt) = \exp[-\mathrm{i} H dt] a(t)$$
where $dt$ is the time step. In this code, we use the Taylor series to propagate over time.

In order to speed up, we used MPI and OpenMP parallelization methods, which run up to 10 times faster. However, it depends on how many cores you allow to be allocated.




At the last point, we use different shapes of laser pulses; therefore,  we defined vector potential in the frequency domain and used the Fourier transform to define it in the time domain. For the Fourier transform, we used the Fast Fourier transform package ``fftwpp".


Implementation:


```
module load mpi.intel/5.0.3
mpiicpc -Wall -O2 -std=c++17 -O3 -DNDEBUG -funroll-loops -ffast-math -lstdc++ -fopenmp fftwpp/fftw++.cc -lfftw3 -lfftw3_omp tridtdse_hybrid_final.cpp 
time mpirun -np 5 ./a.out
```
