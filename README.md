# Time-dependent Schrödinger equation 
time-independent SE is solved using Numerov method in which we caculate eigenvalues and eigenstates for a specific spherical potential for each angulare momentum $\ell$.



Our energy box supports bound states and continumm state up to $E_{\rm max}$. The continum state is caculated in a box which is called ``box" solutions. The boundary condition for a box solution is that a state should reach zero at the end of the box.



Then we caculate the dipole matrix element $d_{n\ell ,  n' \ell^{\prime}} = \langle \psi_{n' \ell^{\prime}} |r| \psi_{n\ell } \rangle $


Having Hamiltoniam in velocity gauge 
$$H_{n\ell ,  n' \ell^{\prime}} = E_{n\ell} \delta_{n\ell ,  n' \ell^{\prime}} + \mathrm{i} A(t) (E_{n\ell} - E_{ n' \ell^{\prime}}) d_{n\ell ,  n' \ell^{\prime}}$$ where $A(t)$ is the vector potential, we can porpagate it in time.
$$a(t + dt) = \exp[-\mathrm{i} H dt] a(t)$$
where $dt$ is the time step. In this code we use Taylor series to propagate in time.

In order to speed up, we used MPI and OpenMP prallelization method of which it runs up to 10 times  faster. Albeit, it depends on how many core you allow to alocate.
