/*
 * created: Dec2019(sajjazizi@gmail.com)
 * module load mpi.intel/5.0.3
 * mpiicpc -Wall -O2 -std=c++17 -O3 -DNDEBUG -funroll-loops -ffast-math -xhost -lstdc++ -fopenmp fftwpp/fftw++.cc -lfftw3 -lfftw3_omp tridtdse_hybrid_final.cpp 
 * time mpirun -np 5 ./a.out
*/
#include <mpi.h>
#include <iostream>
#include <vector>
#include <complex>
#include <cstdlib>
#include <fstream>
#include <stdio.h>
#include <string>
#include <cmath>
#include <omp.h>
#include <iomanip> //this is for setprecision
#include "fftwpp/fftw++.h"
using namespace std;
using namespace utils;
using namespace fftwpp;

//fast nodes  metis[07-11] io[]
//#define NTHREADS mp_get_max_threads()
#define CHUNK 10
#define MASTER 0
#define pi 3.1415926535897932384626433
#define imax(a,b) ((a>b)?a:b)
#define pw2(a) a*a

typedef std::complex<double> dcompx;
typedef std::complex<int> icompx;
typedef std::vector<double> dvec;
typedef std::vector<std::vector<double> > ddvec;
typedef std::vector<dcompx> dcopxvec;
typedef std::vector<std::vector<dcompx> > ddcopxvec;

constexpr dcompx I{0.0, 1.0};
constexpr dcompx zero{0.0,0.0};
constexpr dcompx one{1.0,0.0};

class tridTDSE_hybrid{

    public:
        tridTDSE_hybrid();
        ~tridTDSE_hybrid();

        //functions- time-dependent SE
        void gathering_coefficients(dcopxvec &local_C, dcopxvec &rtlr, int len_E, int l);
        void BlockMartixVector_multi_hybird(dcopxvec &Bmatrix, dcopxvec &Bvector, dcopxvec &result, int len_E, int tlr, int l);
        void Hamiltonian_VG(dcopxvec &hamiltonian, dvec &Intgral_simps,dvec &pulse, dvec &E,int t_index, int len_E, int l);
        void SimpsonRule_overAlltimes_VG(dvec &phi, dvec &Intgral_simps, int procid, int len_E);
        bool Kdelta(int i, int j){return(i==j);}
        int factor(int i){if(i==0){return 1;}else{return i*factor(i-1);}}
        double t_step(int);
        double A_f(int t_index, double Tc);
        double phase_function(dcopxvec &C,int indx, int len_E, int j);
        
        //function for getting fourier transform
        double phase_fourierFunction(double w, double T, double A);
        void Creating_Boundpulse(dvec &pulse,double F_0,double T,double w0, double A);
        void Creating_FElpulse(dvec &pulse,double F_0,double tau,double T,double w0, double A);
        void pulse_generating(dvec &pulse, double A);
        
        //mpi functions
        void send_mpi(dvec &ve, int dest, int tag_);
        void receive_mpi(dvec &ve, int src, int tag_);
        void send_mpi(dcopxvec &ve, int dest, int tag_);
        void receive_mpi(dcopxvec &ve, int src, int tag_);
        
        //functions of numerov's method for solving time-independent SE
        void findingEigenvalues(dvec &E, double eps, double deps, int procid);
        double seteigenvalues_numroveMethod(double eps, double deps, int l, int n_save);
        void finding_wavefunctions_numrovSolver(dvec &wavfunc, dvec &Eigenvalue, int len_E);
        void setk2(dvec &k2,double eps,int l,int n_save);
        void setwavefunction(dvec &Psi, dvec &k2, int n_save);
        double potential(double r, int l);
        void Normalization(dvec &Psi);
        template <class T>
        string cstr(T value);
        void tri_dtindse_hybrid();

            
        //space
        
        double r_max = 2000.0;//maximum r in space graid
        double dr = 0.01;//r_max/(N_graid-1);//step in space graid
        int N_graid =  int(r_max/dr) + 1;//  200000;//number of grid in creating time-independent se wavefunctins
            
        //set trial energy
        double eps = -0.99;//trial energy
        double deps = 0.00001;//step of changing trial energy
        double e_up = 2.0;//uper limit of trial energy;let say E_max
            
        //Angular momentum
        int l_max = 5;//the masximum angular momentum
        int numproces;

        //time
        //int N_time = 54000;//number of griad in time graid
        double t_0 = -10000.0;//the starting point in time graid
        double t_m = -t_0;//the maximum point in time graid
        double dt = 0.1;//(t_m - t_0)/N_time;//step in time graid
        int N_time = int( (t_m - t_0)/dt  );//number of griad in time graid
        
        //fourier
        double dw = 2.0*pi/(N_time*dt);
        double w_0 = -0.5*N_time*dw;// w=[-n/2,n/2]*dw

        //pulse
        double Tc = 0.0;
        
        //the number of sentences of the Taylor series
        int N_L = 10;
        
        // intial state i.e. intial l
        int il_n = 0;  // il=0->start at 1s and il=1->start at 2p
        
        //the number of OpenMP threads 
        int NTHREADS = omp_get_max_threads();
            
};

tridTDSE_hybrid::tridTDSE_hybrid(){
}


tridTDSE_hybrid::~tridTDSE_hybrid(){
}

//phase in frequency domain -- zero means fourier-limited pulse
double tridTDSE_hybrid::phase_fourierFunction(double w, double T, double A){

    double tau = A*0.01*T;
    double ampl = 1.0;
    return ampl*sin(w*tau + 0.0);
}

//Vector potential
void tridTDSE_hybrid::pulse_generating(dvec &pulse, double A){
    
    double k = 1;
    double F_0 = sqrt(k/3.51); // the base is 10^16
    double w0;// = 0.88;
    w0 = 0.9;
    double T = 3.0*41.341;
    double A_0 = (F_0/w0);
    
    //bound pulse
    Creating_Boundpulse(pulse,A_0,T,w0,A);
    
    //FEL pulse
    //double tau=3.0*41.341;// a parameter in FEL pulse
    // tau=T this means the furier limited pulse in FEL pulses
    //Creating_FElpulse(pulse,A_0,tau,T,w0,A);
}

void tridTDSE_hybrid::tri_dtindse_hybrid(){
    
    int len_mpiName;
    MPI::Init();
    MPI_Status status;
    int procid = MPI::COMM_WORLD.Get_rank();
    numproces = MPI::COMM_WORLD.Get_size();
    char name[MPI_MAX_PROCESSOR_NAME];
    MPI_Get_processor_name(name, &len_mpiName);
    l_max = this->numproces;
    cout << "Processor name("<<procid<<"): "<<name<<endl;
    
    dvec local_E;
    dvec E;
    ddvec tE;
    dvec R;
    dvec local_Intgral_simps;
    dvec Intgral_simps;
    dcopxvec C;
    dcopxvec local_C;
    dcopxvec local_hamiltonian;
    dvec pulse;
    dcopxvec local_rtlr;
    dcopxvec rtlr;
    dcopxvec Lrtlr;
    int len_E;
    int flag = 1;
    
    //finding dynamically t_min
    ifstream inpulse0("A_ini.dat");
    inpulse0 >> Tc;
    pulse.resize(N_time);
    pulse_generating(pulse,Tc);
    int i_save = 0;
    for(int i = 0; i < N_time; i++){
        if(pulse[i] > 5.0e-5){
            i_save = i;
            break;
        }
    }
    t_0 = t_0 + i_save*dt;
    N_time = int( (-2.0*t_0)/dt  );
    t_0 = this->t_0;
    N_time = this->N_time;
    dw = 2.0*pi/(N_time*dt);
    w_0 = -0.5*N_time*dw;// w=[-n/2,n/2]*dw
    dw = this->dw;
    w_0 = this->w_0;
    
    if(procid == MASTER){ //master
        
        cout << "Number of CPU: "<<NTHREADS<<endl;
        cout << "L_MAX = "<< l_max<< "\t" << numproces <<endl;
        cout << "N time = " << N_time <<" dt = " << dt << endl;
        ofstream plsout;
        
        ifstream inpulse("A_ini.dat");
        inpulse >> Tc;
        
        pulse.clear();
        
        pulse.resize(N_time);
        pulse_generating(pulse,Tc);
        cout << "plotting ... \n";
        plsout.open("givenpulse_"+cstr(Tc)+".dat");
        for(int t = 0; t < N_time; t++){
            plsout<<t_step(t)<<"\t"<<pulse[t]<<endl;
        }
        plsout.close();
        
        //exit(0);
        
        
        cout << "Step 1: finding eigenvalues and wavefunctions for Time-independent SE(numerove's Method) ....\n";
        ifstream Iin("../../Integraldipole_"+cstr(r_max)+"_"+cstr(l_max)+"_"+cstr(e_up)+".dat");
        ifstream ein("../../eigenvalues_"+cstr(r_max)+"_"+cstr(l_max)+"_"+cstr(e_up)+".dat");
        if(!ein.is_open()){        
            findingEigenvalues(local_E,eps,deps,procid);
            MPI_Barrier(MPI_COMM_WORLD);
            
            E.resize(l_max*local_E.size());
            tE.resize(l_max);for(int i = 0; i < l_max; i++)tE[i].resize(local_E.size());
            for(int i = 0; i < local_E.size(); i++){
                E[0*local_E.size()+ i]=local_E[i];
                tE[0][i] = local_E[i];
                //cout << "energy = " <<"E("<<procid<<","<<i<<")= "<< E[0][i] << endl;
            }
            for(int source = 1; source < numproces; source++){
                receive_mpi(local_E,source,0);
                E.resize(l_max*local_E.size());
                tE.resize(l_max);for(int i = 0; i < l_max; i++)tE[i].resize(local_E.size());
                for(int i = 0; i < local_E.size(); i++){
                    E[source*local_E.size()+i]=local_E[i];
                    tE[source][i] = local_E[i];
                    //cout << "energy = " <<"E("<<source<<","<<i<<")= "<< E[source][i] << endl;
                }
                if(source == (numproces-1)) len_E = local_E.size();
            }
            tE.resize(l_max);
            for(int l = 0; l < l_max; l++)
                tE[l].resize(tE[l_max-1].size());
            
            len_E = tE[0].size();
            cout <<"Number of eigenvalues: "<< l_max*len_E << endl;
            E.resize(l_max*len_E);
            for(int l = 0; l < l_max; l++){
                for(int n = 0; n < len_E; n++)
                    E[l*len_E+n] = tE[l][n];
            }
            ofstream eout("eigenvalues_"+cstr(r_max)+"_"+cstr(l_max)+"_"+cstr(e_up)+".dat");
            for(int l = 0; l < l_max; l++){
                for(int n = 0; n < len_E; n++){
                    eout << l <<"\t"<<n<<"\t"<< E[l*len_E+n] << endl;
                }
            }
            eout.close();
            finding_wavefunctions_numrovSolver(R,E,len_E);
            
            local_E.clear();
            local_E.resize(0);
            local_E.shrink_to_fit();
            tE.clear();
            tE.resize(0);
            tE.shrink_to_fit();
        }
        else{

            double lin,nin,egin;
            int il = 0;
            while(ein>>lin>>nin>>egin){
                if(nin == 0 && lin!=0)il++;
                tE.push_back(dvec());
                tE[il].push_back(egin);
            }
            tE.resize(l_max);
            for(int l = 0; l < l_max; l++)
                    tE[l].resize(tE[l_max-1].size());

            len_E = tE[0].size();
            cout <<"Number of eigenvalues: "<< l_max*len_E << endl;
            E.resize(l_max*len_E);
            for(int l = 0; l < l_max; l++){
                for(int n = 0; n < len_E; n++)
                    E[l*len_E+n] = tE[l][n];
            }
            
            tE.clear();
            tE.resize(0);
            tE.shrink_to_fit();
            
            if(!Iin.is_open())
                finding_wavefunctions_numrovSolver(R,E,len_E);
        }
        
        for(int dest = 1; dest < numproces; dest++){
            MPI_Send(&len_E,1,MPI_INT,dest,2, MPI_COMM_WORLD);
        }
        
        cout << "Step 2: calculating Integral, the dipole part(simpson rule) ....\n";
        if(!Iin.is_open()){
            
            for(int dest = 1; dest < numproces; dest++){
                send_mpi(R,dest,1);
            }
            Intgral_simps.resize(l_max*len_E*l_max*len_E);
            SimpsonRule_overAlltimes_VG(R,local_Intgral_simps, procid,len_E);
            
            for(int n = 0; n < len_E; n++){
                for(int lp = 0; lp < l_max; lp++)
                    for(int np = 0; np < len_E; np++){
                        Intgral_simps[(0*len_E+n)*(l_max*len_E)+(lp*len_E+np)] =local_Intgral_simps[n*(l_max*len_E)+(lp*len_E+np)];
                    }
            }
            for(int source = 1; source < numproces; source++){
                receive_mpi(local_Intgral_simps,source,3);
                for(int n = 0; n < len_E; n++){
                    for(int lp = 0; lp < l_max; lp++)
                        for(int np = 0; np < len_E; np++)
                            Intgral_simps[(source*len_E+n)*(l_max*len_E)+(lp*len_E+np)] =local_Intgral_simps[n*(l_max*len_E)+(lp*len_E+np)];
                }
            }
            ofstream Inout("Integraldipole_"+cstr(r_max)+"_"+cstr(l_max)+"_"+cstr(e_up)+".dat");
            for(int l = 0; l < numproces; l++){
                for(int n = 0; n < len_E; n++){
                    for(int lp = 0; lp < l_max; lp++)
                        for(int np = 0; np < len_E; np++)
                            Inout << Intgral_simps[(l*len_E+n)*(l_max*len_E)+(lp*len_E+np)]<<endl;
                }
            }
            Inout.close();
            local_Intgral_simps.clear();
            local_Intgral_simps.resize(0);
            local_Intgral_simps.shrink_to_fit();
           
        }
        else{
            Intgral_simps.resize(l_max*len_E*l_max*len_E);
            double Intpij;
            for(int l = 0; l < l_max; l++){
                for(int n = 0; n < len_E; n++){
                    for(int lp = 0; lp < l_max; lp++){
                        for(int np = 0; np < len_E; np++){
                            Iin >> Intpij;
                            Intgral_simps[(l*len_E+n)*(l_max*len_E)+(lp*len_E+np)]=(Intpij);
                        }
                    }
                }
            }
        }
        R.clear(); //clear content
        R.resize(0); //resize it to 0
        R.shrink_to_fit();
        

        //exit(1);
        for(int dest = 1; dest < numproces; dest++){
            send_mpi(Intgral_simps,dest,4);
            send_mpi(E,dest,5);
        }
        
        for(int dest = 1; dest < numproces; dest++){
            send_mpi(pulse,dest,71);
        }
        
        C.resize(l_max*len_E);
        for(int i=0;i<l_max;i++)
            for(int j=0;j<len_E;j++)
                C[i*len_E+j] = 0.0;
            
        /****************************************************
        *NOTE 
        * 2p state = n = 2 = n_r + l + 1 = 0 + 1 + 1
        * where the n_r = 0,1,2,.. is radial quantum number
        * n = 1,2,3,... is principal quantum number
        * l = 0,1,2,3,... is angular momentum quantum number
        ****************************************************/
        C[il_n*len_E+0] = 1.0; //C[1][0]//2p state
        
        ddvec lastoccpy;
        for(int l = 0; l < l_max; l++){
            lastoccpy.push_back(dvec());
            for(int n = 0; n < len_E; n++){
                lastoccpy[l].push_back(0.0);
            }
        }
        
        ofstream oout("groundstatoccpy_"+cstr(Tc)+".dat");
        ofstream phout("phase_"+cstr(Tc)+".dat");
        
        
        cout << "Step 3: propagation ... \n";
        for(int t = 0; t < N_time; t++){
            
            oout<<t_step(t)<<"\t"<<pw2(abs(C[0]))<<"\t"<<pw2(abs(C[1*len_E+0]))<<endl;
            double dsum=0.0;for(int i=0;i<l_max;i++)for(int j=0;j<len_E;j++)dsum+=abs(C[i*len_E+j])*abs(C[i*len_E+j]);
            

            Hamiltonian_VG(local_hamiltonian,Intgral_simps,pulse,E,t,len_E, procid);
            for(int dest = 1; dest < numproces; dest++){
                MPI_Send(&t,1,MPI_INT,dest,88, MPI_COMM_WORLD);
            }
            
            //{
            
            rtlr.resize(l_max*len_E*N_L);
            for(int l = 0; l < l_max; l++)
                for(int n = 0; n < len_E; n++)
                    rtlr[(l)*(N_L*len_E)+(n*N_L+(0))] = C[l*len_E+n];
            
            for(int tlr = 1; tlr < N_L; tlr++){
                
                for(int dest = 1; dest < numproces; dest++){
                    MPI_Send(&tlr,1,MPI_INT,dest,9, MPI_COMM_WORLD);
                    send_mpi(rtlr,dest,10);
                }
                
                BlockMartixVector_multi_hybird(local_hamiltonian,rtlr,local_rtlr,len_E,tlr,procid);
                
                for(int n = 0; n < len_E; n++)
                    rtlr[(0)*(N_L*len_E)+(n*N_L+(tlr))] = local_rtlr[n];
                    
                for(int source = 1; source < numproces; source++){
                    receive_mpi(local_rtlr,source,11);
                    for(int n = 0; n < len_E; n++)
                        rtlr[(source)*(N_L*len_E)+(n*N_L+(tlr))] = local_rtlr[n];
                }
            }

            C.clear();
            C.resize(l_max*len_E);
            
            for(int dest = 1; dest < numproces; dest++){
                send_mpi(rtlr,dest,12);
            }
            gathering_coefficients(local_C,rtlr,len_E,procid);
            for(int n = 0; n < len_E; n++)
                C[0*len_E + n] = local_C[n];
            for(int source = 1; source < numproces; source++){
                receive_mpi(local_C,source,13);
                for(int n = 0; n < len_E; n++)
                    C[source*len_E + n] = local_C[n];
            }
            
            local_C.clear();
            local_C.shrink_to_fit();
            rtlr.clear();
            rtlr.shrink_to_fit();
            
            //}
            
            local_hamiltonian.clear();
            local_hamiltonian.resize(0);
            local_hamiltonian.shrink_to_fit();
            
           if(t == N_time-5){
                ofstream lout("lasttimeMPI_"+cstr(Tc)+".dat");
                for(int j = 0; j < len_E; j++){
                    lout <<E[0*len_E+j]<<"\t"<<abs(C[0*len_E+j])*abs(C[0*len_E+j])<<"\t"<<abs(C[1*len_E+j])*abs(C[1*len_E+j])\
                    <<"\t"<<abs(C[2*len_E+j])*abs(C[2*len_E+j])<<"\t"<<abs(C[3*len_E+j])*abs(C[3*len_E+j])<<"\t"<<abs(C[4*len_E+j])*abs(C[4*len_E+j])<<endl;
                }
                lout.close();
                for(int i = 0; i < l_max; i++){
                    for(int j = 0; j < len_E; j++)
                        lastoccpy[i][j] = abs(C[i*len_E+j])*abs(C[i*len_E+j]);
                }
            }
            
            
            //phase of population
            for(int j = 0; j < 1; j++){
                phout<<t_step(t)<<"\t"<<phase_function(C,0,len_E,j)<<"\t"<<phase_function(C,1,len_E,j)<<"\t"<<phase_function(C,2,len_E,j)<<"\t"<<phase_function(C,3,len_E,j)<<"\t"<<phase_function(C,4,len_E,j)<<endl;
            }
            

           if(t%1000 == 0) cout << "time = "<< t << "  & sum(|C_ln|^2)= "<< dsum << endl;
        }
        
        
        Intgral_simps.clear();
        Intgral_simps.resize(0);
        Intgral_simps.shrink_to_fit();
        C.clear();
        C.resize(0);
        C.shrink_to_fit();

        ofstream pout("spectrumMPI_"+cstr(Tc)+".dat");
        auto K_Gaussian = [&](double E, double dE)->double{
            double norm_n = (1./(sqrt(pi)*abs(dE)));
            return norm_n*exp(-(E*E)/(dE*dE));
        };
        double dE = 0.0025;
        for(double Es = -0.0; Es < e_up; Es += 0.0001){
            double p_sum=0.0,p_suml0=0.0,p_suml1=0.0,p_suml2=0.0,p_suml3=0.0,p_suml4=0.0;
            for(int j = 0; j < lastoccpy[0].size(); j++){
                if(E[0*len_E+j]>0)
                    p_suml0 += lastoccpy[0][j]*K_Gaussian(Es - E[0*len_E+j],dE);
            }
            if(l_max>1)
            for(int j = 0; j < lastoccpy[1].size(); j++){
                if(E[1*len_E+j]>0)
                    p_suml1 += lastoccpy[1][j]*K_Gaussian(Es - E[1*len_E+j],dE);
            }
            if(l_max>2)
            for(int j = 0; j < lastoccpy[2].size(); j++){
                if(E[2*len_E+j]>0)
                    p_suml2 += lastoccpy[2][j]*K_Gaussian(Es - E[2*len_E+j],dE);
            }
            if(l_max>3)
            for(int j = 0; j < lastoccpy[3].size(); j++){
                if(E[3*len_E+j]>0)
                    p_suml3 += lastoccpy[3][j]*K_Gaussian(Es - E[3*len_E+j],dE);
            }
            if(l_max>4)
            for(int j = 0; j < lastoccpy[4].size(); j++){
                if(E[4*len_E+j]>0)
                    p_suml4 += lastoccpy[4][j]*K_Gaussian(Es - E[4*len_E+j],dE);
            }
            for(int l = 0; l < l_max; l++){
                for(int j = 0; j < lastoccpy[l].size(); j++){
                    if(E[l*len_E+j]>0)
                        p_sum += lastoccpy[l][j]*K_Gaussian(Es - E[l*len_E+j],dE);
                }
            }
            pout<<Es<<"\t"<<p_sum<<"\t"<<p_suml0<<"\t"<<p_suml1<<"\t"<<p_suml2<<"\t"<<p_suml3<<"\t"<<p_suml4<<endl;
        }
        pout.close();
        
        lastoccpy.clear();
        lastoccpy.resize(0);
        lastoccpy.shrink_to_fit();
        E.clear();
        E.resize(0);
        E.shrink_to_fit();
        
    }
    else{
        
        ifstream ein("../../eigenvalues_"+cstr(r_max)+"_"+cstr(l_max)+"_"+cstr(e_up)+".dat");
        if(!ein.is_open()){
            findingEigenvalues(local_E,eps,deps,procid);
            MPI_Barrier(MPI_COMM_WORLD);
            send_mpi(local_E,MASTER,0);
        }
        
        //int len_E;
        MPI_Recv(&len_E, 1, MPI_INT, MASTER, 2, MPI_COMM_WORLD, &status);
        //cout << len_E << endl;
        ifstream Iin("../../Integraldipole_"+cstr(r_max)+"_"+cstr(l_max)+"_"+cstr(e_up)+".dat");
        if(!Iin.is_open()){
            receive_mpi(R,MASTER,1);
            SimpsonRule_overAlltimes_VG(R,local_Intgral_simps, procid,len_E);
            send_mpi(local_Intgral_simps,MASTER,3);
        }
        
        receive_mpi(Intgral_simps,MASTER,4);
        receive_mpi(E,MASTER,5);
        receive_mpi(pulse,MASTER,71);
        while(flag==1){
            int t_in;
            MPI_Recv(&t_in, 1, MPI_INT, MASTER, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            if(status.MPI_TAG == 88){
                //if(procid==1)cout << "ti = " << t_in << endl;
                Hamiltonian_VG(local_hamiltonian,Intgral_simps,pulse,E,t_in,len_E, procid);
                while(1){
                    int itr;
                    MPI_Recv(&itr, 1, MPI_INT, MASTER, 9, MPI_COMM_WORLD, &status);
                    receive_mpi(rtlr,MASTER,10);
                    BlockMartixVector_multi_hybird(local_hamiltonian,rtlr,local_rtlr,len_E,itr,procid);
                    send_mpi(local_rtlr,MASTER,11);
                    if(itr==N_L-1) break;
                }
                
                receive_mpi(Lrtlr,MASTER,12);
                gathering_coefficients(local_C,Lrtlr,len_E,procid);
                send_mpi(local_C,MASTER,13);
                
                if(t_in == N_time-1) break;
            }
            else{
                flag = 0;
                cout << "heyyyyy I am here look! \n";
            }
        }
        //MPI_Barrier(MPI_COMM_WORLD);
        
    }
    
    MPI_Finalize();
    
}
//Vector potential
double tridTDSE_hybrid::A_f(int t_in, double Tc){

    double k = 1.0;
    double F_0 = sqrt(k/3.51);
    ifstream win("w_ini.dat");
    double w0;// = Tc*0.01;//or 16eV
    win >> w0;
    w0 = 0.01*w0;
    double t = t_step(t_in);
   
    double beta = Tc;
    double T = 0.6*41.341;///sqrt(1.0 + beta*beta);
    double A_0 = (F_0/w0);// * pow(1.0 + beta*beta, 0.25);

    double A_beta = A_0/pow(1.0 + beta*beta, 0.25);
    double T_beta = T*sqrt(1.0 + beta*beta);
    double phase_beta = w0*t + ( 2.0*beta*log(2.0)/(1.0 + beta*beta) )*(t*t/(T*T));

    return A_beta*exp(-2.0*log(2.0)*t*t/(T_beta*T_beta))*cos(phase_beta);
}

//bound pulse
void tridTDSE_hybrid::Creating_Boundpulse(dvec &pulse,double F_0,double T,double w0, double A){
    
    fftw::maxthreads=get_max_threads();
    
    fft1d Forward(N_time,-1);
    Complex *f=ComplexAlign(N_time);
    for(unsigned int i=0; i < N_time; i++){
        double w = w_0 + i*dw;
        //Complex g = exp(I*phase_fourierFunction(w,T,A))*exp(-(w-w0)*(w-w0)*(T*T/(8.0*log(2.0))));
        //f[i] = F_0*(T/sqrt(8.0*M_PI*log(2.0)))*g;
        Complex g = exp(I*phase_fourierFunction(w,T,A))*exp(-(w-w0)*(w-w0)*(T*T/(4)));
        f[i] = F_0*(T/sqrt(4.0*M_PI))*g;
    }
    Forward.fft(f);
    
    for(unsigned int i=0; i < N_time; i+=1){
        int fftshift = (i+N_time/2)%N_time;
        if(i%2==0){
            pulse[i] = real(f[fftshift]*dw);
        }
        else{
            pulse[i] = real(-f[fftshift]*dw);
        }
    }
    deleteAlign(f);
}
//FEL pulse
void tridTDSE_hybrid::Creating_FElpulse(dvec &pulse,double F_0,double tau,double T,double w0, double A){
    
    fftw::maxthreads=get_max_threads();
    //pulse
    double Tp = tau*T*sqrt(2.0/(T*T+tau*tau));
    
    Complex *f=ComplexAlign(N_time);
    
    fft1d Forward(N_time,-1);
    fft1d Backward(N_time,1);
    for(unsigned int i=0; i < N_time; i++){
        double tp = t_0 + i*dt;
        f[i] = exp(-log(2.0)*tp*tp/(tau*tau))*cos(w0*tp);
    }
    Forward.fft(f);
    for(unsigned int i=0; i < N_time; i++){
        double w = w_0 + i*dw;
        f[i] = exp(Complex(0.0,1.0)*phase_fourierFunction(w-w_0,Tp,A))*f[i];
    }
    Backward.fftNormalized(f);
    for(unsigned int i=0; i < N_time; i+=1){
        double t = t_0 + i*dt;
        pulse[i] = real(F_0*exp(-log(2.0)*t*t/(T*T))*f[i]);
    }
    deleteAlign(f);
}

double tridTDSE_hybrid::phase_function(dcopxvec &C,int indx, int len_E, int j){
    double x = real(C[indx*len_E+j]);
    double y = imag(C[indx*len_E+j]);
    double phase_;
    if(x > 0){
        phase_ =  atan(y/x);
    }
    else if(x < 0){
        if(y >= 0){
            phase_ =  atan(y/x) + pi;
        }
        else{
            phase_ =  atan(y/x) - pi;
        }
    }
    
    return atan2(y,x);
    
}


void tridTDSE_hybrid::gathering_coefficients(dcopxvec &local_C, dcopxvec &rtlr, int len_E, int l){
    
    local_C.resize(len_E);
    omp_set_num_threads(NTHREADS);
    #pragma omp parallel for default(shared) schedule(static, CHUNK)
    for(int n = 0; n < len_E; n++){
        local_C[n] = rtlr[(l)*(N_L*len_E)+(n*N_L+(0))] + (pow(-I*dt,1)/double(factor(1)))*rtlr[(l)*(N_L*len_E)+(n*N_L+(1))] + \
            (pow(-I*dt,2)/double(factor(2)))*rtlr[(l)*(N_L*len_E)+(n*N_L+(2))] + \
            (pow(-I*dt,3)/double(factor(3)))*rtlr[(l)*(N_L*len_E)+(n*N_L+(3))] + \
            (pow(-I*dt,4)/double(factor(4)))*rtlr[(l)*(N_L*len_E)+(n*N_L+(4))] + \
            (pow(-I*dt,5)/double(factor(5)))*rtlr[(l)*(N_L*len_E)+(n*N_L+(5))] + \
            (pow(-I*dt,6)/double(factor(6)))*rtlr[(l)*(N_L*len_E)+(n*N_L+(6))] + \
            (pow(-I*dt,7)/double(factor(7)))*rtlr[(l)*(N_L*len_E)+(n*N_L+(7))] + \
            (pow(-I*dt,8)/double(factor(8)))*rtlr[(l)*(N_L*len_E)+(n*N_L+(8))] + \
            (pow(-I*dt,9)/double(factor(9)))*rtlr[(l)*(N_L*len_E)+(n*N_L+(9))];
    }
}

//multiply matrix-vector
void tridTDSE_hybrid::BlockMartixVector_multi_hybird(dcopxvec &Bmatrix, dcopxvec &Bvector, dcopxvec &result, int len_E, int tlr, int l){
    
    result.resize(len_E);
  

    int n,np;
    omp_set_num_threads(NTHREADS);
    #pragma omp parallel shared(Bmatrix,result,Bvector) private(n,np) 
    {
        #pragma omp for schedule(static)
        for(n = 0; n < len_E; n++){
            result[n] = zero;
            for(np = 0; np < len_E; np+=1){
                result[n] = result[n] + Bmatrix[n*(l_max*len_E)+(0*len_E+np)]*Bvector[(0)*(N_L*len_E)+(np*N_L+(tlr-1))] + \
                                        Bmatrix[n*(l_max*len_E)+(1*len_E+np)]*Bvector[(1)*(N_L*len_E)+(np*N_L+(tlr-1))] + \
                                        Bmatrix[n*(l_max*len_E)+(2*len_E+np)]*Bvector[(2)*(N_L*len_E)+(np*N_L+(tlr-1))] + \
                                        Bmatrix[n*(l_max*len_E)+(3*len_E+np)]*Bvector[(3)*(N_L*len_E)+(np*N_L+(tlr-1))] + \
                                        Bmatrix[n*(l_max*len_E)+(4*len_E+np)]*Bvector[(4)*(N_L*len_E)+(np*N_L+(tlr-1))];
            }
        }//(lp)*(N_L*len_E)+(np*N_L+(tlr-1))
    }

}

// for velocity gauge ... A(t)*i*d/dx
void tridTDSE_hybrid::Hamiltonian_VG(dcopxvec &hamiltonian, dvec &Intgral_simps, dvec &pulse, dvec &E, int t_index, int len_E, int l){
    //cout << t_index << endl;
    hamiltonian.resize(len_E*l_max*len_E);
    double At = pulse[t_index];
    int n,lp,np;
    omp_set_num_threads(NTHREADS);
    #pragma omp parallel for default(shared) schedule(static, CHUNK) private(lp,np) 
    for(n = 0; n < len_E; n++){
        for(lp = 0; lp < l_max; lp++){
            for(np = 0; np < len_E; np++){
                if(abs(l-lp)==1)
                    hamiltonian[n*(l_max*len_E)+(lp*len_E+np)]=(E[l*len_E+n]*Kdelta(l,lp)*Kdelta(n,np) - At*I*(E[l*len_E+n] - E[lp*len_E+np])*Intgral_simps[(l*len_E+n)*(l_max*len_E)+(lp*len_E+np)]);
                else
                    hamiltonian[n*(l_max*len_E)+(lp*len_E+np)]=(E[l*len_E+n]*Kdelta(l,lp)*Kdelta(n,np));
            }
        }
    }
}


double tridTDSE_hybrid::t_step(int t_index){
    return t_0 + (t_index)*dt;
}

//for velocity gauge
void tridTDSE_hybrid::SimpsonRule_overAlltimes_VG(dvec &phi, dvec &Intgral_simps, int l, int len_E){
    
    Intgral_simps.resize(l_max*len_E*len_E);

    cout << " /ell :" << l << endl; 
    int ll;
    int n,lp,np;
    //ofstream Inout("intg.dat");
    omp_set_num_threads(NTHREADS);
    #pragma omp parallel for default(shared) schedule(static, CHUNK) private(lp,np)
    for(n = 0; n < len_E; n++){
        for(lp = 0; lp < l_max; lp++){
            for(np = 0; np < len_E; np++){
                if(abs(l-lp)==1){
                    double sum = 0.0;
                    for(int i = 1; i < N_graid; i++){
                        if(i%2==0 && i!=0 && i!=(N_graid - 1)){
                            sum += 2.*phi[(l*len_E*N_graid)+(n*N_graid+i)]*(i*dr)*(i*dr)*(i*dr)*phi[(lp*len_E*N_graid)+(np*N_graid+i)];
                        }
                        else if(i%2!=0 && i!=0 && i!=(N_graid - 1)){
                            sum += 4.*phi[(l*len_E*N_graid)+(n*N_graid+i)]*(i*dr)*(i*dr)*(i*dr)*phi[(lp*len_E*N_graid)+(np*N_graid+i)];
                        }
                        else{
                            sum += phi[(l*len_E*N_graid)+(n*N_graid+i)]*(i*dr)*(i*dr)*(i*dr)*phi[(lp*len_E*N_graid)+(np*N_graid+i)];
                        }
                    }
                    ll = imax(l,lp);
                    Intgral_simps[n*(l_max*len_E)+(lp*len_E+np)]=((dr/3.0)*sum * double(ll*1.0/sqrt(4.0*ll*ll-1.0)));
                    //Inout << ((dr/3.0)*sum * double(ll*1.0/sqrt(4.0*ll*ll-1.0))) << endl;
                }
                else{
                    Intgral_simps[n*(l_max*len_E)+(lp*len_E+np)]=(0.0);
                    //Inout << Intgral_simps[n*(l_max*len_E)+(lp*len_E+np)] << endl;
                }
            }
        }
    }
}

//===========================MPI sending and receiving======================================
void tridTDSE_hybrid::send_mpi(dvec &ve, int dest, int tag_){

    unsigned len = ve.size();
    MPI_Send(&len, 1, MPI_UNSIGNED, dest, tag_, MPI_COMM_WORLD);

    if(len != 0) {
        MPI_Send(ve.data(), len, MPI::DOUBLE, dest, tag_, MPI_COMM_WORLD);
    }
}

void tridTDSE_hybrid::receive_mpi(dvec &ve, int src, int tag_){

    unsigned len;
    MPI_Status s;
    MPI_Recv(&len, 1, MPI_UNSIGNED, src, tag_, MPI_COMM_WORLD, &s);

    if(len != 0){
        ve.resize(len);
        MPI_Recv(ve.data(), len, MPI::DOUBLE, src, tag_, MPI_COMM_WORLD, &s);
    } 
    else ve.clear();
}
//complex
void tridTDSE_hybrid::send_mpi(dcopxvec &ve, int dest, int tag_){

    unsigned len = ve.size();
    MPI_Send(&len, 1, MPI_UNSIGNED, dest, tag_, MPI_COMM_WORLD);

    if(len != 0) {
        MPI_Send(ve.data(), len, MPI::DOUBLE_COMPLEX, dest, tag_, MPI_COMM_WORLD);
    }
}

void tridTDSE_hybrid::receive_mpi(dcopxvec &ve, int src, int tag_){

    unsigned len;
    MPI_Status s;
    MPI_Recv(&len, 1, MPI_UNSIGNED, src, tag_, MPI_COMM_WORLD, &s);

    if(len != 0){
        ve.resize(len);
        MPI_Recv(ve.data(), len, MPI::DOUBLE_COMPLEX, src, tag_, MPI_COMM_WORLD, &s);
    } 
    else ve.clear();
}

//===========================time independent==============================================
void tridTDSE_hybrid::findingEigenvalues(dvec &E, double eps, double deps, int l){
    double e_tr=eps;
    double de = deps;
    int mesh = N_graid-1;
    double *vpot = new double[mesh+1];
    double *f = new double[mesh+1];
    double *y = new double[mesh+1];
    double ddx12 = dr*dr/12.0;
    int n_save = N_graid;
    double r_target;
    //for(int l = 0; l < l_max; l++){
        //E.push_back(dvec());
        int inode = 0;
        while(e_tr < e_up){
            if(e_tr<0.0) de = 0.0001;
            else de = deps;
            if(e_tr < -0.033498056313579){
                for(int i = 0; i <= mesh; ++i){
                    double r = 0.00000000001 + (double) i * dr;
                    vpot[i] = potential(r,l);
                }
                f[0] = ddx12 * (2.*(vpot[0] - e_tr));
                int icl = -1;
                for(int i = 1; i <= mesh; ++i){
                    f[i] = ddx12 * 2. * (vpot[i] - e_tr);
                    /*
                    beware: if f(i) is exactly zero the change of sign is not observed
                    the following line is a trick to prevent missing a change of sign
                    in this unlikely but not impossible case:
                    */
                    if(f[i] == 0.) {
                        f[i] = 1e-20;
                    }
                    /*   store the index 'icl' where the last change of sign has been found */
                    if(f[i] != copysign(f[i],f[i-1])) {
                        icl = i;
                    }
                }
                //r_target = -1.0/e_tr;
                n_save =icl+int(N_graid/30);// int(abs(r_target)/dr) + int(N_graid/100);
                //cout << n_save <<"    "<<int(abs(r_target)/dr) + int(N_graid/100)<<endl;

            }
            else{
                n_save = N_graid;
            }
            //cout << n_save << endl;
            e_tr = seteigenvalues_numroveMethod(e_tr,de,l,n_save);
            E.push_back(e_tr);
            cout <<"E("<<l<<","<<inode<<")= "<<e_tr << endl;

            e_tr = e_tr + de;
            inode++;
        }
        e_tr = E[0];
    //}
    delete[] vpot;delete[] f;
}


void tridTDSE_hybrid::finding_wavefunctions_numrovSolver(dvec &wavfunc, dvec &Eigenvalue, int len_E){
    
    wavfunc.resize(l_max*len_E*N_graid);
    int mesh = N_graid-1;
    double *vpot = new double[mesh+1];
    double *f = new double[mesh+1];
    double *y = new double[mesh+1];
    
    ofstream fout;
    
    double ddx12 = dr*dr/12.0;
    
    for(int l = 0; l < l_max; l++){
        //wavfunc.push_back(ddvec());
        for(int e = 0; e < len_E; e++){
            //cout <<"Energy I am here!!!!!!!!!\n";
            double e_tr = Eigenvalue[l*len_E+e];
            cout <<"E("<<l<<","<<e<<")= "<< e_tr << endl;
            
            if(e_tr < -0.0005){
                for(int i = 0; i <= mesh; ++i){
                    double r = 0.00000000001 + (double) i * dr;
                    vpot[i] = potential(r,l);
                }
                f[0] = ddx12 * (2.*(vpot[0] - e_tr));
                int icl = -1;
                for(int i = 1; i <= mesh; ++i){
                    f[i] = ddx12 * 2. * (vpot[i] - e_tr);
                    /*
                    beware: if f(i) is exactly zero the change of sign is not observed
                    the following line is a trick to prevent missing a change of sign 
                    in this unlikely but not impossible case:
                    */
                    if(f[i] == 0.) {
                        f[i] = 1e-20;
                    }
                    /*   store the index 'icl' where the last change of sign has been found */
                    if(f[i] != copysign(f[i],f[i-1])) {
                        icl = i;
                    }
                }
                
                if (icl < 0 || icl >= mesh - 2) {
                    fprintf (stderr, "%4d %4d\n", icl, mesh);
                    fprintf (stderr, "error in solve_sheq: last change of sign too far\n");
                    exit(1);
                }

                /*   f(x) as required by the Numerov algorithm  */

                for(int i = 0; i <= mesh; ++i) {
                    f[i] = 1. - f[i];
                }

                for(int i = 0; i <= mesh; ++i) {
                    y[i] = 0.;
                }
                
                y[0] = 0.;
                y[1] = dr;
                
                int ncross = 0;
                for(int i = 1; i <= icl-1; ++i) {
                    y[i + 1] = ((12. - f[i] * 10.) * y[i] - f[i - 1] * y[i - 1])/ f[i + 1];
                    if (y[i] != copysign(y[i],y[i+1]))
                        ++ncross;
                }
                double yicl = y[icl];

                y[mesh] = dr;
                y[mesh - 1] = (12. - 10.*f[mesh]) * y[mesh] / f[mesh-1];

                /*Inward integration */
                for(int i = mesh - 1; i >= icl+1; --i) {
                    y[i-1] = ((12. - 10.*f[i]) * y[i] - f[i+1] * y[i+1]) / f[i-1];
                    
                    if(y[i - 1] > 1e10){
                        for(int j = mesh; j >= i-1; --j){
                            y[j] /= y[i - 1];
                        }
                    }
                }
                /*	Rescale function to match at the classical turning point (icl) */
                yicl /= y[icl];
                for(int i = icl; i <= mesh; ++i) {
                    y[i] *= yicl;
                }
                double norm = 0.;
                for(int i = 0; i <= mesh; ++i) {
                    norm += y[i]*y[i];
                }
                norm = sqrt(norm*dr);
                for (int i = 0; i <= mesh; ++i) {
                    y[i] /= norm;
                }
                
               //wavfunc[l].push_back(dvec());
                if(e < 2)
                    fout.open("psi_"+cstr(l)+"_"+cstr(e)+".dat");
                for(int i = 1; i <= mesh; i++){
                    if(e < 2)
                        fout << i*dr << "\t" << y[i]<< endl;
                    //wavfunc[l][e].push_back(y[i]/(i*dr));
                    wavfunc[(l*len_E*N_graid)+(e*N_graid+(i-1))] = y[i]/(i*dr);
                }
                if(e < 2)
                    fout.close();
            }
            else{
                dvec psi_l(N_graid,0);
                dvec k2_l(N_graid,0);
                setk2(k2_l,e_tr,l,N_graid);
                setwavefunction(psi_l,k2_l,N_graid);
                Normalization(psi_l);
                //wavfunc[l].push_back(dvec());
                if(e < 2)
                    fout.open("psi_"+cstr(l)+"_"+cstr(e)+".dat");
                for(int i = 1; i <= mesh; i++){
                    if(e < 2)
                        fout << i*dr << "\t" << psi_l[i]<< endl;
                    //wavfunc[l][e].push_back(psi_l[i]/(i*dr));
                    wavfunc[(l*len_E*N_graid)+(e*N_graid+(i-1))]=psi_l[i]/(i*dr);
                }
                if(e < 2)
                    fout.close();
            }
        }
    }
    delete[] vpot;delete[] f;delete[] y;
}

double tridTDSE_hybrid::seteigenvalues_numroveMethod(double eps, double deps, int l, int n_save){

    dvec Psi(n_save,0);
    dvec k2(n_save,0);
    double P1,P2;

    setk2(k2,eps,l,n_save);
    setwavefunction(Psi,k2,n_save);
    P1 = Psi[n_save-1];
    eps = eps + deps;

    while(abs(deps) > 1.0e-12){
        setk2(k2,eps,l,n_save);
        setwavefunction(Psi,k2,n_save);
        P2 = Psi[n_save-1];

        if(P1*P2 < 0.)
            deps = -deps/2.0;

        eps = eps + deps;
        P1 = P2;
    }
    return eps;
}

void tridTDSE_hybrid::setk2(dvec &k2, double eps, int l, int n_save){
    omp_set_num_threads(NTHREADS);
    #pragma omp parallel for default(shared) schedule(static, CHUNK)
    for(int i = 0; i < n_save; i++){
        double r = 1.0e-25 + i*dr;
        k2[i] = 2.0*(eps - potential(r,l));
    }
}

void tridTDSE_hybrid::setwavefunction(dvec &Psi, dvec &k2, int n_save){

    Psi[0] = 0.0;
    Psi[1] = 1.0e-7;

    for(int i = 2; i < n_save; i++){
        Psi[i] = (2.*(1.0-(5.0/12.)*dr*dr *k2[i-1])*Psi[i-1]
                  -(1.+(1./12.)*dr*dr *k2[i-2])*Psi[i-2])/(1.+(1.0/12.)*dr*dr *k2[i]);
    }
}

void tridTDSE_hybrid::Normalization(dvec &Psi){

    double sum = 0.0;
    for(int i = 0 ; i < Psi.size(); i++)
            sum += Psi[i]*Psi[i];
    sum=sqrt(sum*dr);
    for(int i = 0 ; i < Psi.size() ; i++){
            Psi[i]=Psi[i]/sum;
    }
}

double tridTDSE_hybrid::potential(double r, int l){

    double alp = 2.1325;
    return (l*(l+1.0))/(2.0*r*r)-(1.00+exp(-alp*r))/r;
    
}

template <class T>
string tridTDSE_hybrid::cstr(T value){
    ostringstream out;
    out << value;
    return out.str();
}

int main(){
    
    //MPI::Init();
    tridTDSE_hybrid tri_dtdse_hybrid;
    tri_dtdse_hybrid.tri_dtindse_hybrid();
    
    return 0;
}
