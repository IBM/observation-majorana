# Framework to run experiments, load/save/plot data

import numpy as np
import matplotlib.pyplot as plt
from qiskit import IBMQ
import pickle
import os
from matplotlib.colors import LogNorm
from ExperimentCode import *

class MajoranaSimulation:

    def __init__(self,api_key, hub, group, project, frequency='MZM', theta=np.pi/4, phi=np.pi/8, eta=0, alpha = None, n_qubits=5, n_cycles=10, n_shots = 8192, n_runs = 1, experiment='wavefunction',backend='qasm_simulator',backend_to_simulate = None, noise_type = None, layout = None):
        # parameters to set for experiments
        self.frequency = frequency
        self.theta = theta
        self.phi = phi
        self.eta = eta
        self.alpha = alpha
        self.n_qubits = n_qubits
        self.n_cycles = n_cycles
        self.n_shots = n_shots
        self.n_runs = n_runs
        self.experiment = experiment
        self.backend = backend
        self.backend_to_simulate = backend_to_simulate
        self.noise_type = noise_type
        self.layout = layout

        # objects that data is loaded into to store/make plots
        self.results = None
        self.params = None
        self.full_data = None
        
        # login information for IBMQ
        self.api_key = api_key
        self.hub = hub
        self.group = group
        self.project = project
    
    def load_account(self,api_key=None,reload = False):
        """ Loads user's IBMQ account.
            Inputs:
                api_key (str): your api key.
                reload (bool): if False, loads the current account. 
                               if True, overrides the current account and loads the new account from api_key.
        """
        #if it is the user's first time using this package OR if they wish to load a new account, they should set reload to TRUE; else, set it to FALSE
        if api_key == None: 
            if self.api_key == None:
                raise KeyError("No Qiskit API key specified. Please input a valid API key. See the following link for more details: https://quantum-computing.ibm.com/lab/docs/iql/manage/account/ibmq")
            else:
                api_key = self.api_key
        if reload == True:
            IBMQ.save_account(api_key,overwrite=True)
        else:
            IBMQ.load_account()
        

    def execute(self):
        """ Runs an experiment and stores the results locally 
            in the self.results, self.params, and self.full_data objects.
        """
        if self.backend.lower() != 'qasm_simulator' and self.layout == None:
            raise Exception('User must specify a qubit layout to run experiment on.')
        if self.experiment.lower() == 'braiding':
            execute_function = self.braiding_function
        elif self.experiment.lower() == 'wavefunction':
            execute_function = self.wavefunction_function
        elif self.experiment.lower() == 'two_point_correlation':
            execute_function = self.two_point_correlation_function
        else:
            raise Exception('Experiment type must be one of the following: "braiding", "wavefunction", "two_point_correlation". Input {} not understood'.format(self.experiment))
        result, params, full_data = execute_function()
        self.results = result
        self.params = params
        self.full_data = full_data

    def braiding_function(self):
        """ Implements braiding experiment, a noiseless theory simulation of the same data,
            does | \psi | ^2 = 1 normalization for the wavefunctions and the errors, and returns
            Python dictionaries with results, parameters, and the raw data.
            Outputs:
                results_dict (dict): a dictionary with the numerical results
                params_dict (dict): a dictionary with the experimental parameters
                full_data_dict (dict): a dictionary with the raw data
        
        """
        # load params from MajoranaCircuit class

        mc,omega = self.init_mc()
        n_runs = self.n_runs
        alpha = self.alpha
        fourier_end = 0

        ############################
        # Run braiding experiment  #
        ############################
        n_meas_gates_array = np.arange(1,mc.n_qubits+1)
        left_ZY = np.zeros([mc.n_qubits,n_runs])
        left_ZX = np.zeros([mc.n_qubits,n_runs])
        right_ZY = np.zeros([mc.n_qubits,n_runs])
        right_ZX = np.zeros([mc.n_qubits,n_runs])
        full_expval_data = np.zeros([2,n_runs], dtype = 'object')
        full_unavgd_expval_data = np.zeros([2,n_runs], dtype = 'object')
        for index in range(n_runs):
            direc = ['left','right']
            types = ['ZY','ZX']
            mc.meas_qubit = direc
            mc.meas_type = types
            final_meas_group, final_unavgd_group, braided_circuit_list = mc.braid_majoranas(n_meas_gates_array,braiding=True, alpha_val = alpha)
            full_expval_data[0,index] = final_meas_group
            full_unavgd_expval_data[0,index] = final_unavgd_group

            asd = 0
            for qub in direc:
                for typ in types:
                    filename = 'braided_{}_{}.csv'.format(qub,typ)
                    mc.meas_qubit = qub
                    mc.meas_type = typ
                    final_array = []
                    for n in range(len(n_meas_gates_array)):
                        n_gates = n_meas_gates_array[n]
                        final_result = 0
                        final_meas_array = final_meas_group[:,n,asd]
                        final_result = mc.fourier_transform(final_meas_array,omega,mc.n_cycles_total+1, end=fourier_end)
                        final_array.append(final_result)
                    if asd == 0:
                        left_ZY[:,index] = np.real(final_array)
                    elif asd == 1:
                        left_ZX[:,index] = np.real(final_array)
                    elif asd == 2:
                        right_ZY[:,index] = np.real(final_array)
                    elif asd == 3:
                        right_ZX[:,index] = np.real(final_array)
                    asd += 1

        
        majoranaL_zx = np.array(np.average(left_ZX,axis=1))
        majoranaL_zy = np.array(np.average(left_ZY,axis=1))
        majoranaR_zx = -1*np.array(np.average(right_ZX,axis=1)[::-1])
        majoranaR_zy = np.array(np.average(right_ZY,axis=1)[::-1])

        errorL_zx = np.array(np.std(left_ZX,axis=1))
        errorL_zy = np.array(np.std(left_ZY,axis=1))
        errorR_zx = np.array(np.std(right_ZX,axis=1)[::-1])
        errorR_zy = np.array(np.std(right_ZY,axis=1)[::-1])

        left_ZY = np.zeros([mc.n_qubits,n_runs])
        left_ZX = np.zeros([mc.n_qubits,n_runs])
        right_ZY = np.zeros([mc.n_qubits,n_runs])
        right_ZX = np.zeros([mc.n_qubits,n_runs])

        mc.meas_qubit = direc
        mc.meas_type = types
        final_meas_group, final_unavgd_group, unbraided_circuit_list = mc.braid_majoranas(n_meas_gates_array,braiding=False, alpha_val = alpha)

        for index in range(n_runs):
            asd = 0
            for qub in direc:
                for typ in types:
                    mc.meas_qubit = qub
                    mc.meas_type = typ
                    final_array = []
                    for n in range(len(n_meas_gates_array)):
                        n_gates = n_meas_gates_array[n]
                        final_result = 0
                        final_meas_array = final_meas_group[:,n,asd]
                        final_result = mc.fourier_transform(final_meas_array,omega,mc.n_cycles_total+1, end=fourier_end)
                        final_array.append(final_result)
                    if asd == 0:
                        left_ZY[:,index] = np.real(final_array)
                    elif asd == 1:
                        left_ZX[:,index] = np.real(final_array)
                    elif asd == 2:
                        right_ZY[:,index] = np.real(final_array)
                    elif asd == 3:
                        right_ZX[:,index] = np.real(final_array)
                    asd += 1

        umajoranaL_zx = np.array(np.average(left_ZX,axis=1))
        umajoranaL_zy = np.array(np.average(left_ZY,axis=1))
        umajoranaR_zx = -1*np.array(np.average(right_ZX,axis=1)[::-1])
        umajoranaR_zy = np.array(np.average(right_ZY,axis=1)[::-1])

        uerrorL_zx = np.array(np.std(left_ZX,axis=1))
        uerrorL_zy = np.array(np.std(left_ZY,axis=1))
        uerrorR_zx = np.array(np.std(right_ZX,axis=1)[::-1])
        uerrorR_zy = np.array(np.std(right_ZY,axis=1)[::-1])

        #####################
        # THEORY COMPARISON #
        #####################
        
        mc.run_type = 'simulator'
        mc.noise_type = None

        ####################################
        n_meas_gates_array = np.arange(1,mc.n_qubits+1)
        left_ZY = []
        left_ZX = []
        right_ZY = []
        right_ZX = []
        direc = ['left','right']
        types = ['ZY','ZX']

        mc.meas_qubit = direc
        mc.meas_type = types

        final_meas_group, final_unavgd_group, braided_circuit_list = mc.braid_majoranas(n_meas_gates_array,braiding=True, alpha_val = alpha)

        asd = 0
        for qub in direc:
            for typ in types:
                mc.meas_qubit = qub
                mc.meas_type = typ
                final_array = []
                for n in range(len(n_meas_gates_array)):
                    n_gates = n_meas_gates_array[n]
                    final_result = 0
                    final_meas_array = final_meas_group[:,n,asd]
                    final_result = mc.fourier_transform(final_meas_array,omega,mc.n_cycles_total+1, end=fourier_end)
                    final_array.append(final_result)
                if asd == 0:
                    left_ZY = final_array
                elif asd == 1:
                    left_ZX = final_array
                elif asd == 2:
                    right_ZY = final_array
                elif asd == 3:
                    right_ZX = final_array
                asd += 1
                
        theory_majoranaL_zx = np.array(left_ZX)
        theory_majoranaL_zy = np.array(left_ZY)
        theory_majoranaR_zx = -1*np.array(right_ZX[::-1])
        theory_majoranaR_zy = np.array(right_ZY[::-1])

        mc.meas_qubit = direc
        mc.meas_type = types
        final_meas_group, final_unavgd_group, unbraided_circuit_list = mc.braid_majoranas(n_meas_gates_array,braiding=False, alpha_val = alpha)

        asd = 0
        for qub in direc:
            for typ in types:
                mc.meas_qubit = qub
                mc.meas_type = typ
                final_array = []
                for n in range(len(n_meas_gates_array)):
                    n_gates = n_meas_gates_array[n]
                    final_result = 0
                    final_meas_array = final_meas_group[:,n,asd]
                    final_result = mc.fourier_transform(final_meas_array,omega,mc.n_cycles_total+1, end=fourier_end)
                    final_array.append(final_result)
                if asd == 0:
                    left_ZY = final_array
                elif asd == 1:
                    left_ZX = final_array
                elif asd == 2:
                    right_ZY = final_array
                elif asd == 3:
                    right_ZX = final_array
                asd += 1

        theory_umajoranaL_zx = np.array(left_ZX)
        theory_umajoranaL_zy = np.array(left_ZY)
        theory_umajoranaR_zx = -1*np.array(right_ZX[::-1])
        theory_umajoranaR_zy = np.array(right_ZY[::-1])

        left_majorana_expt = np.real(np.append(umajoranaL_zx,umajoranaL_zy))
        left_majorana_error = np.real(np.append(uerrorL_zx, uerrorL_zy))
        left_bmajorana_expt = np.real(np.append(majoranaL_zx,majoranaL_zy))
        left_bmajorana_error = np.real(np.append(errorL_zx, errorL_zy))
        right_majorana_expt = np.real(np.append(umajoranaR_zx,umajoranaR_zy))
        right_majorana_error = np.real(np.append(uerrorR_zx, uerrorR_zy))
        right_bmajorana_expt = np.real(np.append(majoranaR_zx, majoranaR_zy))
        right_bmajorana_error = np.real(np.append(errorR_zx, errorR_zy))

        left_majorana_theory = np.real(np.append(theory_umajoranaL_zx,theory_umajoranaL_zy))
        left_bmajorana_theory = np.real(np.append(theory_majoranaL_zx,theory_majoranaL_zy))
        right_majorana_theory = np.real(np.append(theory_umajoranaR_zx,theory_umajoranaR_zy))
        right_bmajorana_theory = np.real(np.append(theory_majoranaR_zx,theory_majoranaR_zy))

        # normalization of expt data by |\psi|^2 = 1
        psi_2 = 0
        for comp in left_majorana_expt:
            psi_2 += comp**2
        norm_leftm = 1/np.sqrt(psi_2)
        psi_2 = 0
        for comp in right_majorana_expt:
            psi_2 += comp**2
        norm_rightm = 1/np.sqrt(psi_2)
        psi_2 = 0
        for comp in left_bmajorana_expt:
            psi_2 += comp**2
        norm_leftb = 1/np.sqrt(psi_2)
        psi_2 = 0
        for comp in right_bmajorana_expt:
            psi_2 += comp**2
        norm_rightb = 1/np.sqrt(psi_2)

        left_majorana_normed = norm_leftm*left_majorana_expt
        left_uerror_normed = norm_leftm*left_majorana_error
        left_bmajorana_normed = norm_leftb*left_bmajorana_expt
        left_error_normed = norm_leftm*left_bmajorana_error
        right_majorana_normed = norm_rightm*right_majorana_expt
        right_uerror_normed = norm_rightm*right_majorana_error
        right_bmajorana_normed = norm_rightb*right_bmajorana_expt
        right_error_normed = norm_rightm*right_bmajorana_error
        
        # normalization of theory data by |\psi|^2 = 1
        psi_2 = 0
        for comp in left_majorana_theory:
            psi_2 += comp**2
        tnorm_leftm = 1/np.sqrt(psi_2)
        psi_2 = 0
        for comp in right_majorana_theory:
            psi_2 += comp**2
        tnorm_rightm = 1/np.sqrt(psi_2)
        psi_2 = 0
        for comp in left_bmajorana_theory:
            psi_2 += comp**2
        tnorm_leftb = 1/np.sqrt(psi_2)
        psi_2 = 0
        for comp in right_bmajorana_theory:
            psi_2 += comp**2
        tnorm_rightb = 1/np.sqrt(psi_2)

        left_majorana_tnormed = tnorm_leftm*left_majorana_theory
        left_bmajorana_tnormed = tnorm_leftb*left_bmajorana_theory
        right_majorana_tnormed = tnorm_rightm*right_majorana_theory
        right_bmajorana_tnormed = tnorm_rightb*right_bmajorana_theory

        # saving data
        if mc.eta == None:
            eta_fraction = 0
        else:
            eta_fraction = mc.eta/np.pi
        if alpha == None:
            alpha_fraction = 0
        else:
            alpha_fraction = alpha/np.pi
        if fourier_end == 0:
            n_cycles = mc.n_cycles_total
        else:
            n_cycles = fourier_end

        params_dict = {'alpha': alpha_fraction, 'regime': self.frequency, 'num_qubits': mc.n_qubits, 'num gate cycles': n_cycles, 'num_shots': mc.n_avgs, 'num_runs': n_runs, 'device': self.backend, 'backend_to_simulate':self.backend_to_simulate, 'noise_type': self.noise_type, 'layout':self.layout, 'theta': mc.theta/np.pi, 'phi': mc.phi/np.pi, 'eta': eta_fraction, 'experiment': self.experiment}
        results_dict = {'x_axis':range(mc.n_qubits), 
                        'left ZX unbraided (theory)': left_majorana_tnormed[:mc.n_qubits],
                        'left ZY unbraided (theory)': left_majorana_tnormed[mc.n_qubits:], 
                        'right ZX unbraided (theory)': right_majorana_tnormed[:mc.n_qubits],
                        'right ZY unbraided (theory)': right_majorana_tnormed[mc.n_qubits:], 
                        'left ZX braided (theory)': left_bmajorana_tnormed[:mc.n_qubits],
                        'left ZY braided (theory)': left_bmajorana_tnormed[mc.n_qubits:],
                        'right ZX braided (theory)': right_bmajorana_tnormed[:mc.n_qubits],
                        'right ZY braided (theory)': right_bmajorana_tnormed[mc.n_qubits:],
                        
                        'left ZX unbraided (expt)': left_majorana_normed[:mc.n_qubits],
                        'left ZY unbraided (expt)': left_majorana_normed[mc.n_qubits:], 
                        'right ZX unbraided (expt)': right_majorana_normed[:mc.n_qubits],
                        'right ZY unbraided (expt)': right_majorana_normed[mc.n_qubits:], 
                        'left ZX unbraided std (expt)': left_uerror_normed[:mc.n_qubits],
                        'left ZY unbraided std (expt)': left_uerror_normed[mc.n_qubits:],
                        'right ZX unbraided std (expt)': right_uerror_normed[:mc.n_qubits],
                        'right ZY unbraided std (expt)': right_uerror_normed[mc.n_qubits:],
                        
                        'left ZX braided (expt)': left_bmajorana_normed[:mc.n_qubits],
                        'left ZY braided (expt)': left_bmajorana_normed[mc.n_qubits:],
                        'right ZX braided (expt)': right_bmajorana_normed[:mc.n_qubits],
                        'right ZY braided (expt)': right_bmajorana_normed[mc.n_qubits:],
                        'left ZX braided std (expt)': left_error_normed[:mc.n_qubits],
                        'left ZY braided std (expt)': left_error_normed[mc.n_qubits:],
                        'right ZX braided std (expt)': right_error_normed[:mc.n_qubits],
                        'right ZY braided std (expt)': right_error_normed[mc.n_qubits:]}
        full_data_dict = {'expval data': full_expval_data, 'raw data': full_unavgd_expval_data}
        return results_dict, params_dict, full_data_dict
    
    def save_braiding_data(self,directory):
        """ Saves data from braiding experiment to the folder specified with the `directory` parameter.
            Input:
                directory (str): path to the folder you wish to save the data in.
        
        """
        with open(directory + '/results.pkl', 'wb') as handle:
            pickle.dump(self.results, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(directory + '/params.pkl', 'wb') as handle:
            pickle.dump(self.params, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(directory + '/full_data.pkl', 'wb') as handle:
            pickle.dump(self.full_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('Braiding data saved successfully to {}'.format(directory))
    

    def plot_braiding_data(self):
        """ Plots braiding data currently in self.results.
        """
        # loading data from self.results
        col = ['b','r','g','m']
        fig,ax = plt.subplots(2,2,figsize = (5.5,5))
        ZX_left_theory_unbrd = self.results['left ZX unbraided (theory)']
        ZY_left_theory_unbrd = self.results['left ZY unbraided (theory)']
        ZX_right_theory_unbrd = self.results['right ZX unbraided (theory)']
        ZY_right_theory_unbrd = self.results['right ZY unbraided (theory)']

        ZX_left_expt_unbrd = self.results['left ZX unbraided (expt)']
        ZY_left_expt_unbrd = self.results['left ZY unbraided (expt)']
        ZX_right_expt_unbrd = self.results['right ZX unbraided (expt)']
        ZY_right_expt_unbrd = self.results['right ZY unbraided (expt)']

        ZX_left_theory_brd = self.results['left ZX braided (theory)']
        ZY_left_theory_brd = self.results['left ZY braided (theory)']
        ZX_right_theory_brd = self.results['right ZX braided (theory)']
        ZY_right_theory_brd = self.results['right ZY braided (theory)']

        ZX_left_expt_brd = self.results['left ZX braided (expt)']
        ZY_left_expt_brd = self.results['left ZY braided (expt)']
        ZX_right_expt_brd = self.results['right ZX braided (expt)']
        ZY_right_expt_brd = self.results['right ZY braided (expt)']

        ZX_left_expt_brd_std = self.results['left ZX braided std (expt)']
        ZY_left_expt_brd_std = self.results['left ZY braided std (expt)']
        ZX_right_expt_brd_std = self.results['right ZX braided std (expt)']
        ZY_right_expt_brd_std = self.results['right ZY braided std (expt)']

        ZX_left_expt_unbrd_std = self.results['left ZX unbraided std (expt)']
        ZY_left_expt_unbrd_std = self.results['left ZY unbraided std (expt)']
        ZX_right_expt_unbrd_std = self.results['right ZX unbraided std (expt)']
        ZY_right_expt_unbrd_std = self.results['right ZY unbraided std (expt)']

        x = np.array(self.results['x_axis'])+1

        # plotting data
        ax[0,0].plot(x,ZX_left_theory_unbrd,c=col[0],ls='-',label = 'left odd')
        ax[0,0].plot(x,ZY_left_theory_unbrd,c=col[1],ls='--',label = 'left even')
        ax[1,0].plot(x,ZX_right_theory_unbrd,c=col[2],ls='-',label = 'right odd')
        ax[1,0].plot(x,ZY_right_theory_unbrd,c=col[3],ls='--',label = 'right even')

        ax[0,0].set_xticks(x)
        ax[0,1].set_xticks(x)
        ax[1,0].set_xticks(x)
        ax[1,1].set_xticks(x)

        ax[0,0].errorbar(x,ZX_left_expt_unbrd,yerr = ZX_left_expt_unbrd_std,capsize=4.0,
                        marker = 's',ms=3,lw = 0,elinewidth=1,c=col[0])#,alpha=0.5)
        ax[0,0].errorbar(x,ZY_left_expt_unbrd,yerr = ZY_left_expt_unbrd_std,capsize=4.0,
                        marker = 'o',ms=4,lw = 0,elinewidth=1,c=col[1])#,alpha=0.5)
        ax[1,0].errorbar(x,ZX_right_expt_unbrd,yerr = ZX_right_expt_unbrd_std,capsize=4.0,
                        marker = 's',ms=3,lw = 0,elinewidth=1,c=col[2])
        ax[1,0].errorbar(x,ZY_right_expt_unbrd,yerr = ZY_right_expt_unbrd_std,capsize=4.0,
                        marker = 'o',ms=3,lw = 0,elinewidth=1,c=col[3])

        ax[0,1].plot(x,ZX_left_theory_brd,c=col[1],ls='--',label = 'right odd')
        ax[0,1].plot(x,ZY_left_theory_brd,c=col[0],ls='-',label = 'right odd')
        ax[1,1].plot(x,ZX_right_theory_brd,c=col[3],ls='--',label = 'right odd')
        ax[1,1].plot(x,ZY_right_theory_brd,c=col[2],ls='-',label = 'right odd')

        ax[0,1].errorbar(x,ZX_left_expt_brd,yerr = ZX_left_expt_brd_std,capsize=4.0,
                        marker = 's',ms=3,lw = 0,elinewidth=1,c=col[1])#,alpha=0.5)
        ax[0,1].errorbar(x,ZY_left_expt_brd,yerr = ZY_left_expt_brd_std,capsize=4.0,
                        marker = 'o',ms=4,lw = 0,elinewidth=1,c=col[0])#,alpha=0.5)
        ax[1,1].errorbar(x,ZX_right_expt_brd,yerr = ZX_right_expt_brd_std,capsize=4.0,
                        marker = 's',ms=3,lw = 0,elinewidth=1,c=col[3])
        ax[1,1].errorbar(x,ZY_right_expt_brd,yerr = ZY_right_expt_brd_std,capsize=4.0,
                        marker = 'o',ms=3,lw = 0,elinewidth=1,c=col[2])

        ax[0,0].legend()
        ax[1,0].legend()

        fig.tight_layout()

        
    def init_mc(self):
        """ Calls MajoranaCircuit code and sets parameters from current MajoranaSimulation class attributes
        """
        
        mc = MajoranaCircuit(self.api_key, self.hub,self.group,self.project)
        mc.n_qubits = self.n_qubits
        mc.n_cycles_total = self.n_cycles
        mc.n_avgs = self.n_shots
        frequency = self.frequency
        if self.frequency == 'MZM':
            omega = 0*np.pi
        elif self.frequency == 'MPM':
            omega = 1*np.pi
            
        if self.backend == 'qasm_simulator':
            mc.run_type = 'simulator'
            mc.noise_type = self.noise_type
            if self.noise_type == 'backend_model':
                mc.device = self.backend_to_simulate
            else:
                mc.device = self.backend
        else:
            mc.device = self.backend 
            mc.run_type = 'computer'
            mc.noise_type = None
        mc.layout = self.layout
        mc.theta = self.theta
        mc.phi = self.phi
        mc.eta = self.eta
        if mc.eta == 0:
            mc.zz_gates = False
        else:
            mc.zz_gates = True
        return mc, omega

    def wavefunction_function(self):
        """ Implements wavefunction experiment, a noiseless theory simulation for comparison,
            does | \psi | ^2 = 1 normalization for the wavefunctions and returns
            Python dictionaries with results, parameters, and the raw data.
            Outputs:
                results_dict (dict): a dictionary with the numerical results
                params_dict (dict): a dictionary with the experimental parameters
                full_data_dict (dict): a dictionary with the raw data
        
        """
        # calling MajoranaCircuit class
        mc,omega = self.init_mc()
        
        # running experiment
        o_array, error_array, all_meas_data, all_raw_data, circuit = mc.calculate_eigenmodes_no_save(omega = omega, load_from_file = False,numerical=False)
        
        # running theory simulations
        if mc.zz_gates == True:
            free_fermion = False
        elif mc.zz_gates == False:
            free_fermion = True
        eta = mc.eta
        phi = mc.phi 
        theta = mc.theta
        if eta == None:
            eta_frac = 0
        else:
            eta_frac = eta/np.pi
        phi_frac = phi/np.pi
        theta_frac = theta/np.pi
        N = mc.n_qubits
        cycles = mc.n_cycles_total+1 #must be even for MPM
        typ = self.frequency
        if free_fermion == False:
            zx_str, zy_str = show_mode(N,cycles,theta_frac,phi_frac,eta_frac,tp=typ)
        elif free_fermion == True:
            if typ == 'MZM':
                psiR,psiL = majorana_modes(N,phi=phi_frac*np.pi,theta=theta_frac*np.pi,mode = 'MZM')

            if typ == 'MPM':
                psiR,psiL = majorana_modes(N,phi=phi_frac*np.pi,theta=theta_frac*np.pi,mode = 'MPM')


        o_list = [o_array]
        error_list = [error_array]
        norm_list = np.zeros(len(o_list),dtype = 'object')
        enorm_list= np.zeros(len(error_list),dtype = 'object')
        
        o_arrayy = o_list[0]
        error_arrayy = error_list[0]
        norm_array = o_arrayy
        enorm = error_arrayy
        if free_fermion == True:
            t_array = np.column_stack([np.abs(psiR[1::2]), np.abs(psiR[::2]), np.abs(psiL[::2]),np.abs(psiL[1::2])]) #this goes in code calling the function
        else:
            t_array = np.column_stack([zx_str[::-1],zy_str[::-1],zx_str,zy_str])

        X = range(mc.n_qubits)
        theory_Lzx = t_array[:,2]
        theory_Lzy = t_array[:,3]
        theory_Rzx = t_array[:,0]
        theory_Rzy = t_array[:,1]
        exp_Lzx = norm_array[:,2]
        exp_Lzy = norm_array[:,3]
        exp_Rzx = norm_array[:,0] 
        exp_Rzy = norm_array[:,1]

        # normalizing wavefunctions
        right_norm = np.sqrt(np.sum(exp_Rzx**2+exp_Rzy**2))
        left_norm = np.sqrt(np.sum(exp_Lzx**2+exp_Lzy**2))

        right_norm_th = np.sqrt(np.sum(theory_Rzx**2+theory_Rzy**2))
        left_norm_th = np.sqrt(np.sum(theory_Lzx**2+theory_Lzy**2))
        
        if free_fermion == False:
            theory_source = 'show_mode'
        elif free_fermion == True:
            theory_source = 'majorana_modes'

        # saving data
        params_dict = {'theory': 'adapted from Oles "{}" code'.format(theory_source) , 'regime': self.frequency, 'num_qubits': mc.n_qubits, 'num gate cycles': mc.n_cycles_total, 'num_shots': mc.n_avgs, 'device': self.backend, 'backend_to_simulate':self.backend_to_simulate, 'noise_type': self.noise_type, 'layout':self.layout, 'theta': mc.theta/np.pi, 'phi': mc.phi/np.pi, 'eta': eta_frac, 'experiment': self.experiment}
        full_data_dict = {'full exp data': all_meas_data, 'full raw data': all_raw_data}
        results_dict = {'x_axis':range(mc.n_qubits),
                        'theory for left ZX':theory_Lzx/left_norm_th,'theory for left ZY':theory_Lzy/left_norm_th,
                        'theory for right ZX':theory_Rzx/right_norm_th,'theory for right ZY':theory_Rzy/right_norm_th,
                        'experiment for left ZX':exp_Lzx/left_norm,'experiment for left ZY':exp_Lzy/left_norm,
                        'experiment for right ZX':exp_Rzx/right_norm,'experiment for right ZY':exp_Rzy/right_norm}
        return results_dict, params_dict, full_data_dict
    

    def save_wavefunction_data(self, directory):
        """ Saves data from wavefunction experiment to the folder specified with 
        the `directory` parameter.
            Input:
                directory (str): path to the folder you wish to save the data in.
        
        """
        with open(directory + '/results.pkl', 'wb') as handle:
            pickle.dump(self.results, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(directory + '/params.pkl', 'wb') as handle:
            pickle.dump(self.params, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(directory + '/full_data.pkl', 'wb') as handle:
            pickle.dump(self.full_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('Wavefunction data saved successfully to {}'.format(directory))


    def plot_wavefunction_data(self):
        """ Plots wavefunction data currently in self.results.
        """
        # loading data
        n_qubits = self.params['num_qubits']
        x_axis = self.results['x_axis']
        n_cycles_total = self.params['num gate cycles']
        ZX_right_theory = self.results['theory for right ZX']
        ZX_left_theory = self.results['theory for left ZX']
        ZY_right_theory = self.results['theory for right ZY']
        ZY_left_theory = self.results['theory for left ZY']
        
        ZX_right_experiment = self.results['experiment for right ZX']
        ZX_left_experiment = self.results['experiment for left ZX']
        ZY_right_experiment = self.results['experiment for right ZY']
        ZY_left_experiment = self.results['experiment for left ZY']
        
        # creating plot
        fig,ax = plt.subplots(1,figsize = (4,3))
        ax.plot(x_axis,ZX_right_theory,c='b',ls='-',label = 'right odd')
        ax.plot(x_axis,ZX_right_experiment,marker = 'o',ms=8,lw = 0,c='b',markeredgecolor=None,alpha=0.5)

        ax.plot(x_axis,ZX_left_theory,c='r',ls='--',label = 'left odd')
        ax.plot(x_axis,ZX_left_experiment,marker = 's',ms=7,lw = 0,c='r',markeredgecolor=None,alpha=0.5)

        ax.plot(x_axis,ZY_right_theory,c='g',ls='-.',label = 'right even')
        ax.plot(x_axis,ZY_right_experiment,marker = '^',ms=8,lw = 0,c='g',markeredgecolor=None,alpha=0.5)

        ax.plot(x_axis,ZY_left_theory,c='m',ls=':',label = 'left even')
        ax.plot(x_axis,ZY_left_experiment,marker = 'D',ms=7,lw = 0,c='m',markeredgecolor=None,alpha=0.5)

        x_label_list = np.arange(1,n_qubits+1)
        y_label_list = np.round(np.arange(0,1.05,0.2),2)
        ax.set_xticks(np.arange(1,n_qubits+1)-1)
        ax.set_yticks(y_label_list)

        fs = 16
        fs2 = 14
        font = 'serif'
        ax.set_xticklabels(x_label_list,fontdict={'fontsize':fs2,'fontname':font})
        ax.set_yticklabels(y_label_list,fontdict={'fontsize':fs2,'fontname':font})
        plt.tight_layout()
        plt.show()

        fig1, ax1 = plt.subplots(1, figsize = (4,3))
        result_array_R = np.zeros([200,n_qubits])
        result_array_L = np.zeros([200,n_qubits])
        omega_array = np.linspace(-3*np.pi/2,3*np.pi/2, 200)
        gate_array = range(1,n_qubits+1)
        base_str = 'X'
        
        meas_data = self.full_data['full exp data']
        meas_data_R = meas_data[:,:,0]
        meas_data_L = meas_data[:,:,2]
        for i in range(len(gate_array)):
            z_str = ''
            for d in range(i):
                z_str += 'Z'
            measurement_gate = z_str + base_str
            final_meas_array_R = MajoranaCircuit.load_measurement(MajoranaCircuit,meas_data_R, measurement_gate)
            final_meas_array_L = MajoranaCircuit.load_measurement(MajoranaCircuit,meas_data_L, measurement_gate)
            for j in range(len(omega_array)):
                omega = omega_array[j]
                result_R = MajoranaCircuit.fourier_transform(MajoranaCircuit,final_meas_array_R,omega,n_cycles_total+1)
                result_array_R[j,i] = np.abs(result_R**2)
                result_L = MajoranaCircuit.fourier_transform(MajoranaCircuit,final_meas_array_L,omega,n_cycles_total+1)
                result_array_L[j,i] = np.abs(result_L**2)
        
        Xm,Ym = np.meshgrid(gate_array,omega_array)
        Zm = result_array_L + np.flip(result_array_R,axis=1)

        ax1.pcolor(Xm,Ym,Zm,shading='auto',norm=LogNorm(vmin=Zm.min(), vmax=Zm.max()))#,extent = [1, n_qubits, -3*np.pi/2, 3*np.pi/2],cmap = 'hot')
        y_label_list = ['-$\pi$', '0', '$\pi$']
        x_label_list = range(1,n_qubits+1)
        ax1.set_yticks([-np.pi, 0, np.pi])
        ax1.set_xticks(range(1,n_qubits+1))
        ax1.set_xticklabels(x_label_list,fontdict={'fontsize':fs2,'fontname':font})
        ax1.set_yticklabels(y_label_list,fontdict={'fontsize':fs2,'fontname':font})
        ax1.set_ylabel("Freq. ω",fontdict={'fontsize':fs,'fontname':font})
        ax1.set_xlabel(r'Position $x$',fontdict={'fontsize':fs,'fontname':font})
        plt.tight_layout()
        plt.show()
    

    def two_point_correlation_function(self, fourier_end = None):
        """ Implements two-point correlation experiment and returns
            Python dictionaries with results, parameters, and the raw data.
            Outputs:
                results_dict (dict): a dictionary with the numerical results
                params_dict (dict): a dictionary with the experimental parameters
                full_data_dict (dict): a dictionary with the raw data
        
        """
        # calling MajoranaCircuit code and setting initial parameters
        mc = MajoranaCircuit(self.api_key, self.hub,self.group,self.project)
        mc.n_qubits = self.n_qubits
        mc.n_cycles_total = self.n_cycles
        mc.n_avgs = self.n_shots
        frequency = self.frequency
        mc,omega = self.init_mc()
        if mc.eta != 0:
            print('Parameter $ \eta $ will be set to 0 for this experiment.')
            mc.eta = 0
        n_meas_gates_array = np.arange(1,mc.n_qubits+1)
        coup_phi = mc.phi
        coup_theta = mc.theta

        fourier_start = 0
        n_runs = self.n_runs 

        mc.meas_qubit = 'left'
        mc.meas_type = 'ZX' 
        phi_edge = 0
        theta_edge = 0
        if fourier_end == None:
            n_cycles = mc.n_cycles_total
        else:
            n_cycles = fourier_end

        # running circuits and saving results
        decop_array = np.zeros([mc.n_qubits,n_runs])
        cop_array = np.zeros([mc.n_qubits,n_runs])
        circ_array = np.zeros([1,n_runs], dtype = 'object')
        full_expval_data = np.zeros([2,n_runs], dtype = 'object')
        full_unavgd_expval_data = np.zeros([2,n_runs], dtype = 'object')
        for avg in range(n_runs):

            for decoupled in [True,False]:
                if decoupled == True:
                    mc.phi = coup_theta
                    mc.theta = coup_phi
                elif decoupled == False:
                    mc.theta = coup_theta
                    mc.phi = coup_phi
                if decoupled == True:
                    final_meas_group, final_unavgd_group, d_circuit_list = mc.run_decoupled_circuit(n_meas_gates_array,decoupled=decoupled,phi_red=phi_edge,theta_red=theta_edge, man_avg = False, unitary_approach = True)
                    circ_array[0,avg] = d_circuit_list[0][0]
                elif decoupled == False:
                    final_meas_group,final_unavgd_group, _ = mc.run_decoupled_circuit(n_meas_gates_array,decoupled=decoupled,phi_red=phi_edge,theta_red=theta_edge, man_avg = False, unitary_approach = True)
                full_expval_data[int(decoupled),avg] = final_meas_group
                full_unavgd_expval_data[int(decoupled), avg] = final_unavgd_group

                for n in range(len(n_meas_gates_array)):
                    n_gates = n_meas_gates_array[n]
                    final_result = 0
                    final_meas_array = final_meas_group[:,n]
                    final_result = np.abs(mc.fourier_transform(final_meas_array,omega,n_cycles+1,start=fourier_start,end=n_cycles))
 
                    if decoupled == True:
                        decop_array[n,avg] = final_result
                    elif decoupled == False:
                        cop_array[n,avg] = final_result

        # calculating average results and std
        avg_decop_array = np.array(np.average(decop_array,axis=1))
        avg_cop_array = np.array(np.average(cop_array,axis=1))
        error_decop = np.array(np.std(decop_array,axis=1))
        error_cop = np.array(np.std(cop_array,axis=1))

        # saving to file
        params_dict = {'type': '{} {}'.format(mc.meas_qubit, mc.meas_type), 'regime': self.frequency, 'num_qubits': mc.n_qubits, 'num gate cycles': n_cycles, 'num_shots': mc.n_avgs, 'num_runs': self.n_runs, 'device': self.backend, 'backend_to_simulate':self.backend_to_simulate, 'noise_type': self.noise_type, 'layout':self.layout, 'theta': mc.theta/np.pi, 'phi': mc.phi/np.pi, 'phi edge':phi_edge, 'theta_edge':theta_edge, 'eta': mc.eta, 'init edge state': '2|0> + 3i|1>', 'experiment': self.experiment}
        results_dict = {'x_axis':range(mc.n_qubits),
                    'decoupled edge fermion':avg_decop_array,'coupled majorana mode':avg_cop_array,
                    'decoupled errorbars':error_decop,'coupled errorbars':error_cop}
        full_data_dict = {'expval data': full_expval_data, 'raw data': full_unavgd_expval_data}
        return results_dict, params_dict, full_data_dict
    
    def save_two_point_correlation_data(self,directory):
        """ Saves data from two-point correlation experiment to the folder 
            specified with the `directory` parameter.
            Input:
                directory (str): path to the folder you wish to save the data in.
        
        """
        with open(directory + '/results.pkl', 'wb') as handle:
            pickle.dump(self.results, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(directory + '/params.pkl', 'wb') as handle:
            pickle.dump(self.params, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(directory + '/full_data.pkl', 'wb') as handle:
            pickle.dump(self.full_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('Two-point correlation data saved successfully to {}'.format(directory))


    def plot_two_point_correlation_data(self):
        """ Plots two-point correlation data currently in self.results.
        """
        n_qubits = self.params['num_qubits']
        avg_decop_array = self.results['decoupled edge fermion']
        avg_cop_array = self.results['coupled majorana mode']
        error_decop = self.results['decoupled errorbars']
        error_cop = self.results['coupled errorbars']

        fig,ax = plt.subplots(figsize = (4,4))
        x_label_list = np.arange(1,n_qubits+1,2)+1
        vmax = max(np.max(avg_decop_array),np.max(avg_cop_array))
        ax.errorbar(range(n_qubits),avg_decop_array,yerr = error_decop, marker = 'o',ms=8,lw = 1,c='b',capsize=4.0, label='Decoupled Edge Fermion')
        ax.errorbar(range(n_qubits),avg_cop_array,yerr = error_cop,marker = 's',ms=7,lw = 1,c='r',capsize=4.0, label='Coupled Majorana Mode')
        ax.set_xticks(np.arange(1,n_qubits+1,2))
        ax.set_xticklabels(x_label_list)
        ax.set_xlabel(r'Position $x$')
        ax.set_ylabel(r'Two-point function |$T_{1,2x}$|')
        ax.legend()

    def load_data(self, directory):
        """ Loads data saved in folder specified by `directory` to
            self.results, self.params, and self.full_data, while
            setting self.experiment to correct experiment type.
            Input:
                directory (str): path to folder containing datafiles.
        """
        with open(directory + '/results.pkl', 'rb') as handle:
            results_dict = pickle.load(handle)
        with open(directory + '/params.pkl', 'rb') as handle:
            params_dict = pickle.load(handle)
        with open(directory + '/full_data.pkl', 'rb') as handle:
            data_dict = pickle.load(handle)
        self.results = results_dict
        self.params = params_dict
        self.full_data = data_dict
        self.experiment = self.params['experiment']

        print('{} data loaded successfully from {}'.format(self.experiment, directory))

    def save_data(self, directory):
        """ Saves data loaded in self.results, self.params, and self.full_data
            to folder specified by `directory`
            Input:
                directory (str): path to folder where datafiles will be saved.
        """
        #first, making sure there indeed are results available to save
        try:
            if self.results == None:
                raise TypeError('No results available to save.')
        except ValueError:
            if self.results.any() == None:
                raise TypeError('No results available to save.')
        
        #next, saving the data based on the experiment
        if self.experiment.lower() == 'braiding':
            save_function = self.save_braiding_data
        elif self.experiment.lower() == 'wavefunction':
            save_function = self.save_wavefunction_data
        elif self.experiment.lower() == 'two_point_correlation':
            save_function = self.save_two_point_correlation_data
        isDIR = os.path.isdir(directory)
        if isDIR == True:
            pass
        else:
            os.mkdir(directory)
        save_function(directory)
        
    def plot_data(self):
        """ Plots data currently in self.results.
        """
        #first, making sure there indeed are results available to plot
        try:
            if self.results == None:
                raise TypeError('No data available to plot.')
        except ValueError:
            if self.results.any() == None:
                raise TypeError('No data available to plot.')
        
        #next, plotting the data based on the experiment
        if self.experiment.lower() == 'braiding':
            plot_function = self.plot_braiding_data
        elif self.experiment.lower() == 'wavefunction':
            plot_function = self.plot_wavefunction_data
        elif self.experiment.lower() == 'two_point_correlation':
            plot_function = self.plot_two_point_correlation_data
        elif self.experiment.lower() == 'expval':
            plot_function = self.plot_expval_data
        elif self.experiment.lower() == 'fourier':
            plot_function = self.plot_fourier_data
        plot_function()

    def plot_fourier_data(self):
        """ Plotting Fourier component data from Fig. 1c of the paper.
        """
        # calling plot objects and setting parameters
        fig = plt.figure()
        ax = plt.figure().add_subplot(projection='3d')
        spacing = 0.02
        offset = 0.1

        # calling loaded data
        phi_array = self.results['phi_array']
        omegas = self.results['frequency_array']
        results = self.results['fourier_component']

        cols = np.linspace(0,1,len(phi_array))
        for i in np.flip(range(len(phi_array))):
            phi = phi_array[i]
                
            z = results[i]
            x = omegas
            y = np.ones(len(x))*phi

            # making plot
            ax.plot(x/np.pi, y/np.pi, z,c=(cols[i],0,1-cols[i]))
            ax.set_zlim(0,0.5)
            spacing += offset

        # ax.set_xticks([-1.5,-1,0.5,0,-0.5,1,1.5])
        # ax.set_yticks([0,0.125,0.25,0.375,0.5])
        ax.set_xlabel(r'Frequency $\omega$')
        ax.set_ylabel(r'Z-gate angle $\phi$')
        ax.set_xticks([-1,-0.5,0,0.5,1])
        ax.set_xticklabels([r'-$\pi$',r'-$\pi$/2','0',r'$\pi$/2',r'$\pi$'])
        ax.set_yticks([0,0.125,0.25,0.375,0.5])
        ax.set_yticklabels(['0','',r'$\pi$/4','',r'$\pi$/2'])
            
        plt.show()

    def plot_expval_data(self):
        """ Plotting expectation value data from Fig. 5 of the paper.
        """
        strings = ['X', 'ZZX']
        fig, ax = plt.subplots(len(strings),1, figsize = (6,len(strings)*4),sharex=True)
        for i, strg in enumerate(strings):
            # load data
            gamma = self.params['gamma']
            n_cycles_total = self.params['n_cycles']
            dataname = self.params['regime']
            exp_paulis = self.results['data ({})'.format(strg)]
            exp_paulis_f = self.results['normalized data ({})'.format(strg)]
            theory_paulis = self.results['noiseless sim ({})'.format(strg)]
            
            # plot data
            ax[i].plot(range(n_cycles_total+1), exp_paulis, marker = 'o',label='data', c = 'C0')
            ax[i].plot(range(n_cycles_total+1), exp_paulis_f, marker = 's',label='normalized data', c = 'C1')
            ax[i].plot(range(n_cycles_total+1), theory_paulis, label = 'noiseless sim', c = 'k', ls = '--')

            # specify matplotlib parameters
            ax[0].legend()
            if dataname != 'MPM':
                ax[0].set_ylim([-0.05,1.05])
                ax[1].set_ylim([-.75,.75])
            ax[i].set_xlabel('Floquet cycles')
            ax[i].set_ylabel(r'Observable $ \langle$' + strg + r'$\rangle $')
        ax[0].set_xticks([])
        ax[0].set_xticks([], minor=True)
        ax[1].set_xticks(range(0,n_cycles_total+1, 2))
        title = '10q {}'.format(dataname)
        ax[0].set_title(title)

        plt.tight_layout()