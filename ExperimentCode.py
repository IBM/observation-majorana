# Experimental code

import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import pandas as pd
from scipy.optimize import leastsq
from scipy.linalg import expm
import time
import math
import random
from TheoryCode import *
from qiskit import QuantumCircuit, transpile
from qiskit import IBMQ, Aer, execute
from qiskit.providers.ibmq import RunnerResult
from qiskit.tools.monitor import job_monitor
from qiskit.providers.aer import noise
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise import thermal_relaxation_error


#building subcircuit to measure in ZZ...X basis
sub_circ = QuantumCircuit(2, name='ZX meas')
sub_circ.rz(-np.pi/2,0)
sub_circ.sx(1)
sub_circ.sx(0)
sub_circ.rz(-0.76262,1)
sub_circ.rz(-3*np.pi/4,0)
sub_circ.sx(1)
sub_circ.rz(-np.pi/2,1)
sub_circ.cx(0,1)
sub_circ.rz(np.pi/4,0)
sub_circ.rz(np.pi/2,1)
sub_circ.sx(1)
sub_circ.sx(0)
sub_circ.rz(-np.pi/2,0)
sub_circ.rz(0.80818,1)
sub_circ.sx(1)

measurement_gate = sub_circ.to_instruction()

#building subcircuit to measure in ZZ...Y basis

sub_circ1 = QuantumCircuit(2, name='ZY meas')

sub_circ1.sx(0)
sub_circ1.rz(np.pi/2,1)
sub_circ1.sx(1)
sub_circ1.rz(5.52057014286000,1)
sub_circ1.sx(1)
sub_circ1.rz(7*np.pi/2,1)
sub_circ1.cx(0,1)
sub_circ1.rz(-np.pi/2,0)
sub_circ1.rz(-np.pi/2,1)
sub_circ1.sx(0)
sub_circ1.sx(1)
sub_circ1.rz(-np.pi,0)
sub_circ1.rz(5.47500414470428,1)
sub_circ1.sx(1)
sub_circ1.rz(5*np.pi/2,1)

measurement_gate1 = sub_circ1.to_instruction()

test = QuantumCircuit(1,name='Initialize')
test.rz(np.pi/2,0)
test.sx(0)
test.rz(4.06888787159141,0)
test.sx(0)
test.rz(7*np.pi/2,0)
init_gate = test.to_instruction()

init_new = QuantumCircuit(1,name='Initialize new')
init_new.sx(0)
init_new.rz(-1.176,0)
init_new.sx(0)
init_new.rz(-np.pi/2,0)
init_gate_new = init_new.to_instruction()

X = np.array([[0,1],[1,0]])
Y = np.array([[0,-1j],[1j,0]])
uxy = expm(1j*np.pi/4*np.kron(X,Y))
circuits = QuantumCircuit(2,2)
circuits.unitary(uxy, [0,1])
basis_gates = ['sx','rz','x','id','cx','reset']
trans_qc = transpile(circuits, basis_gates = basis_gates)
trans_qc.name = 'U_xy gate'
uxy_circ = QuantumCircuit(2)
uxy_circ.sx(0)
uxy_circ.rz(np.pi/2,1)
uxy_circ.sx(1)
uxy_circ.rz(5*np.pi/4,1)
uxy_circ.sx(1)
uxy_circ.rz(7*np.pi/2,1)
uxy_circ.cx(0,1)
uxy_circ.rz(np.pi/2,1)
uxy_circ.rz(-np.pi/2,0)
uxy_circ.sx(0)
uxy_circ.sx(1)
uxy_circ.rz(-np.pi,0)
uxy_circ.rz(7*np.pi/4,1)
uxy_circ.sx(1)
uxy_circ.rz(7*np.pi/2,1)
uxy_circ.name = 'U_xy gate'
uxy_gate = uxy_circ.to_instruction()

uxy_dg = uxy.T.conj()
circuits1 = QuantumCircuit(2,2)
circuits1.unitary(uxy_dg, [0,1])
dg_trans_qc = transpile(circuits1, basis_gates = basis_gates)
dg_trans_qc.name = 'U_xy Conj'
dg_uxy_circ = QuantumCircuit(2)
dg_uxy_circ.rz(-np.pi,0)
dg_uxy_circ.rz(np.pi/2,1)
dg_uxy_circ.sx(0)
dg_uxy_circ.sx(1)
dg_uxy_circ.rz(-np.pi,0)
dg_uxy_circ.rz(5*np.pi/4,1)
dg_uxy_circ.sx(1)
dg_uxy_circ.rz(7*np.pi/2,1)
dg_uxy_circ.cx(0,1)
dg_uxy_circ.rz(np.pi/2,0)
dg_uxy_circ.rz(np.pi/2,1)
dg_uxy_circ.sx(0)
dg_uxy_circ.sx(1)
dg_uxy_circ.rz(7*np.pi/4,1)
dg_uxy_circ.sx(1)
dg_uxy_circ.rz(7*np.pi/2,1)
dg_uxy_circ.name = 'U_xy Conj'
uxy_dg_gate = dg_uxy_circ.to_instruction()

class MajoranaCircuit:

    def __init__(self,api_key, i_theta=np.pi/4, i_phi=np.pi/8,n_qubits=10, n_cycles_total=40,n_avgs=8192,run_type='simulator',zz_gates=False,i_eta=None,noise_type=None,device='ibmq_mumbai',meas_type = 'ZX', meas_qubit = 'right',layout=None, hub='ibm-q-internal', group='deployed', project='default'):
        self.api_key = api_key
        self.theta = i_theta
        self.phi = i_phi
        self.n_qubits = n_qubits
        self.n_cycles_total = n_cycles_total
        self.n_avgs = n_avgs
        self.run_type = run_type
        self.zz_gates = zz_gates
        self.eta = i_eta
        self.noise_type = noise_type
        self.device = device
        self.meas_type = meas_type
        self.meas_qubit = meas_qubit
        self.layout = layout
        self.hub = hub
        self.group = group
        self.project = project

    def initialize(self,reload = False):
        if reload == True:
            IBMQ.save_account(self.api_key,overwrite=True)
        else:
            IBMQ.load_account()
        print('Your IBMQ account has been loaded.')

    def get_params(self):
        print(r'XX Gate Angle $ \theta $ is {}'.format(self.theta))
        print(r'Z Gate Angle $ \phi $ is {}'.format(self.phi))
        print('{} qubits'.format(self.n_qubits))
        print('Up to {} gate cycles'.format(self.n_cycles_total))
        print('{} averages'.format(self.n_avgs))
        print('Platform: {}'.format(self.run_type))
        print('ZZ gates: {}'.format(self.zz_gates))
        print(r'ZZ gate angle $ \eta $ is {}'.format(self.eta))
        print('Noise type: {}'.format(self.noise_type))
        print('Layout on {} is {}'.format(self.device,self.layout))
        print('Measurement is {} {}'.format(self.meas_type, self.meas_qubit))


    # Useful Functions
    def x_measurement(self,qc, qubit, cbit):
        """Measure 'qubit' in the X-basis, and store the result in 'cbit'"""
        qc.h(qubit)
        qc.measure(qubit, cbit)
        return qc

    def y_measurement(self,qc, qubit, cbit):
        """Measure 'qubit' in the Y-basis, and store the result in 'cbit'"""
        qc.sdg(qubit)
        qc.h(qubit)
        qc.measure(qubit, cbit)
        return qc


    def run_circuit_new(self, n_meas_gates_array, show_array = False):
        if isinstance(self.meas_type,str) == True:
            meas_type = [self.meas_type]
        else:
            meas_type = self.meas_type
        if isinstance(self.meas_qubit,str) == True:
            meas_qubit = [self.meas_qubit]
        else:
            meas_qubit = self.meas_qubit
        final_meas_array = np.zeros([self.n_cycles_total+1,len(n_meas_gates_array),len(meas_type)*len(meas_qubit)])
        final_unavgd_array = np.zeros([self.n_cycles_total+1,self.n_avgs,len(n_meas_gates_array),len(meas_type)*len(meas_qubit)])
        full_counts_array = [] #circuit_list = [] #np.zeros(n_cycles_total+1,dtype='object')
        count = 0
        full_circuit_list = []
        mem_list = []
        for mqubit in meas_qubit:
            for mtype in meas_type:
                for gindex in range(len(n_meas_gates_array)): #add n_qubits+1 circuits n_qubits times for each mqubit/mtype combo
                    n_meas_gates = n_meas_gates_array[gindex]
                    for n_cycles in range(self.n_cycles_total+1): #for each number of circuit cycles
                        # Step 1: Createself.n_qubits
                        Circuit = QuantumCircuit(self.n_qubits,1)
                        qr = Circuit.qregs[0]

                        # Step 2: Initialize all qubits to |+>
                        for qubit in range(self.n_qubits):
                            Circuit.h(qubit)

                        # Step 3: Apply 1 and 2 qubit gate n_cycles times
                        # first the 1-qubit gates
                        for i in range(n_cycles):
                            for qubit in range(self.n_qubits): #DO Z GATE ON EVERY QUBIT
                                Circuit.rz(2*self.phi, qubit) #rotated Z gate
                            for num in range(self.n_qubits): #DO A XX GATE FOR EVERY NEIGHBORING PAIR OF QUBITS
                                if num%2 ==1:
                                    if num-1 < 0:
                                        pass
                                    else:
                                        Circuit.rxx(2*self.theta,num-1,num)
                            for num in range(self.n_qubits): 
                                if num%2 ==0:
                                    if num-1 < 0:
                                        pass
                                    else:
                                        Circuit.rxx(2*self.theta,num-1,num)
                            if self.zz_gates == True: #DO A ZZ GATE FOR EVERY NEIGHBORING PAIR OF QUBITS
                                for num in range(self.n_qubits): 
                                    if num%2 ==1:
                                        if num-1 < 0:
                                            pass
                                        else:
                                            Circuit.rzz(2*self.eta,num-1,num)
                                for num in range(self.n_qubits): 
                                    if num%2 ==0:
                                        if num-1 < 0:
                                            pass
                                        else:
                                            Circuit.rzz(2*self.eta,num-1,num)

                        # Step 4: Measurement of each qubit
                        if mtype == 'ZX':
                            applied_gate = measurement_gate
                            measurement_function = self.x_measurement
                        elif mtype == 'ZY':
                            applied_gate = measurement_gate1
                            measurement_function = self.y_measurement
                        if mqubit == 'right':
                            for num in range(n_meas_gates-1):
                                num = n_meas_gates-1-num
                                Circuit.append(applied_gate, [self.n_qubits-num-1,self.n_qubits-num-0])
                            measurement_function(Circuit, self.n_qubits-1, 0)
                        elif mqubit == 'left':
                            for num in range(n_meas_gates-1)[::-1]:
                                num =self.n_qubits-1-num
                                Circuit.append(applied_gate, [self.n_qubits-num-0,self.n_qubits-num-1])
                            measurement_function(Circuit, 0, 0)
                        if self.layout != None and (self.run_type == 'computer' or self.noise_type == 'backend_model'):
                            provider = IBMQ.get_provider(hub=self.hub, group=self.group, project=self.project)
                            backend = provider.get_backend(self.device)
                            t_Circuit = transpile(Circuit,backend,basis_gates = ['id','rz','sx','x','cx','reset'],initial_layout=self.layout,optimization_level=3)
                            full_circuit_list.append(t_Circuit)
                        else:
                            full_circuit_list.append(Circuit)
                count += 1
        length = len(full_circuit_list)
        rec_circuit_list = []
        circ_ratio = length/299
        rnd = math.ceil(circ_ratio)
        for asd in range(rnd):
            circuit_list = full_circuit_list[int(length*asd/circ_ratio):int(length*(asd+1)/circ_ratio)]

            # Step 5: Measure qubit state and record in array
            if self.run_type == 'simulator':
                simulator = Aer.get_backend('qasm_simulator')
                if self.noise_type != None:
                    result = self.simulate_with_noise_model(circuit_list)
                else:
                    #Execute the circuit on Aer's qasm_simulator
                    basis_gates = ['cx', 'id', 'reset', 'rz', 'sx', 'x']
                    job = execute(circuit_list, simulator, basis_gates=basis_gates, shots=self.n_avgs,memory=True)
                    #Grab results from the job
                    result = job.result()
            elif self.run_type == 'computer':
                #or Execute the circuit on the quantum computer
                provider = IBMQ.get_provider(hub=self.hub, group=self.group, project=self.project)
                backend = provider.get_backend(self.device)
                job = backend.run(circuit_list,shots=self.n_avgs,memory=True)
                #job = qiskit.execute(circuit_list, backend=backend,initial_layout=layout,shots=n_avgs,memory=True)
                job_monitor(job)
                #Grab results from the job
                result = job.result()
            elif self.run_type == 'runtime':
                provider = IBMQ.get_provider(hub=self.hub, group=self.group, project=self.project)
                backend = provider.get_backend(self.device)
                result = provider.run_circuits(circuit_list, backend=backend, shots=self.n_avgs,memory=True).result()
            else:
                raise Exception('run_type parameter not understood: {}. Must be one of ["simulator","computer","runtime"]'.format(self.run_type))
            for cir in circuit_list:
                rec_circuit_list.append(cir)
                memory = result.get_memory(cir)
                mem_list.append(memory)

        
            #Return counts
            counts_array = result.get_counts()
            for dic in counts_array:
                full_counts_array.append(dic)
        for mindex in range(len(meas_type)*len(meas_qubit)):
            for gindex in range(len(n_meas_gates_array)):
                for n_cycles in range(self.n_cycles_total+1):
                    circ_index = n_cycles + gindex*(self.n_cycles_total+1) + len(n_meas_gates_array)*(self.n_cycles_total+1)*mindex
                    counts = full_counts_array[circ_index]
                    try:
                        high_count = counts['1'] #corresponds to -1
                    except KeyError:
                        high_count = 0
                    try:
                        low_count = counts['0'] #corresponds to +1
                    except KeyError:
                        low_count = 0
                    average = (low_count-high_count)/self.n_avgs
                    memory = mem_list[circ_index]
                    runs = []
                    for mem in memory:
                        if mem == '0':
                            val = -1
                            runs.append(val)
                        elif mem == '1':
                            val = 1
                            runs.append(val)
                    final_meas_array[n_cycles,gindex,mindex] = average
                    final_unavgd_array[n_cycles,:,gindex,mindex] = runs
        if show_array == True:
            plt.plot(np.array(final_meas_array),marker='o')
            plt.xlabel("cycle")
            plt.ylabel(r'Observable $\langle X_n \rangle$')
            plt.show()
        return final_meas_array, final_unavgd_array, Circuit

    def run_decoupled_circuit(self, n_meas_gates_array, decoupled=True,theta_red = 0, phi_red = 0, unitary_approach = False, man_avg = False):
        final_meas_array = np.zeros([self.n_cycles_total+1,len(n_meas_gates_array)])
        final_unavgd_array = np.zeros([self.n_cycles_total+1,self.n_avgs,len(n_meas_gates_array)])
        if man_avg == True:
            avg_array = np.arange(1, self.n_avgs+1)
        else:
            avg_array = [1]
        full_circuit_list = np.zeros([len(n_meas_gates_array)*(self.n_cycles_total+1), len(avg_array)], dtype=object)
        full_counts_array = np.zeros([len(n_meas_gates_array)*(self.n_cycles_total+1), len(avg_array)], dtype=object)
        mem_list = []
        rand_init_list = np.zeros([self.n_qubits-2, len(avg_array)])
        for indx in range(self.n_qubits-2):
            for avgdex in range(len(avg_array)):
                rand_init_list[indx, avgdex] = random.randint(0,1)
        circuit_index = 0
        for gindex in range(len(n_meas_gates_array)): #add n_qubits+1 circuits n_qubits times for each meas_qubit/mtype combo
            n_meas_gates = n_meas_gates_array[gindex]
            for n_cycles in range(self.n_cycles_total+1): #for each number of circuit cycles
                for average in avg_array:
                    # Step 1: Create self.n_qubits qubits
                    if unitary_approach == True:
                        n_cbits = 1
                    else:
                        n_cbits = n_meas_gates
                    Circuit = QuantumCircuit(self.n_qubits,n_cbits)
                    # Step 2: Initialize edge qubits to [2,3i] and middle qubits randomly to |0> or |1>
                    Circuit.append(init_gate_new,[0])
                    Circuit.append(init_gate_new,[self.n_qubits-1])
                    for qubit in range(self.n_qubits-2):
                        rint = rand_init_list[qubit, avgdex]
                        if rint == 1:
                            Circuit.x(1+qubit)
                        elif rint == 0:
                            Circuit.id(1+qubit)
                    if decoupled == True:
                        for i in range(n_cycles):
                            for qubit in range(self.n_qubits-2): #DO Z GATE ON MIDDLE QUBITS
                                Circuit.rz(2*self.phi, qubit+1)
                            Circuit.rz(2*self.phi*phi_red,0) #do reduced Z gate on edge qubits
                            Circuit.rz(2*self.phi*phi_red,self.n_qubits-1)
                            for num in range(self.n_qubits-2): #DO A XX GATE FOR EVERY NEIGHBORING PAIR OF MIDDLE QUBITS
                                if num%2 ==1:
                                    if num-1 < 0:
                                        pass
                                    else:
                                        Circuit.rxx(2*self.theta,num,num+1)
                            for num in range(self.n_qubits-2): 
                                if num%2 ==0:
                                    if num-1 < 0:
                                        pass
                                    else:
                                        Circuit.rxx(2*self.theta,num,num+1)
                            Circuit.rxx(2*self.theta*theta_red,0,1) #do reduced XX gate on edge qubits
                            Circuit.rxx(2*self.theta*theta_red,self.n_qubits-2,self.n_qubits-1)
                    elif decoupled == False:
                        # Step 3: Apply 1 and 2 qubit gate n_cycles times
                        # first the 1-qubit gates
                        for i in range(n_cycles):
                            for qubit in range(self.n_qubits): #DO Z GATE ON EVERY QUBIT
                                Circuit.rz(2*self.phi, qubit) #rotated Z gate
                            for num in range(self.n_qubits): #DO A XX GATE FOR EVERY NEIGHBORING PAIR OF QUBITS
                                if num%2 ==1:
                                    if num-1 < 0:
                                        pass
                                    else:
                                        Circuit.rxx(2*self.theta,num-1,num)
                            for num in range(self.n_qubits): 
                                if num%2 ==0:
                                    if num-1 < 0:
                                        pass
                                    else:
                                        Circuit.rxx(2*self.theta,num-1,num)
                            if self.zz_gates == True: #DO A ZZ GATE FOR EVERY NEIGHBORING PAIR OF QUBITS
                                for num in range(self.n_qubits): 
                                    if num%2 ==1:
                                        if num-1 < 0:
                                            pass
                                        else:
                                            Circuit.rzz(2*self.eta,num-1,num)
                                for num in range(self.n_qubits): 
                                    if num%2 ==0:
                                        if num-1 < 0:
                                            pass
                                        else:
                                            Circuit.rzz(2*self.eta,num-1,num)
                    if unitary_approach == True:
                        if n_meas_gates == 1:
                            if self.meas_qubit == 'right':
                                Circuit.measure(self.n_qubits-1,0)
                            elif self.meas_qubit == 'left':
                                Circuit.measure(0,0)
                        else:
                            applied_gate = measurement_gate1
                            if self.meas_qubit == 'right':
                                Circuit.sdg(self.n_qubits-1)
                                Circuit.h(self.n_qubits-1)
                                for num in range(n_meas_gates-2):
                                    num = n_meas_gates-1-num
                                    Circuit.append(applied_gate, [self.n_qubits-num-1,self.n_qubits-num])
                                Circuit.append(applied_gate, [self.n_qubits-2,self.n_qubits-1])
                                self.y_measurement(Circuit, self.n_qubits-1, 0)
                            elif self.meas_qubit == 'left':
                                Circuit.sdg(0)
                                Circuit.h(0)
                                for num in range(n_meas_gates-2)[::-1]:
                                    num = self.n_qubits-1-num
                                    Circuit.append(applied_gate, [self.n_qubits-num+1,self.n_qubits-num])
                                Circuit.append(applied_gate, [1,0])
                                self.y_measurement(Circuit, 0, 0)
                    else:
                        if n_meas_gates == 1:
                            if self.meas_qubit == 'right':
                                Circuit.measure(self.n_qubits-1,0)
                            elif self.meas_qubit == 'left':
                                Circuit.measure(0,0)
                        elif n_meas_gates == 2:
                            if self.meas_qubit == 'right':
                                self.y_measurement(Circuit,self.n_qubits-1,0)
                                self.y_measurement(Circuit,self.n_qubits-2,1)
                            elif self.meas_qubit == 'left':
                                self.y_measurement(Circuit,0,0)
                                self.y_measurement(Circuit,1,1)
                        elif n_meas_gates > 2:
                            if self.meas_qubit == 'right':
                                self.y_measurement(Circuit,self.n_qubits-1,0)
                                self.y_measurement(Circuit,self.n_qubits-n_meas_gates-1,n_meas_gates)
                                for qubit in range(self.n_qubits-n_meas_gates,self.n_qubits-1):
                                    Circuit.measure(qubit,self.n_qubits-1-qubit)
                            elif self.meas_qubit == 'left':
                                self.y_measurement(Circuit,0,0)
                                self.y_measurement(Circuit,n_meas_gates-1,n_meas_gates-1)
                                for qubit in range(1,n_meas_gates-1):
                                    Circuit.measure(qubit,qubit)

                    
                    if self.layout != None and (self.run_type == 'computer' or self.noise_type == 'backend_model'):
                        provider = IBMQ.get_provider(hub=self.hub, group=self.group, project=self.project)
                        backend = provider.get_backend(self.device)
                        t_Circuit = transpile(Circuit,backend,basis_gates = ['id','rz','sx','x','cx','reset'],initial_layout=self.layout,optimization_level=3)
                        full_circuit_list[circuit_index,average-1] = t_Circuit
                    else:
                        full_circuit_list[circuit_index,average-1] = Circuit
                circuit_index += 1
        if man_avg == True:
            num_shots = 1
        else:
            num_shots = self.n_avgs
        for average in avg_array:
            ind = 0
            length = len(full_circuit_list)
            rec_circuit_list = np.zeros([len(n_meas_gates_array)*(self.n_cycles_total+1), len(avg_array)], dtype=object)
            circ_ratio = length/800
            rnd = math.ceil(circ_ratio)
            for asd in range(rnd):
                circuit_list = full_circuit_list[int(length*asd/circ_ratio):int(length*(asd+1)/circ_ratio), average-1].tolist()
                # Step 5: Measure qubit state and record in array
                if self.run_type == 'simulator':
                    simulator = Aer.get_backend('qasm_simulator')
                    if self.noise_type != None:
                        result = self.simulate_with_noise_model(circuit_list)
                    else:
                        #Execute the circuit on Aer's qasm_simulator
                        basis_gates = ['cx', 'id', 'reset', 'rz', 'sx', 'x']
                        job = execute(circuit_list, simulator, basis_gates=basis_gates, shots=num_shots,memory=True)
                        #Grab results from the job
                        result = job.result()
                elif self.run_type == 'computer':
                    #or Execute the circuit on the quantum computer
                    provider = IBMQ.get_provider(hub=self.hub, group=self.group, project=self.project)
                    backend = provider.get_backend(self.device)
                    job = backend.run(circuit_list,shots=num_shots,memory=True)
                    job_monitor(job)
                    #Grab results from the job
                    result = job.result()
                elif self.run_type == 'runtime':  
                    provider = IBMQ.get_provider(hub=self.hub, group=self.group, project=self.project)
                    backend = provider.backend.ibmq_montreal
                    result = provider.run_circuits(circuit_list, backend=backend, shots=num_shots,memory=True).result()
                cir_index = 0
                for cir in circuit_list:
                    rec_circuit_list[int(length*asd/circ_ratio)+cir_index,average-1] = cir
                    #rec_circuit_list.append(cir)
                    memory = result.get_memory(cir)
                    if man_avg == False:
                        mem_list.append(memory)
                    cir_index += 1
            
                #Return counts
                counts_array = result.get_counts()
                for dic in counts_array:
                    full_counts_array[ind,average-1] = dic
                    #full_counts_array.append(dic)
                    ind += 1
        if man_avg == False:
            for gindex in range(len(n_meas_gates_array)):
                for n_cycles in range(self.n_cycles_total+1):
                    circ_index = n_cycles + gindex*(self.n_cycles_total+1)
                    counts = full_counts_array[circ_index, 0]
                    if unitary_approach == True:
                        try:
                            high_count = counts['1']
                        except KeyError:
                            high_count = 0
                        try:
                            low_count = counts['0']
                        except KeyError:
                            low_count = 0
                        average = (low_count-high_count)/self.n_avgs
                    else:
                        average = 0
                        for count in counts:
                            frequency = counts[count]
                            meas_val = 1
                            for index in range(len(count)):
                                meas_val = (2*int(count[index])-1)*meas_val
                            result = meas_val*frequency/(self.n_avgs)
                            average += result
                    #circuit_rec = full_circuit_list[circ_index]
                    #memory = result.get_memory(circuit_rec)
                    memory = mem_list[circ_index]
                    runs = []
                    if unitary_approach == True:
                        for mem in memory:
                            if mem == '0':
                                val = -1
                                runs.append(val)
                            elif mem == '1':
                                val = 1
                                runs.append(val)
                    else:
                        for mem in memory:
                            meas_val = 1
                            for index in range(len(mem)):
                                meas_val = (2*int(mem[index])-1)*meas_val
                            runs.append(meas_val)
                    final_meas_array[n_cycles,gindex] = average
                    final_unavgd_array[n_cycles,:,gindex] = runs
        elif man_avg == True:
            for gindex in range(len(n_meas_gates_array)):
                for n_cycles in range(self.n_cycles_total+1):
                    circ_index = n_cycles + gindex*(self.n_cycles_total+1)
                    runs = []
                    for avg_index in avg_array:
                        counts = full_counts_array[circ_index, avg_index-1]
                        if unitary_approach == True:
                            try:
                                high_count = counts['1']
                            except KeyError:
                                high_count = 0
                            try:
                                low_count = counts['0']
                            except KeyError:
                                low_count = 0
                            average = (low_count-high_count)
                            runs.append(average)
                        else:
                            average = 0
                            for count in counts:
                                frequency = counts[count]
                                meas_val = 1
                                for index in range(len(count)):
                                    meas_val = (2*int(count[index])-1)*meas_val
                                result = meas_val*frequency
                                average += result
                            runs.append(average)
                    final_meas_array[n_cycles,gindex] = np.mean(runs)
                    final_unavgd_array[n_cycles,:,gindex] = runs
        return final_meas_array, final_unavgd_array, full_circuit_list


    def build_circuit(self, n_cycles, input_circuit=None, init = True):
        # Step 1: Create or load circuit (n = self.n_qubits)
        if input_circuit == None:
            Circuit = QuantumCircuit(self.n_qubits,1)
        else:
            Circuit = input_circuit

        # Step 2: Initialize all qubits to |+> (optional)
        if init == True:
            for qubit in range(self.n_qubits):
                Circuit.h(qubit)

        # Step 3: Apply 1 and 2 qubit gate n_cycles times
        # first the 1-qubit gates
        for i in range(n_cycles):
            for qubit in range(self.n_qubits): #DO Z GATE ON EVERY QUBIT
                Circuit.rz(2*self.phi, qubit) #rotated Z gate
            for num in range(self.n_qubits): #DO A XX GATE FOR EVERY NEIGHBORING PAIR OF QUBITS
                if num%2 ==1:
                    if num-1 < 0:
                        pass
                    else:
                        Circuit.rxx(2*self.theta,num-1,num)
            for num in range(self.n_qubits): 
                if num%2 ==0:
                    if num-1 < 0:
                        pass
                    else:
                        Circuit.rxx(2*self.theta,num-1,num)
            if self.zz_gates == True: #DO A ZZ GATE FOR EVERY NEIGHBORING PAIR OF QUBITS
                for num in range(self.n_qubits): 
                    if num%2 ==1:
                        if num-1 < 0:
                            pass
                        else:
                            Circuit.rzz(2*self.eta,num-1,num)
                for num in range(self.n_qubits): 
                    if num%2 ==0:
                        if num-1 < 0:
                            pass
                        else:
                            Circuit.rzz(2*self.eta,num-1,num)
        return Circuit 

    def braid_majoranas(self, n_meas_gates_array, braiding=True, alpha_val = None):
        if isinstance(self.meas_type,str) == True:
            meas_type = [self.meas_type]
        else:
            meas_type = self.meas_type
        if isinstance(self.meas_qubit,str) == True:
            meas_qubit = [self.meas_qubit]
        else:
            meas_qubit = self.meas_qubit
        final_meas_array = np.zeros([self.n_cycles_total+1,len(n_meas_gates_array),len(meas_type)*len(meas_qubit)])
        final_unavgd_array = np.zeros([self.n_cycles_total+1,self.n_avgs,len(n_meas_gates_array),len(meas_type)*len(meas_qubit)])
        full_counts_array = []
        count = 0
        full_circuit_list = []
        mem_list = []
        for mqubit in meas_qubit:
            for mtype in meas_type:
                for gindex in range(len(n_meas_gates_array)): #add n_qubits+1 circuits n_qubits times for each mqubit/mtype combo
                    n_meas_gates = n_meas_gates_array[gindex]
                    for n_cycles in range(self.n_cycles_total+1): #for each number of circuit cycles
                        blankcircuit = QuantumCircuit(self.n_qubits, 1)
                        Circuit = self.build_circuit(n_cycles, input_circuit = blankcircuit, init = True)
                        if braiding == True:
                            #APPLY BRAIDING GATES HERE
                            #first, the U(+ipi/4 XY) gates
                            for qubit in range(self.n_qubits-2):
                                Circuit.append(uxy_gate, [self.n_qubits-1 - qubit,self.n_qubits - qubit-2])
                            if alpha_val == None:
                                alpha_val = np.pi*5/16
                            alpha = alpha_val
                            Circuit.ryy(2*alpha, 1,0)#self.n_qubits-2, self.n_qubits-1)
                            #last, the U(+ipi/4 XY) dagger gates
                            for qubit in range(self.n_qubits-2)[::-1]:
                                Circuit.append(uxy_dg_gate, [self.n_qubits-1 -qubit,self.n_qubits-1 - qubit-1])
                            #END BRAIDING GATES SECTION
                            Circuit = self.build_circuit(n_cycles, input_circuit = Circuit, init = False)
                        # Step 4: Measurement of each qubit
                        if mtype == 'ZX':
                            applied_gate = measurement_gate
                            measurement_function = self.x_measurement
                        elif mtype == 'ZY':
                            applied_gate = uxy_gate #measurement_gate1
                            measurement_function = self.y_measurement
                        if mqubit == 'right':
                            for num in range(n_meas_gates-1):
                                num = n_meas_gates-1-num
                                Circuit.append(applied_gate, [self.n_qubits-num-1,self.n_qubits-num-0])
                            measurement_function(Circuit, self.n_qubits-1, 0)
                        elif mqubit == 'left':
                            for num in range(n_meas_gates-1)[::-1]:
                                num =self.n_qubits-1-num
                                Circuit.append(applied_gate, [self.n_qubits-num-0,self.n_qubits-num-1])
                            measurement_function(Circuit, 0, 0)
                        full_circuit_list.append(Circuit)
                count += 1
        length = len(full_circuit_list)
        rec_circuit_list = []
        circ_ratio = length/299
        rnd = math.ceil(circ_ratio)
        if self.run_type == 'simulator':
            simulator = Aer.get_backend('qasm_simulator')
            if self.noise_type != None:
                provider = IBMQ.get_provider(hub=self.hub, group=self.group, project=self.project)
                backend = provider.get_backend(self.device)
                noise_model = NoiseModel.from_backend(backend)
        for asd in range(rnd):
            circuit_list = full_circuit_list[int(length*asd/circ_ratio):int(length*(asd+1)/circ_ratio)]
            # Step 5: Measure qubit state and record in array
            if self.run_type == 'simulator':
                if self.noise_type != None:
                    result = self.simulate_with_noise_model(circuit_list, opt_level = 2,noise_model=noise_model,simulator=simulator)
                else:
                    #Execute the circuit on Aer's qasm_simulator
                    basis_gates = ['cx', 'id', 'reset', 'rz', 'sx', 'x']
                    job = execute(circuit_list, simulator, basis_gates=basis_gates, shots=self.n_avgs,memory=True)
                    #Grab results from the job
                    result = job.result()
            elif self.run_type == 'computer':
                #or Execute the circuit on the quantum computer
                provider = IBMQ.get_provider(hub=self.hub, group=self.group, project=self.project)
                backend = provider.get_backend(self.device)
                basis_gates = ['cx', 'id', 'reset', 'rz', 'sx', 'x']
                #job = backend.run(circuit_list,shots=self.n_avgs,memory=True)
                job = execute(circuit_list, backend=backend,initial_layout=self.layout,basis_gates = basis_gates,shots=self.n_avgs,optimization_level=3,memory=True)
                print(job.job_id())
                job_monitor(job)
                #Grab results from the job
                result = job.result()
            elif self.run_type == 'runtime':
                provider = IBMQ.get_provider(hub=self.hub, group=self.group, project=self.project)
                backend = provider.get_backend(self.device)
                t_circuit_list = transpile(circuit_list,backend,basis_gates = ['id','rz','sx','x','cx','reset'],initial_layout=self.layout,optimization_level=3)
                program_inputs = {
                    'circuits': t_circuit_list,
                    'optimization_level': 3,
                    'measurement_error_mitigation': True,
                    'shots': self.n_avgs,
                    'initial_layout': self.layout,
                    'memory': True}
                options = {'backend_name': backend.name()}
                job = provider.runtime.run(program_id="circuit-runner",
                                        options=options,
                                        inputs=program_inputs)
                print(f"job ID: {job.job_id()}")
                result = job.result(decoder=RunnerResult)
            else:
                raise Exception('"run_type" parameter not understood: {}. Must be one of ["simulator","computer","runtime"]'.format(self.run_type))
            for cir in circuit_list:
                rec_circuit_list.append(cir)
                memory = result.get_memory(cir)
                mem_list.append(memory)
        
            #Return counts
            if self.run_type == 'runtime':
                counts_array = result.get_quasiprobabilities()
            else:
                counts_array = result.get_counts()

            for dic in counts_array:
                full_counts_array.append(dic)
        for mindex in range(len(meas_type)*len(meas_qubit)):
            for gindex in range(len(n_meas_gates_array)):
                for n_cycles in range(self.n_cycles_total+1):
                    circ_index = n_cycles + gindex*(self.n_cycles_total+1) + len(n_meas_gates_array)*(self.n_cycles_total+1)*mindex
                    counts = full_counts_array[circ_index]
                    try:
                        high_count = counts['1']
                    except KeyError:
                        high_count = 0
                    try:
                        low_count = counts['0']
                    except KeyError:
                        low_count = 0
                    average = (low_count-high_count)/self.n_avgs
                    memory = mem_list[circ_index]
                    runs = []
                    for mem in memory:
                        if mem == '0':
                            val = -1
                            runs.append(val)
                        elif mem == '1':
                            val = 1
                            runs.append(val)
                    final_meas_array[n_cycles,gindex,mindex] = average
                    final_unavgd_array[n_cycles,:,gindex,mindex] = runs
        return final_meas_array, final_unavgd_array, full_circuit_list


        #part 2 - evaluate Fourier Transform:
    def function(self, omega, final_meas_array):
        x_array = final_meas_array
        result = 0
        count = 0
        for value in x_array:
            term = value*np.exp(-1j*omega*(count))
            result += term
            count += 1
        result = result/self.n_cycles_total
        return result

    def fourier_transform(self,a,w,max_depth,n_terms = None, start = 0,end=0):
        if n_terms == None:
            pass
        else:
            a = a[0:n_terms]
            max_depth = n_terms-1
            if max_depth == 0:
                max_depth == 1
        if start != 0:
            a[:start] = np.zeros(start)
        if end != 0:
            a[(end+1):] = np.zeros(len(a)-end-1)
        return np.sum(a*np.exp(1j*w*np.arange(len(a))))/max_depth


    def simulate_with_noise_model(self,circuit, opt_level = 3,noise_model = None,simulator = None):
        if noise_model == None:
            if self.noise_type == 'backend_model':
                provider = IBMQ.get_provider(hub=self.hub, group=self.group, project=self.project)
                backend = provider.get_backend(self.device)
                noise_model = NoiseModel.from_backend(backend)  
                #print(noise_model)
            elif self.noise_type == 'depolarizing':
                # Depolarizing quantum errors
                prob_x = 0.00146
                prob_cx = 0.025
                prob_rz = 0
                x_error = noise.depolarizing_error(prob_x, 1)
                cx_error = noise.depolarizing_error(prob_cx, 2)
                rz_error = noise.depolarizing_error(prob_rz, 1)

                # Add errors to noise model
                noise_model = noise.NoiseModel()
                noise_model.add_all_qubit_quantum_error(x_error, ['sx', 'x','id'])
                noise_model.add_all_qubit_quantum_error(rz_error, ['rz'])
                noise_model.add_all_qubit_quantum_error(cx_error, ['cx'])
                #print(noise_model)
            elif self.noise_type == 'thermal':
                # T1 and T2 values for qubits 0-4
                t1_time = 150e3
                t2_time = 120e3
                T1s = np.random.normal(t1_time,20e3,5) #(50e3, 10e3, 5) # Sampled from normal distribution mean 50 microsec
                T2s = np.random.normal(t2_time,20e3,5) #(70e3, 10e3, 5)  # Sampled from normal distribution mean 50 microsec

                # Truncate random T2s <= T1s
                T2s = np.array([min(T2s[j], 2 * T1s[j]) for j in range(self.n_qubits)])

                # Instruction times (in nanoseconds)
                time_u1 = 0.00000000000000001   # virtual gate
                time_u2 = 50  # (single X90 pulse)
                time_u3 = 100 # (two X90 pulses)
                time_cx = 400 #300
                time_reset = 1000  # 1 microsecond
                time_measure = 1000 # 1 microsecond

                # QuantumError objects
                errors_reset = [thermal_relaxation_error(t1, t2, time_reset)
                                for t1, t2 in zip(T1s, T2s)]
                errors_measure = [thermal_relaxation_error(t1, t2, time_measure)
                                for t1, t2 in zip(T1s, T2s)]
                errors_rz = [thermal_relaxation_error(t1, t2, time_u1)
                            for t1, t2 in zip(T1s, T2s)]
                errors_u3  = [thermal_relaxation_error(t1, t2, time_u3)
                            for t1, t2 in zip(T1s, T2s)]
                errors_sx  = [thermal_relaxation_error(t1, t2, time_u2)
                            for t1, t2 in zip(T1s, T2s)]
                errors_x  = [thermal_relaxation_error(t1, t2, time_u3)
                            for t1, t2 in zip(T1s, T2s)]
                errors_cx = [[thermal_relaxation_error(t1a, t2a, time_cx).expand(
                            thermal_relaxation_error(t1b, t2b, time_cx))
                            for t1a, t2a in zip(T1s, T2s)]
                            for t1b, t2b in zip(T1s, T2s)]

                # Add errors to noise model
                noise_model = NoiseModel()
                for j in range(self.n_qubits):
                    noise_model.add_quantum_error(errors_reset[j], "reset", [j])
                    noise_model.add_quantum_error(errors_measure[j], "measure", [j])
                    noise_model.add_quantum_error(errors_rz[j],'rz',[j])
                    noise_model.add_quantum_error(errors_u3[j], "u3", [j])
                    noise_model.add_quantum_error(errors_sx[j], "sx", [j])
                    noise_model.add_quantum_error(errors_x[j], "x", [j])
                    for k in range(self.n_qubits):
                        noise_model.add_quantum_error(errors_cx[j][k], "cx", [j, k])

        # Get the basis gates for the noise model
        basis_gates = ['cx', 'id', 'reset', 'rz', 'sx', 'x']#noise_model.basis_gates

        # Select the QasmSimulator from the Aer provider
        if simulator == None:
            simulator = Aer.get_backend('qasm_simulator')
        if self.noise_type == 'backend_model' and self.layout is not None:
            # Execute noisy simulation and get counts
            result_noise = execute(circuit, simulator,
                                noise_model=noise_model,
                                basis_gates=basis_gates, shots=self.n_avgs,layout = self.layout,optimization_level = opt_level, memory=True).result()
        else:
            result_noise = execute(circuit, simulator,
                                noise_model=noise_model,
                                basis_gates=basis_gates, shots=self.n_avgs,optimization_level = opt_level,memory=True).result()
        return result_noise

    def store_measurement(self,final_meas_group,final_unavgd_group,directory,filename,n_meas_gates_array,numerical=False):
        num_meas_gates = max(n_meas_gates_array)
        length = 18
        if self.zz_gates == True:
            length += 2
        if self.layout is not None and self.run_type == 'computer':
            length += 2
        y_width = max(self.n_avgs+3,length)
        result = np.zeros([(self.n_cycles_total+4)*num_meas_gates+1,y_width],dtype = 'object')
        for x in range((self.n_cycles_total+4)*num_meas_gates+1):
            for y in range(y_width):
                result[x,y] = ''
        if self.meas_type == 'ZX':
            basic_str = 'X'
        elif self.meas_type == 'ZY':
            basic_str = 'Y'
        else:
            raise Exception("Gate type not recognized. Please input either 'ZX' or 'ZY'.")
        #gate angles, presence of ZZ gate, whether the measurement was ZZ…X or ZZ…. Y, number of qubits and number of avgs, cycles
        result[0,0] = 'Number of qubits is:'
        result[0,1] = self.n_qubits
        result[0,2] = 'Number of cycles is:'
        result[0,3] = self.n_cycles_total
        result[0,4] = 'Number of averages is:'
        result[0,5] = self.n_avgs
        result[0,6] = 'This data was taken on a:'
        result[0,7] = self.run_type
        result[0,8] = 'The measurement type is:'
        result[0,9] = self.meas_type
        result[0,10] = 'Gate angle theta is {}pi:'.format(self.theta/np.pi)
        result[0,11] = self.theta
        result[0,12] = 'Gate angle phi is {}pi:'.format(self.phi/np.pi)
        result[0,13] = self.phi
        result[0,14] = 'Measurement qubit:'
        result[0,15] = self.meas_qubit
        result[0,16] = 'Type of noise:'
        result[0,17] = self.noise_type
        if self.zz_gates == True:
            result[0,18] = 'Gate angle eta is {}pi:'.format(self.eta/np.pi)
            result[0,19] = self.eta
        if self.layout is not None and self.run_type == 'computer':
            result[0,20] = 'Device qubit layout is:'
            result[0,21] = self.layout
        for i in range(1,num_meas_gates+1):
            final_meas_array = final_meas_group[:,i-1]
            final_unavgd_array = final_unavgd_group[:,:,i-1]
            if numerical == True:
                depth_arr = np.arange(self.n_cycles_total+1)
                if self.meas_type == 'ZX':
                    basis = 'x'
                elif self.meas_type == 'ZY':
                    basis = 'y'
                final_meas_array = observable(self.n_qubits,depth_arr,theta,self.eta,i-1,basis = basis)
            if i == 1:
                meas_gate = basic_str
            else:
                prepend = ''
                for length in range(i-1):
                    prepend = prepend + 'Z'
                meas_gate = prepend + basic_str
            result[(i-1)*(self.n_cycles_total+4)+1,0]  = '{} measurement'.format(meas_gate)
            result[(i-1)*(self.n_cycles_total+4)+1,1]  = 'Average Result'
            for avg in range(1,self.n_avgs+1):
                result[(i-1)*(self.n_cycles_total+4)+1,avg+1]  = 'Run {}'.format(avg)
            for cycle in range(self.n_cycles_total+1):
                result[(i-1)*(self.n_cycles_total+4)+2+cycle,0] = '{} gate cycles'.format(cycle)
            if numerical == False:
                result[((i-1)*(self.n_cycles_total+4)+2):((i)*(self.n_cycles_total+4)-1),2:self.n_avgs+2] = final_unavgd_array
            result[((i-1)*(self.n_cycles_total+4)+2):((i)*(self.n_cycles_total+4)-1),1] = final_meas_array
        writename = directory + '/' + filename
        with open(writename,'w') as f:
            writer = csv.writer(f)
            writer.writerows(result)


    #this loads the full set of measurement runs from the Excel file
    def load_run_data(self,filename, returned_vals = None, printed_vals = 'all',full = True,num_meas_gates=None):
        df = pd.read_csv(filename,header=None,dtype='object')
        result = np.array(df)
        #code that reads out number of averages, cycles, qubits, etc.
        n_qubits = int(result[0,1])
        n_cycles_total = int(result[0,3])
        n_avgs = int(result[0,5]) 
        run_type = result[0,7]
        meas_type = result[0,9]
        theta = float(result[0,11])
        phi = float(result[0,13])
        meas_qubit = result[0,15]
        try:
            noise_type = str(result[0,17])
        except TypeError:
            noise_type = None
        try:
            eta = float(result[0,19])
        except IndexError:
            eta = None
        #code that displays the above information
        if printed_vals == 'all':
            print('Experiment Parameters:','\n',
                '{} qubits'.format(n_qubits),'\n', 
                '{} gate cycles'.format(n_cycles_total),'\n',
                '{} averages'.format(n_avgs),'\n',
                'Data taken on a {}'.format(run_type),'\n',
                'Qubits measured in {} basis'.format(meas_type),'\n',
                r"Gate angle $\theta $ is {}$\pi $".format(theta/np.pi),'\n',
                r"Gate angle $\phi $ is {}$\pi $".format(phi/np.pi),'\n',
                'Measurement qubit is on the {} side'.format(meas_qubit),'\n',
                'Noise: {}'.format(noise_type))
            if eta != None:
                print(r" Gate angle $\eta $ is {}$\pi $".format(eta/np.pi))
        #code that writes whole array of measurement runs
        if num_meas_gates == None:
            num_meas_gates = n_qubits
        meas_data = np.zeros([n_cycles_total+1,num_meas_gates])
        full_data = np.zeros([n_cycles_total+1,n_avgs,num_meas_gates])  
        for i in range(num_meas_gates+1):
            final_meas_array = result[((i-1)*(n_cycles_total+4)+2):((i)*(n_cycles_total+4)-1),1]
            meas_data[:,i-1] = final_meas_array
            if full == True:
                final_unavgd_array = result[((i-1)*(n_cycles_total+4)+2):((i)*(n_cycles_total+4)-1),2:n_avgs+2]
                full_data[:,:,i-1] = final_unavgd_array
        if full == True:
            if returned_vals == None:
                return meas_data, full_data
            elif returned_vals == 'all':
                if eta != None:
                    return meas_data, full_data, [theta, phi, eta],n_qubits, n_cycles_total, n_avgs, run_type, meas_type, meas_qubit, noise_type
                else:
                    return meas_data, full_data, [theta, phi],n_qubits, n_cycles_total, n_avgs, run_type, meas_type, meas_qubit, noise_type
            elif returned_vals == 'angles':
                if eta != None:
                    return meas_data, full_data, [theta, phi, eta]
                else:
                    return meas_data, full_data, [theta, phi]
        else:
            if returned_vals == None:
                return meas_data
            elif returned_vals == 'all':
                if eta != None:
                    return meas_data, [theta, phi, eta],n_qubits, n_cycles_total,n_avgs, run_type, meas_type, meas_qubit, noise_type
                else:
                    return meas_data, [theta, phi],n_qubits, n_cycles_total,n_avgs, run_type, meas_type, meas_qubit, noise_type
            elif returned_vals == 'angles':
                if eta != None:
                    return meas_data, [theta, phi, eta]
                else:
                    return meas_data, [theta, phi]
                
    def load_measurement(self,meas_data, observable):
        """ This function loads a specified observable measured at different depths.
            Args: meas_data (2D array): the data from which you're pulling out a specific observable
                  observable (str): of the form 'ZZ..ZX' or 'ZZ..ZY'. Do not deviate from this form
            Returns: final_meas_array (1D array): slice of meas_data for the observable you specify. 
        """
        substring = 'Z'
        if observable == 'X' or observable == 'Y':
            slide = 0
        else:
            count = observable.count(substring)
            slide = count
        final_meas_array = meas_data[:,slide]
        return final_meas_array
            
        
    def calculate_eigenmodes_no_save(self,directory = None,omega = 0, load_from_file = False,numerical = False):

        if load_from_file:
            readname = directory + '/right_ZX.csv'
            _, _, angles, n_qubits, n_cycles_total, n_avgs, run_type, meas_type, meas_qubit, noise_type = self.load_run_data(readname, returned_vals = 'all', printed_vals = None, full = True)
            self.theta = angles[0]
            self.phi = angles[1]
            try:
                self.eta = angles[2]
                if np.isnan(self.eta) == True:
                    self.eta = 0
            except KeyError:
                self.eta = 0
            if self.eta == 0:
                self.zz_gates = False
            else:
                self.zz_gates = True
            self.n_qubits = n_qubits
            self.n_cycles_total = n_cycles_total
            self.n_avgs = n_avgs
        else:
            og_meas_type = self.meas_type
            og_meas_qubit = self.meas_qubit
        o_array = np.zeros([self.n_qubits,4])
        error_array = np.zeros([self.n_qubits,4])
        directions = ['right','left']
        gates = ['ZX', 'ZY']
        count = -1
        if load_from_file == False:
            if numerical == False:
                n_meas_gates_array = np.arange(1,self.n_qubits+1)
                self.meas_qubit = directions
                self.meas_type = gates
                final_meas_matrix, final_unavgd_matrix, circuit = self.run_circuit_new(n_meas_gates_array)
        for starting_position in directions:
            for meas_gate in gates:
                count += 1
                base_str = meas_gate[1]
                gate_array = [base_str]
                for i in range(1,self.n_qubits):
                    prepend = ''
                    for length in range(i):
                        prepend = prepend + 'Z'
                    gate_array.append(prepend+base_str)
                filename = '{}_{}.csv'.format(starting_position,meas_gate)
                if load_from_file == False:
#                     final_meas_group = final_meas_matrix[:,:,count]
#                     final_unavgd_group = final_unavgd_matrix[:,:,:,count]
                    meas_data = final_meas_matrix[:,:,count]
                    full_data = final_unavgd_matrix[:,:,:,count]
                    self.meas_type = meas_gate
                    self.meas_qubit = starting_position
#                     self.store_measurement(final_meas_group, final_unavgd_group,directory,filename,n_meas_gates_array, numerical=numerical)
                elif load_from_file:
                    readname = directory + '/' + filename
                    if numerical == False:
                        meas_data, full_data, angles, n_qubits, n_cycles_total, n_avgs, run_type, meas_type, meas_qubit, noise_type = self.load_run_data(readname, returned_vals = 'all', printed_vals = None, full = True)
                        params = [angles, n_qubits, n_cycles_total, n_avgs, run_type, meas_type, meas_qubit, noise_type]
                    elif numerical == True:
                        meas_data, angles = self.load_run_data(readname, returned_vals = 'angles', printed_vals = None, full = False)
                    if count == 0:
                        len0, len1 = meas_data.shape[0], meas_data.shape[1]
                        final_meas_matrix = np.zeros([len0,len1,4])
                        len0, len1, len2 = full_data.shape[0], full_data.shape[1], full_data.shape[2]
                        final_unavgd_matrix = np.zeros([len0,len1,len2,4])
                    final_meas_matrix[:,:,count] = meas_data
                    final_unavgd_matrix[:,:,:,count] = full_data

                #calculating \sqrt{|O_1(\omega_\alpha)|}
                if starting_position == 'right' and meas_gate == 'ZX':
                    final_base_array = self.load_measurement(meas_data, 'X')
                    #result = function(omega, n_qubits, final_base_array)
                    result = self.fourier_transform(final_base_array,omega,self.n_cycles_total)

                    o_1 = np.sqrt(np.abs(result))
                elif starting_position == 'left' and meas_gate == 'ZX': 
                    final_base_array = self.load_measurement(meas_data, 'X')
                    #result = function(omega, n_qubits, final_base_array)
                    result = self.fourier_transform(final_base_array,omega,self.n_cycles_total)

                    o_2n = np.sqrt(np.abs(result))
                if numerical == False:
                    errors = self.calculate_error_bars(full_data, omega)
                    if starting_position == 'left':
                        errors = errors*(1/o_2n) #normalizing the standard deviation since we're dividing the result by o_1/2n
                    else:
                        errors = errors*(1/o_1)
                    error_array[:,count] = errors
                #calculating O_{\mu}(\omega) for all measurements
                for index in range(len(gate_array)):
                    gate = gate_array[index]
                    final_meas_array = self.load_measurement(meas_data,gate)
                    #result = function(omega,n_qubits,final_meas_array)
                    result = self.fourier_transform(final_meas_array,omega,self.n_cycles_total)
                    if starting_position == 'left':
                        normed = np.abs(result)/o_2n
                    else:
                        normed = np.abs(result)/o_1
                    o_array[index,count] = normed
                if starting_position == 'right':
                    o_array[:,count] = o_array[:,count][::-1]
                    if numerical == False:
                        error_array[:,count] = error_array[:,count][::-1]

        if load_from_file == False:
            self.meas_type = og_meas_type
            self.meas_qubit = og_meas_qubit
        if numerical == True:
            if load_from_file == True:
                return o_array
            else:
                return o_array
        elif numerical == False:
            if load_from_file == True:
                return o_array,error_array, final_meas_matrix, final_unavgd_matrix, params
            else:
                return o_array,error_array, final_meas_matrix, final_unavgd_matrix, circuit

    def calculate_eigenmodes(self,directory, omega = 0, load_from_file = False,numerical = False):
        og_meas_type = self.meas_type
        og_meas_qubit = self.meas_qubit
        isDIR = os.path.isdir(directory)
        if isDIR == True:
            pass
        else:
            os.mkdir(directory)
        isFILE = os.path.isfile(directory + '/' + 'right_ZY.csv')
        if isFILE == True and load_from_file == False:
            answer = input("You are about to overwrite the files in folder {}. Do you want to proceed? (y/n): ".format(directory))
            if answer == 'n':
                print('Aborting program.')
                raise Exception("User blocked overwrite of folder {}. To save files elsewhere, please change the 'directory' parameter.".format(directory))
        o_array = np.zeros([self.n_qubits,4])
        error_array = np.zeros([self.n_qubits,4])
        directions = ['right','left']
        gates = ['ZX', 'ZY']
        count = 0
        if load_from_file == False:
            if numerical == False:
                n_meas_gates_array = np.arange(1,self.n_qubits+1)
                self.meas_qubit = directions
                self.meas_type = gates
                final_meas_matrix, final_unavgd_matrix, circuit = self.run_circuit_new(n_meas_gates_array)
        for starting_position in directions:
            for meas_gate in gates:
                base_str = meas_gate[1]
                gate_array = [base_str]
                for i in range(1,self.n_qubits):
                    prepend = ''
                    for length in range(i):
                        prepend = prepend + 'Z'
                    gate_array.append(prepend+base_str)
                filename = '{}_{}.csv'.format(starting_position,meas_gate)
                if load_from_file == False:
                    final_meas_group = final_meas_matrix[:,:,count]
                    final_unavgd_group = final_unavgd_matrix[:,:,:,count]
                    self.meas_type = meas_gate
                    self.meas_qubit = starting_position
                    self.store_measurement(final_meas_group, final_unavgd_group,directory,filename,n_meas_gates_array, numerical=numerical)
                readname = directory + '/' + filename
                if numerical == False:
                    meas_data, full_data, angles = self.load_run_data(readname, returned_vals = 'angles', printed_vals = None, full = True)
                elif numerical == True:
                    meas_data, angles = self.load_run_data(readname, returned_vals = 'angles', printed_vals = None, full = False)
                
                # calculating \sqrt{|O_1(\omega_\alpha)|}
                print(meas_data)
                if starting_position == 'right' and meas_gate == 'ZX':
                    final_base_array = self.load_measurement(meas_data, 'X')
                    #result = function(omega, n_qubits, final_base_array)
                    result = self.fourier_transform(final_base_array,omega,self.n_cycles_total)
                    if result < 0:
                        print('F_1 is negative', result)
                    o_1 = np.sqrt(np.abs(result))
                elif starting_position == 'left' and meas_gate == 'ZX':
                    final_base_array = self.load_measurement(meas_data, 'X')
                    #result = function(omega, n_qubits, final_base_array)
                    result = self.fourier_transform(final_base_array,omega,self.n_cycles_total)
                    if result < 0:
                        print('F_2n is negative',result)
                    o_2n = np.sqrt(np.abs(result))
                if numerical == False:
                    errors = self.calculate_error_bars(full_data, omega)
                    if starting_position == 'left':
                        errors = errors*(1/o_2n) #normalizing the standard deviation since we're dividing the result by o_1/2n
                    else:
                        errors = errors*(1/o_1)
                    error_array[:,count] = errors
                #calculating O_{\mu}(\omega) for all measurements
                for index in range(len(gate_array)):
                    gate = gate_array[index]
                    final_meas_array = self.load_measurement(meas_data,gate)
                    result = self.fourier_transform(final_meas_array,omega,self.n_cycles_total)
                    if starting_position == 'left':
                        normed = np.abs(result)/o_2n
                    else:
                        normed = np.abs(result)/o_1
                    o_array[index,count] = normed
                if starting_position == 'right':
                    o_array[:,count] = o_array[:,count][::-1]
                    if numerical == False:
                        error_array[:,count] = error_array[:,count][::-1]
                count += 1
        iterator = 0
        for starting_position in directions:
            for meas_gate in gates:
                if meas_gate == 'ZX':
                    style = '-'
                else:
                    style = '--'
                if starting_position == 'left':
                    color = 'r'
                    if numerical == False:
                        plt.errorbar(range(self.n_qubits), o_array[:,iterator],error_array[:,iterator],label = "{}, {}".format(starting_position, meas_gate), linestyle=style, color = color)
                    else:
                        plt.plot(range(self.n_qubits), o_array[:,iterator], label = "{}, {}".format(starting_position, meas_gate), linestyle=style, color = color)
                elif starting_position == 'right':
                    color = 'b'
                    if numerical == False:
                        plt.errorbar(range(self.n_qubits), o_array[:,iterator],error_array[:,iterator],label = "{}, {}".format(starting_position, meas_gate), linestyle=style, color = color)
                    else:
                        plt.plot(range(self.n_qubits), o_array[:,iterator], label = "{}, {}".format(starting_position, meas_gate), linestyle=style, color = color)  
                else:
                    raise Exception("Starting position not recognized. Please choose edge qubit 0 (n_qubits-1) by choosing 'left' ('right')")
                iterator += 1
        if self.eta == None:
            plt.title("θ = {}π, ϕ = {}π".format(angles[0]/np.pi,angles[1]/np.pi))
        else:
            plt.title("θ = {}π, ϕ = {}π, η= {}π".format(angles[0]/np.pi,angles[1]/np.pi, angles[2]/np.pi))
        plt.xlabel("Position of X/Y Measurement Gate in Qubit Chain")
        plt.ylabel(r'Eigenmode wavefunction at $\omega = {:.2f}$'.format(omega))
        plt.legend()
        plt.show()
        self.meas_type = og_meas_type
        self.meas_qubit = og_meas_qubit
        if numerical == True:
            if load_from_file == True:
                return o_array
            else:
                return o_array
        elif numerical == False:
            if load_from_file == True:
                return o_array,error_array
            else:
                return o_array,error_array, circuit


    def calculate_error_bars(self,full_data, omega):
        error_array = np.zeros(self.n_qubits)
        value_array = np.zeros([self.n_qubits, self.n_avgs])
        #for each set of data corresponding to a measurement gate
        for meas in range(self.n_qubits):
            #for each run in that set
            for avg in range(self.n_avgs):
                spec_meas = full_data[:,avg,meas]
                #we calculate the Fourier component at omega
                #result = function(omega,self.n_qubits, spec_meas)
                result = self.fourier_transform(spec_meas,omega,self.n_cycles_total)
                #and store it in our array of Fourier components
                value_array[meas,avg] = np.real(result)
            #after we have a Fourier component for every run in the data for a specific measurement gate
            #we calculate the standard deviation of the Fourier components for the set of runs
            #and add it to our error_array for the measurement gate in question
            error_array[meas] = np.std(value_array[meas,:])
        #before returning the completed error_array we divide it by the square root of the number of averages
        error_array = error_array/np.sqrt(self.n_avgs)
        return error_array