import tensorflow as tf
import numpy as np
from system_dynamics import System
from networks import NN_Dense
import matplotlib.pyplot as plt
from dynamics_methods import *
from embedding_methods import *

# from sklearn.feature_selection import mutual_info_regression
# from sklearn.metrics import mutual_info_score
# from sklearn.neighbors import KernelDensity

class Delay_System(System):
    # A class that contains a system dynamics, does delay embedding, and interfaces with Delay NN.
    def __init__(self, d, init=None, t0=0, tf=1, dt=0.01, noise=0, pred=0):
        self.pred = pred
        super().__init__(d, init, t0, tf, dt, noise)
    
    def est_delay(self):
        # Delay estimation using AMI: Finds the AMI for increasing delay value,
        # until it finds the first local minima. (Ty et al)
        pass
        
    ## 0714 I'm phasing out the methods below. Refer to embedding_methods.py instead.
    # Functions for delay embedding. Decided to endow this method to this class, instead
    # of the dynamics class.
    # This function is for generating training data, not estimating the optimal delay choice. 
#     def delay_embed(self, dt_int, de, pred=-1, inds=[], Inputset=None, Outputset=None, symmetric=False):
#         # As of now, we assume that:
#         # 1) dt_int is an integer specifying the delay as the number of dts.
#         #    Currently don't support intervals that aren't integer multiples of dt.
#         # 2) de is a provided embedding dimension.
#         #    Might consider implementing FalseNN for automatic de detection later.
#         # 3) This implementation, by default, assumes that:
#         #    - The caller wants to know the embedding at any instant;
#         #    - The caller wants to learn the relationship between embedding and some
#         #      specific output value (not an embedding, but only a slice).
#         #      If the output should be an embedding, create a subclass and overwrite dis.
        
#         # The symmetric flag indicates how the output frames match up with input frames.
#         # As an example, consider input = [1,2,3,4,5] and output = [a,b,c,d,e],
#         # with delay = 1 and dimension = 3.
#         # If symmetric is False: 
#         #     We'll have input as [[1,2,3], [2,3,4], [3,4,5]], and output as [c,d,e].
#         # If symmetric is True:
#         #     We'll have input as [[1,2,3], [2,3,4], [3,4,5]], and output as [b,c,d].
#         # The "offset" value also decides other stuff, such as how much time into the future
#         # would the dataset want to predict. 
#         if pred < 0:
#             pred = self.pred
#         if symmetric:
#             offset = (de-1 - (de//2))*dt_int +pred
#         else:
#             offset = (de-1)*dt_int +pred
        
#         if Inputset is None:
#             Inputset = self.Inputset
#         if Outputset is None:
#             Outputset = self.Outputset
#         if len(inds) <= 0:
#             inds = range(len(Inputset))
#         dembed_in = []
#         dembed_out = []
#         for i in inds:
#             dembed_in.append( self.framing_helper(Inputset[i], de, stride=dt_int) )
#             # If outputset is going to only include a scalar value for each frame...
#             # Then keep the same stride, and our input would have to start from the first scalar instead.
#             dembed_out.append( self.framing_helper(Outputset[i], 1, stride=dt_int, offset=offset) )
#                               #, Nframes=dembed_in[-1].shape[0]) )
#             # Check if they have the same dimensions. If not, shrink one of them.
#             Nframes = min( dembed_in[-1].shape[0], dembed_out[-1].shape[0] )
#             dembed_in[-1] = dembed_in[-1][:Nframes]
#             dembed_out[-1] = dembed_out[-1][:Nframes]
#             print(offset, pred, Nframes, dembed_in[-1].shape[0], dembed_out[-1].shape[0])
#         return (dembed_in, dembed_out)

#     def find_delay_from_MI(self, data, max_delay=100, method='mir', Nbin=50):
#         # Future modification: Run this for each state in data, and return each state's mutual info.
#         if len(data.shape) == 1:
#             data = data.reshape(-1,1)
#         else:
#             data = data.T # Ending shape should be (Nsamples, Nfeatures) for mutual_info_regression and KernelDensity
#         prev_cor = self.AMI(data, data[:,0], method=method, Nbin=Nbin)*100
# #         prev_cor = mutual_info_regression(data, data[:,0])[0]*100
#         local_min_found = False
#         local_min_delay = 0

#         for t in range(1, max_delay+1):
#             # Currently only works with 1D data...
#             x1 = data[:-t]
#             x2 = data[t:,0]
#             curr_cor = self.AMI(x1, x2, method=method, Nbin=Nbin)
# #             curr_cor = mutual_info_regression(x1, x2)
# #             curr_cor = curr_cor[0]

#             if curr_cor >= prev_cor:
#                 local_min_found = True
#             if curr_cor < prev_cor and not local_min_found:
#                 local_min_delay = t

#             prev_cor = curr_cor

#         return local_min_delay, local_min_found
        
#     def AMI(self, X, Y, method=None, Nbin=50, ep=1e-10):
#         # Helper function for finding AMI (Average Mutual Information) for two given data, assuming scalar
#         if method == 'kernel':
#             # Construct grids for estimation
#             grid_step1 = (np.amax(X) - np.amin(X)) / Nbin
#             grid1 = np.arange(np.amin(X), np.amax(X), grid_step1)
#             grid_step2 = (np.amax(Y) - np.amin(Y)) / Nbin
#             grid2 = np.arange(np.amin(Y), np.amax(Y), grid_step2)
#             Xo, Xd = np.meshgrid(grid1, grid2)
            
#             # Find the pdf from the object. The matrix operations are learned from the examples of:
#             # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html
#             skkde_12 = KernelDensity(kernel='gaussian', bandwidth=min(grid_step1, grid_step2)).fit(np.hstack((X, Y)))
            
#             # Get probability. Notice the "T" after the end of vstack.
#             prob12 = skkde_12.score_samples( np.vstack( (Xo.ravel(), Xd.ravel()) ).T ).reshape(Xo.shape)
#             # sklearn's KernelDensity also automatically uses log. We also add ep to avoid zero probability.
#             prob12 = np.exp(prob12) * grid_step1 * grid_step2 + ep # Is this the right way?
            
#             # Compute the entropies and marginal probs
#             H12 = np.sum( prob12 * np.log(prob12) )
#             prob1 = np.sum( prob12, axis=0 )
#             H1 = np.sum( prob1 * np.log(prob1) )
#             prob2 = np.sum( prob12, axis=1 )
#             H2 = np.sum( prob2 * np.log(prob2) )
            
#             # Get MI value
#             MI = H12 - H1 - H2
            
#         elif method == 'hist':
#             # Histogram fixed bin version
#             # Source: https://stackoverflow.com/a/20505476
#             H, edge_Xo, edge_Xd = np.histogram2d( X, Y, bins=Nbin )
#             MI = mutual_info_score(None, None, contingency=(H+ep))
            
#         else:
#             # Default method: Scikit-learn, continuous method
#             # Notice that X has to be 2D and Y has to be 1D in this case...
#             MI = mutual_info_regression( X, Y.ravel() )[0]
#         return MI
        
#     def MMI(self):
#         # Multivariate Mutual Information
#         # Maybe not now... too many degrees of freedom here, and too hard to interpret.
#         # And definitely not here!
#         # https://en.wikipedia.org/wiki/Multivariate_mutual_information
#         pass
    
class LorenzD(Delay_System):
    def __init__(self, init=None, t0=0, tf=1, dt=0.01, noise=0, 
                 sigma=16, rho=45.92, beta=4, pred=0):
        super().__init__(3, init, t0, tf, dt, noise, pred)
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        # Returns z(t) in the Abarbanel (1994) paper
        self.output_d = 1
    
    # Implementing Duffing-oscillator-specific dynamics and outputs
    def dynamics(self, x, t, u):
        # Arguments: x = [x,y,z]^T, t = current time.
        # Output: dx/dt
        # Assumes scalar position. Lorenz doesn't accept inputs u(t).
        return np.array([self.sigma * (x[1] - x[0]), 
                         x[0] * (self.rho - x[2]) - x[1], 
                         x[0] * x[1] - self.beta * x[2]  ])
    
    def output(self, x):
        # Note that this input x depends on the Inputset, and might contain time.
        # Here, x = [t, x,y,z]. Different context from the dynamics method. 
        # The original Abarbanel paper only returned z(t).
        return x[3]
    
    # Telling the model how to plot the dataset
    def plot_dataset(self, title='Training data', plot_skip_rate=1):
        # Arguments:
        # plot_skip_rate: Only plot a point for every this number of samples
        plt.clf()
        Nseg = len(self.u_func_list)
        Ncol = 2
        fig,axs = plt.subplots(Nseg, Ncol, constrained_layout=True, figsize = (12, 3*Nseg), squeeze=False)

        for i in range(Nseg):
            axs[0][i].plot(self.Inputset[i][0,::plot_skip_rate], self.Inputset[i][1,::plot_skip_rate])
            axs[0][i].set_title('x(t)')
            axs[0][i].set_xlabel('t')
            axs[1][i].plot(self.Inputset[i][0,::plot_skip_rate], self.Inputset[i][3,::plot_skip_rate])
            axs[1][i].set_title('z(t)')
            axs[1][i].set_xlabel('t')
        fig.suptitle(title)
        return (fig, axs)

class LorenzFullStateD(LorenzD):
    def __init__(self, init=None, t0=0, tf=1, dt=0.01, noise=0, 
                 sigma=16, rho=45.92, beta=4, pred=0):
        super().__init__(init, t0, tf, dt, noise, sigma, rho, beta, pred)
        # d = 3 for full state
        self.output_d = 3
    
    def output(self, x):
        return x[1:] # Returns full state
    
    # Telling the model how to plot the dataset
    def plot_dataset(self, title='Training data', plot_skip_rate=1):
        # Arguments:
        # plot_skip_rate: Only plot a point for every this number of samples
        plt.clf()
        Nseg = len(self.u_func_list)
        Nrow = 3
        fig,axs = plt.subplots(Nrow, Nseg, constrained_layout=True, figsize = (12, 3*Nseg), squeeze=False)

        for i in range(Nseg):
            axs[0][i].plot(self.Inputset[i][0,::plot_skip_rate], self.Inputset[i][1,::plot_skip_rate])
            axs[0][i].set_title('x(t)')
            axs[0][i].set_xlabel('t')
            axs[1][i].plot(self.Inputset[i][0,::plot_skip_rate], self.Inputset[i][2,::plot_skip_rate])
            axs[1][i].set_title('y(t)')
            axs[1][i].set_xlabel('t')
            axs[2][i].plot(self.Inputset[i][0,::plot_skip_rate], self.Inputset[i][3,::plot_skip_rate])
            axs[2][i].set_title('z(t)')
            axs[2][i].set_xlabel('t')
        fig.suptitle(title)
        return (fig, axs)

class GlycolyticFullState(Delay_System):
    # Dynamics source: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0119821, eq(19)
    # Default parameters give a nice cycle in dynamics
    def __init__(self, init=None, t0=0, tf=1, dt=0.01, noise=0, 
                 params=None, output_inds=None, pred=0):
        super().__init__(7, init, t0, tf, dt, noise, pred)
        if params is None:
            self.params = [ 2.5, 100, 6, 16, 100, 1.28, 12, # J0, k1-k6
                            1.8,  13, 4, 0.52, 0.1,  1, 4]  # k, kappa, q, K1, psi, N, A
        elif len(params) == 14:
            self.params = params
        if output_inds is None or len(output_inds) == 0:
            self.output_inds = [0,1,2,3,4,5,6] # Full state observation by default
            self.output_d = 7
        else:
            self.output_inds = output_inds
            self.output_d = len(output_inds)
    
    # Implementing Glycolytic dynamics and outputs
    def dynamics(self, x, t, u):
        # Arguments: x = (Nstates, _)-shaped array, t = current time.
        # Output: dx/dt
        # Assumes scalar position. Currently doesn't accept inputs u(t).
        sp = self.params
        k1_to_q = sp[1] * x[0] * x[5] / ( 1 + np.power(x[5]/sp[10], sp[9]) )
        k2_to_5 = sp[2] * x[1] * (sp[-2] - x[4])
        k3_to_6 = sp[3] * x[2] * (sp[-1] - x[5])
        return np.array([sp[0] - k1_to_q,
                         2 * k1_to_q - k2_to_5 - sp[6] * x[1] * x[4],
                         k2_to_5 - k3_to_6,
                         k3_to_6 - sp[4] * x[3] * x[4] - sp[8] * (x[3] - x[6]),
                         k2_to_5 - sp[4] * x[3] * x[4] - sp[6] * x[1] * x[4],
                         -2 * k1_to_q + 2 * k3_to_6 - sp[5] * x[5],
                         sp[-3] * sp[8] * (x[3] - x[6]) - sp[7] * x[6] ])
    
    def output(self, x):
        # Note that this input x depends on the Inputset, and might contain time.
        # Here, x = [t, S1, ..., S7]. Different context from the dynamics method. 
        return x[ [i+1 for i in self.output_inds] ]
    
    # Telling the model how to plot the dataset
    def plot_dataset(self, title='Training data', plot_skip_rate=1):
        # Arguments:
        # plot_skip_rate: Only plot a point for every this number of samples
        plt.clf()
        Nseg = len(self.u_func_list)
        Ncol = 2
        fig,axs = plt.subplots(Nseg, Ncol, constrained_layout=True, figsize = (12, 3*Nseg), squeeze=False)

        # Due to the nature of the given model, we plot S1, S2, S6 in one plot, and the rest in the other.
        for i in range(Nseg):
            axs[i][0].plot(self.Inputset[i][0,::plot_skip_rate], self.Inputset[i][1,::plot_skip_rate])
            axs[i][0].plot(self.Inputset[i][0,::plot_skip_rate], self.Inputset[i][2,::plot_skip_rate])
            axs[i][0].plot(self.Inputset[i][0,::plot_skip_rate], self.Inputset[i][6,::plot_skip_rate])
            axs[i][0].set_title('S1, S2, S6(t)')
            axs[i][0].set_xlabel('t')
            axs[i][0].legend(['S1', 'S2', 'S6'])
            axs[i][1].plot(self.Inputset[i][0,::plot_skip_rate], self.Inputset[i][3,::plot_skip_rate])
            axs[i][1].plot(self.Inputset[i][0,::plot_skip_rate], self.Inputset[i][4,::plot_skip_rate])
            axs[i][1].plot(self.Inputset[i][0,::plot_skip_rate], self.Inputset[i][5,::plot_skip_rate])
            axs[i][1].plot(self.Inputset[i][0,::plot_skip_rate], self.Inputset[i][7,::plot_skip_rate])
            axs[i][1].set_title('S3, S4, S5, S7(t)')
            axs[i][1].set_xlabel('t')
            axs[i][1].legend(['S3', 'S4', 'S5', 'S7'])
        fig.suptitle(title)
        return (fig, axs)

class CoupledRosslerFullState(Delay_System):
    # Dynamics source: http://pdfs.semanticscholar.org/51d2/36c73d53df834a5d77584bab72be940c0220.pdf, eq(3)
    # from Buccaletti et al. 2002
    def __init__(self, init=None, t0=0, tf=1, dt=0.01, noise=0, 
                 params=None, output_inds=None, pred=0):
        if init is None:
            init = np.array([0.1,0.2,0.3,0,0,15,-20])
        super().__init__(7, init, t0, tf, dt, noise, pred)
        if params is None:
            self.params = [ 0.925, 0.008 ]  # omega, epsilon
        elif len(params) == 2:
            self.params = params
        if output_inds is None or len(output_inds) == 0:
            self.output_inds = [0,1,2,3,4,5,6] # Full state observation by default
            self.output_d = 7
        else:
            self.output_inds = output_inds
            self.output_d = len(output_inds)
    
    def dynamics(self, x, t, u):
        # Arguments: x = (Nstates, _)-shaped array, t = current time.
        # Output: dx/dt
        # Assumes scalar position. Currently this system doesn't accept inputs u(t).
        sp = self.params
        return np.array([- sp[0] * x[1] - x[2] + sp[1] * (x[3]-x[0]),
                         sp[0] * x[0] + 0.15 * x[1],
                         0.2 + x[2] * (x[0] - 10),
                         x[6] + 0.25 * x[3] + x[5] + sp[1] * (x[0]-x[3]),
                         3 + x[4] * x[6],
                         -0.5 * x[4] + 0.05 * x[5],
                         -x[3] - x[4] ])
    
    def output(self, x):
        # Note that this input x depends on the Inputset, and might contain time.
        # Here, x = [t, S1, ..., S7]. Different context from the dynamics method. 
        return x[ [i+1 for i in self.output_inds] ]
    
    # Telling the model how to plot the dataset
    def plot_dataset(self, title='Training data', plot_skip_rate=1):
        # Arguments:
        # plot_skip_rate: Only plot a point for every this number of samples
        plt.clf()
        Nseg = len(self.u_func_list)
        Ncol = 4
        fig,axs = plt.subplots(Nseg, Ncol, constrained_layout=True, figsize = (12, 3*Nseg), squeeze=False)

        # Due to the nature of the given model, we plot S1, S2, S6 in one plot, and the rest in the other.
        for i in range(Nseg):
            axs[i][0].plot(self.Inputset[i][0,::plot_skip_rate], self.Inputset[i][1,::plot_skip_rate])
            axs[i][0].plot(self.Inputset[i][0,::plot_skip_rate], self.Inputset[i][2,::plot_skip_rate])
            axs[i][0].set_title('S1/x1, S2/y1(t)')
            axs[i][0].set_xlabel('t')
            axs[i][0].legend(['S1/x1', 'S2/y1'])
            axs[i][1].plot(self.Inputset[i][0,::plot_skip_rate], self.Inputset[i][7,::plot_skip_rate])
            axs[i][1].plot(self.Inputset[i][0,::plot_skip_rate], self.Inputset[i][4,::plot_skip_rate])
            axs[i][1].set_title('S7/w2, S4/x2(t)')
            axs[i][1].set_xlabel('t')
            axs[i][1].legend(['S7/w2', 'S4/x2'])
            axs[i][2].plot(self.Inputset[i][0,::plot_skip_rate], self.Inputset[i][3,::plot_skip_rate])
            axs[i][2].plot(self.Inputset[i][0,::plot_skip_rate], self.Inputset[i][5,::plot_skip_rate])
            axs[i][2].set_title('S3/z1, S5/y2(t)')
            axs[i][2].set_xlabel('t')
            axs[i][2].legend(['S3/z1', 'S5/y2'])
            axs[i][3].plot(self.Inputset[i][0,::plot_skip_rate], self.Inputset[i][6,::plot_skip_rate])
            axs[i][3].set_title('S6/z2(t)')
            axs[i][3].set_xlabel('t')
            axs[i][3].legend(['S6'])
        fig.suptitle(title)
        return (fig, axs)

class RikitakeFullState(Delay_System):
    # Source: https://link.springer.com/content/pdf/10.1140/epjst/e2015-02481-0.pdf, eq.(4)
    # Hyperchaotic
    def __init__(self, init=None, t0=0, tf=1, dt=0.01, noise=0, 
                 params=None, output_inds=None, pred=0):
        super().__init__(5, init, t0, tf, dt, noise, pred)
        if params is None:
            self.params = [ 1, 1, 0.7, 1.1, 0.1 ] # a, b, c, p, q
        elif len(params) == 5:
            self.params = params
        if output_inds is None or len(output_inds) == 0:
            self.output_inds = [0,1,2,3,4] # Full state observation by default
            self.output_d = 5
        else:
            self.output_inds = output_inds
            self.output_d = len(output_inds)
    
    # Implementing Glycolytic dynamics and outputs
    def dynamics(self, x, t, u):
        # Arguments: x = (Nstates, _)-shaped array, t = current time.
        # Output: dx/dt
        # Assumes scalar position. Currently doesn't accept inputs u(t).
        sp = self.params
        k1_to_q = sp[1] * x[0] * x[5] / ( 1 + np.power(x[5]/sp[10], sp[9]) )
        k2_to_5 = sp[2] * x[1] * (sp[-2] - x[4])
        k3_to_6 = sp[3] * x[2] * (sp[-1] - x[5])
        return np.array([ -sp[0] * x[0] + x[1] * x[2] - sp[3] * x[3] + sp[4] * x[4],
                          -sp[0] * x[1] + x[0] * (x[2] - sp[1]) - sp[3] * x[3] + sp[4] * x[4],
                          1 - x[0] * x[1], 
                          sp[2] * x[1], 
                          sp[4] * (x[0] + x[1] + x[3])
                        ])
    
    def output(self, x):
        # Note that this input x depends on the Inputset, and might contain time.
        # Here, x = [t, S1, ..., S7]. Different context from the dynamics method. 
        return x[ [i+1 for i in self.output_inds] ]
    
    # Telling the model how to plot the dataset
    def plot_dataset(self, title='Training data', plot_skip_rate=1):
        # Arguments:
        # plot_skip_rate: Only plot a point for every this number of samples
        plt.clf()
        Nseg = len(self.u_func_list)
        Ncol = 5
        fig,axs = plt.subplots(Nseg, Ncol, constrained_layout=True, figsize = (15, 3*Nseg), squeeze=False)

        # Due to the nature of the given model, we plot S1, S2, S6 in one plot, and the rest in the other.
        for i in range(Nseg):
            for j in range(5):
                axs[i][j].plot(self.Inputset[i][0,::plot_skip_rate], self.Inputset[i][j+1,::plot_skip_rate])
                axs[i][j].set_title('x{0}(t)'.format(j+1))
                axs[i][j].set_xlabel('t')
        fig.suptitle(title)
        return (fig, axs)
    
    
    
    
# Class for delay embedding learning
# Use these as networks paired up with Delay_System.
# Modified on top of the old version of NN_Delay_Old in networks.py
class NN_Delay(NN_Dense):
    def __init__(self, dynamics, input_mask, seed=2020, log_dir=None, tensorboard=False,
                 Nlayer=2, Nneuron=5, learning_rate=0.001, activation='relu', output_activation='none', optimizer='adam', 
                 de=-1, delay_int=-1, sym=False):
#         if de <= 0:
#             de = self.find_de_via_fnn()
#         if isinstance(dynamics, Delay_System):
#             print('Warning: It is suggested that you use a Delay_System object as dynamics, even though there is\
#                   no difference between the two as of now.')
        self.delay_int = delay_int
        self.de = de
        self.sym = sym
        super().__init__(dynamics, input_mask, seed, log_dir, tensorboard,
                 Nlayer, Nneuron, learning_rate, activation, output_activation, optimizer, frame_size=de)
        # Its super calculates the shapes for inputs and outputs of the network, and nothing else.
        # Shouldn't need to do anything more than this regading the parent class methods. 
    
    # Method that finds delay embedding dimension via false NN. 
    # To be implemented (or borrowed) later.
    def find_de_via_fnn(self):
        return 1
    
    # Modified the original data geneation method, but not changing its structure by too much.
    # This in turn modifies the training procedure, because self.train() calls this method in its beginning.
    # Helper method to gather data and make them into framed training data
    def train_data_generation_helper(self, inds=[]):
        # First step: Using the available data, find the best delay and dimension
        if self.delay_int <= 0:
#             ######TODO: Modify this. Make it work for all datasets. etc.etc. #######
#             (self.delay_int, MI_success) = find_delay_from_MI(
#                 self.Inputset[0][:,self.input_mask,:], max_delay=20, method='mir', Nbin=50)
            (self.delay_int, MI_success) = find_delay(
                self.Inputset[0][:,self.input_mask,:], max_delay=20, method='mi', MImethod='mir', Nbin=50, 
                ep=1e-10, start_delay=1, end_early=True, verbose=False)
            if not MI_success:
                print('Warning: Didn\'t converge when trying to find best delay from mutual information.')
        
        if self.frame_size_changed:
#             (self.Inputset, self.Outputset) = delay_embed(self.delay_int, self.de, symmetric=self.sym)
            (self.Inputset, self.Outputset) = delay_embed(
                self.delay_int, self.de, self.dynamics.Inputset, self.dynamics.Outputset, self.dynamics.pred, symmetric=self.sym)
            self.frame_size_changed = False
        # Train the model and keep track of history
        if len(inds) <= 0:
            Inputset = self.Inputset
            Outputset = self.Outputset
        else:
            (Inputset, Outputset) = ( [self.Inputset[i] for i in inds], [self.Outputset[i] for i in inds] )
        # Put all data into one array, so that it could train
        Inputset = np.concatenate(Inputset)
        Outputset = np.concatenate(Outputset)
        # Mask input data that should remain unseen
        if len(self.input_mask) > 0:
            Inputset = Inputset[:,self.input_mask,:]
        return (Inputset, Outputset)
    
    # Modify the original test method, because we need to embed the input data...
    # If using external input, this method expects that input to follow the same input structure
    # as the ones stored in the model. 
    def test(self, Inputset=None, Outputset=None, inds=[], squeeze=True):
#         (Inputset, Outputset) = delay_embed(self.delay_int, self.de, inds, Inputset, Outputset)
        (Inputset, Outputset) = delay_embed(self.delay_int, self.de, self.dynamics.Inputset, self.dynamics.Outputset, self.dynamics.pred)
        if len(self.input_mask) > 0:
            results = [self.model.predict(inputset[:,self.input_mask,:]) for inputset in Inputset]
        else:
            results = [self.model.predict(inputset) for inputset in Inputset]
        
        # Squeezing reduces unnecessary dimensions.
        if squeeze:
            results = [np.squeeze(result) for result in results]
            Outputset = [np.squeeze(result) for result in Outputset]
            #Inputset = [np.squeeze(result) for result in Inputset]
        
        return results, Outputset, Inputset
    
    # Because delay embedding dimension is directly tied with the frame_size field in parent class,
    # disable the original set method, and write a new one. 
    def set_de(self, de):
        self.de = de
        self.frame_size = de
        self.frame_size_changed = True
    def set_frame_size(self, frame_size):
        pass
    # Because the data generation also relies on delay value, abuse the flag frame_size_changed
    # and mark the change as well.
    def set_delay_int(self, dt_int):
        self.delay_int = dt_int
        self.frame_size_changed = True
    

# Parent class for NNs that use FNN-related methods during training.
# Used to be abstract, but I find it too cumbersome.
class NN_FNN(NN_Dense):
    def __init__(self, dynamics, input_mask, ratio=10, stop_threshold=0, max_tau=20, max_de=10, verbose=False, 
                 fnn_ind=0, FNNtype='kennel', uniform_delay=True, # The last argument is only for Cao and Kennel.
                 inverse=True, local_max=True, twoD=False, # Only for Garcia. inverse also appears for Kennel.
                 seed=2020, log_dir=None, tensorboard=False,
                 Nlayer=2, Nneuron=5, learning_rate=0.001, activation='relu', output_activation='none', optimizer='adam'):
        self.verbose=verbose
        # It is assumed that the dynamics object comes with its own data already generated.
        # Also assumes that the first dataset in Inputset is all the data we would use for finding the embedding.
        # Also assumes that the hyperparameters are given, supervised.
        self.FNNtype = FNNtype
        self.ratio = ratio
        self.stop_threshold = stop_threshold
        self.max_tau = max_tau
        self.max_dim = max_de
        
        if len(input_mask) <= 0:
            input_mask = [i+1 for i in range(dynamics.d)] # Because the first row of Inputset is always time
        self.input_mask = input_mask
        self.dynamics = dynamics
        
        self.uniform_delay = uniform_delay
        self.inverse=inverse
        self.local_max=local_max
        self.twoD=twoD
        
        # Call the fnn method and obtain some returned values.
        # self.js stores the variable index for each delay embedding dimension, while self.ts stores the corresponding
        # delay values (accumulatively). self.FNNargs stores all the rest.
        (self.js, self.tts, self.FNNargs) = self.find_de_via_fnn(fnn_ind)
        
        self.de = len(self.js) + len(input_mask)
        # Below I'll hack the meaning of "input_mask"...
        # You see, in the parent class, the length of "input_mask" is directly tied to the neural net input shape.
        # The shape was seen as (len(input_mask),self.de) in that implementation, because it's expecting
        # uniform delay for all the observation variables, and input_mask indicates which variables are observed.
        # Here, the meaning of input dimension has significantly changed... Our input is now of the shape 
        # (Nfeatures+len(js), 1), as indicated by the Garcia paper. 
        # Because all other uses of "input_mask" was overwritten by the methods below, we can safely hack it.
        input_mask_fake = [i for i in range(self.de)]
        super().__init__(dynamics, input_mask_fake, seed, log_dir, tensorboard,
                 Nlayer, Nneuron, learning_rate, activation, output_activation, optimizer, frame_size=1)
        # Its super calculates the shapes for inputs and outputs of the network, and nothing else.
        # Rectify the mistaken input mask here
        self.input_mask = input_mask
    
    def find_de_via_fnn(self, fnn_ind=0):
        return find_delay(self.dynamics.Inputset[fnn_ind][self.input_mask,:], 
                          method='FNN', FNNmethod=self.FNNtype, 
                          min_delay=1, max_delay=self.max_tau, max_dim=self.max_dim, end_early=True, verbose=self.verbose, 
                          ratio=self.ratio, pred=self.dynamics.pred, stop_threshold=self.stop_threshold, 
                          uniform_delay=self.uniform_delay, inverse=self.inverse, local_max=self.local_max, twoD=self.twoD)
#         if self.FNNtype.lower() = 'garcia':
#             js, ts, args = find_delay_by_FNN_Garcia(self.dynamics.Inputset[fnn_ind][self.input_mask,:], 
#                                         ratio=self.ratio, pred=self.dynamics.pred, stop_threshold=self.stop_threshold, 
#                                         min_tau=1, max_tau=self.max_tau, max_dim=self.max_dim, 
#                                         init_i=0, end_early=True, verbose=self.verbose, 
#                                         inverse=self.inverse, local_max=self.local_max, twoD=self.twoD)
#             tts = np.cumsum(ts) # Cumulative delay values
#             return js, tts, args
#         elif self.FNNtype.lower() = 'cao':
#             js, ttau, args = find_delay_by_FNN_Cao(self.dynamics.Inputset[fnn_ind][self.input_mask,:], 
#                                      pred=self.dynamics.pred, min_tau=1, max_tau=self.max_tau, max_dim=self.max_dim, 
#                                      uniform_delay=self.uniform_delay,
#                                      init_i=0, end_early=True, inverse=self.inverse, verbose=self.verbose)
#             # Return ttau[1:], because the method includes an extra 0 at the start of it.
#             return js, ttau[1:], args
#         else:
#             # Default Kennel method
#             js, ttau, args = find_delay_by_FNN_Kennel(self.dynamics.Inputset[fnn_ind][self.input_mask,:], 
#                                      ratio=self.ratio, pred=self.dynamics.pred, stop_threshold=self.stop_threshold, 
#                                      min_tau=1, max_tau=self.max_tau, max_dim=self.max_dim, 
#                                      init_i=0, end_early=True, verbose=self.verbose)
    
    def delay_embed_fnn(self, inputsets):
        if self.FNNtype.lower() == 'garcia':
            return delay_embed_Garcia(self.js, self.tts, 
                                  inputsets, self.dynamics.Outputset, self.dynamics.pred, Timeset=self.dynamics.Timeset)
        else:
            return delay_embed_Cao(self.FNNargs, 
                                  inputsets, self.dynamics.Outputset, self.dynamics.pred, Timeset=self.dynamics.Timeset)
    
#     # Implement this method in child classes for generating the embedded version of the provided dataset.
#     # Keeps "inputset" as an argument, because we need to apply mask to self.Inputset.
#     def delay_embed_fnn(self, inputset):
#         pass
#     # Thank goodness Python's abstract methods only need the name to be the same, and don't check the signatures.
#     # Because different FNN approaches could have different arguments for this delay_embed method.
    
    # Modified the original data geneation method, but not changing its structure by too much.
    # This in turn modifies the training procedure, because self.train() calls this method in its beginning.
    # Helper method to gather data and make them into framed training data.
    # Notice that, UNLIKE other helper methods, this one has to do the input masking earlier.
    def train_data_generation_helper(self, inds=[]):
        if self.frame_size_changed:
            inputsets = [inputset[self.input_mask,:] for inputset in self.dynamics.Inputset]
            (self.Inputset, self.Outputset, self.TrainingTimeset) = self.delay_embed_fnn(
                                                                                         inputsets)
            self.frame_size_changed = False
        # Train the model and keep track of history
        if len(inds) <= 0:
            Inputset = self.Inputset
            Outputset = self.Outputset
        else:
            (Inputset, Outputset) = ( [self.Inputset[i] for i in inds], [self.Outputset[i] for i in inds] )
        # Put all data into one array, so that it could train
        Inputset = np.concatenate(Inputset)
        Outputset = np.concatenate(Outputset)
        if self.verbose:
            print('Inputset size = {0}; outputset size = {1}'.format(Inputset.shape, Outputset.shape))
            print('Input set is masked by ', self.input_mask)
        # Mask input data that should remain unseen <-- This step became unnecessary after we moved it to the front.
#         if len(self.input_mask) > 0:
#             Inputset = Inputset[:,self.input_mask,:]
        return (Inputset, Outputset)
    
    # Modify the original test method, because we need to embed the input data...
    # __I changed the embedding method and this method to allow it to return time data in frames__
    # __Also because Garcia FNN has a different embedding__
    def test(self, inds=[], squeeze=True):
        inputsets = [inputset[self.input_mask,:] for inputset in self.dynamics.Inputset]
        (Inputset, Outputset, Timeset) = self.delay_embed_fnn(
                                                              inputsets)
        results = [self.model.predict(inputset) for inputset in Inputset]
        
        # Squeezing reduces unnecessary dimensions.
        if squeeze:
            results = [np.squeeze(result) for result in results]
            Outputset = [np.squeeze(result) for result in Outputset]
            Timeset = [np.squeeze(result) for result in Timeset]
        
        if self.verbose:
            print('Dimensions: Outputset = {0}, results = {1}'.format(Outputset[0].shape, results[0].shape))
        return results, Outputset, Inputset, Timeset
     
    # Because delay embedding dimension is directly tied with the frame_size field in parent class,
    # disable the original method that could arbitrarily set it. 
    def set_frame_size(self, frame_size):
        pass

class NN_Garcia(NN_Dense):
    def __init__(self, dynamics, input_mask, ratio=10, stop_threshold=0, max_tau=20, max_de=10, verbose=False, fnn_ind=0,
                 inverse=True, local_max=True, twoD=False,
                 seed=2020, log_dir=None, tensorboard=False,
                 Nlayer=2, Nneuron=5, learning_rate=0.001, activation='relu', output_activation='none', optimizer='adam'):
        super().__init__(dynamics, input_mask, ratio, stop_threshold, max_tau, max_de, verbose, fnn_ind, 'garcia',
                         inverse, local_max, twoD,
                         seed, log_dir, tensorboard, Nlayer, Nneuron, learning_rate, activation, output_activation, optimizer)

class NN_Cao(NN_FNN):
    def __init__(self, dynamics, input_mask, max_tau=20, max_de=10, verbose=False, fnn_ind=0, 
                 seed=2020, log_dir=None, tensorboard=False,
                 Nlayer=2, Nneuron=5, learning_rate=0.001, activation='relu', output_activation='none', optimizer='adam'):
        super().__init__(dynamics, input_mask, 10, 0, max_tau, max_de, verbose, fnn_ind, 'cao', 
                         True, True, False, seed, log_dir, tensorboard, 
                         Nlayer, Nneuron, learning_rate, activation, output_activation, optimizer)

# # Checkpoint

# Subclass for delay embedding with nonuniform delays identified by False Nearest Neighbor methods
# class NN_Garcia(NN_FNN):
#     def __init__(self, dynamics, input_mask, ratio=10, stop_threshold=0, max_tau=20, max_de=10, verbose=False, fnn_ind=0, 
#                  inverse=True, local_max=True, twoD=False,
#                  seed=2020, log_dir=None, tensorboard=False,
#                  Nlayer=2, Nneuron=5, learning_rate=0.001, activation='relu', output_activation='none', optimizer='adam'):
#         self.inverse=inverse
#         self.local_max=local_max
#         self.twoD=twoD
#         super().__init__(dynamics, input_mask, ratio, stop_threshold, max_tau, max_de, verbose, fnn_ind, 
#                         seed, log_dir, tensorboard, Nlayer, Nneuron, learning_rate, activation, output_activation, optimizer)
    
#     def find_de_via_fnn(self, fnn_ind=0):
#         js, ts, args = find_delay_by_FNN_Garcia(self.dynamics.Inputset[fnn_ind][self.input_mask,:], 
#                                         ratio=self.ratio, pred=self.dynamics.pred, stop_threshold=self.stop_threshold, 
#                                         min_tau=1, max_tau=self.max_tau, max_dim=self.max_dim, 
#                                         init_i=0, end_early=True, verbose=self.verbose, 
#                                         inverse=self.inverse, local_max=self.local_max, twoD=self.twoD)
#         tts = np.cumsum(ts) # Cumulative delay values
#         return js, tts, args
    
#     def delay_embed_fnn(self, inputsets):
#         return delay_embed_Garcia(self.js, self.tts, 
#                                   inputsets, self.dynamics.Outputset, self.dynamics.pred, Timeset=self.dynamics.Timeset)

# class NN_Cao(NN_FNN):
#     def __init__(self, dynamics, input_mask, max_tau=20, max_de=10, verbose=False, fnn_ind=0, 
#                  seed=2020, log_dir=None, tensorboard=False,
#                  Nlayer=2, Nneuron=5, learning_rate=0.001, activation='relu', output_activation='none', optimizer='adam'):
#         super().__init__(dynamics, input_mask, 10, 0, max_tau, max_de, verbose, fnn_ind, seed, log_dir, tensorboard, 
#                          Nlayer, Nneuron, learning_rate, activation, output_activation, optimizer)
    
#     def find_de_via_fnn(self, fnn_ind=0):
#         js, ttau, args = find_delay_by_FNN_Cao(self.dynamics.Inputset[fnn_ind][self.input_mask,:], 
#                                      pred=self.dynamics.pred, min_tau=1, max_tau=self.max_tau, max_dim=self.max_dim, 
#                                      init_i=0, end_early=True, verbose=self.verbose)
#         # Return ttau[1:], because the method includes an extra 0 at the start of it.
#         return js, ttau[1:], args
    
#     def delay_embed_fnn(self, inputsets):
#         return delay_embed_Cao(self.FNNargs, 
#                                   inputsets, self.dynamics.Outputset, self.dynamics.pred, Timeset=self.dynamics.Timeset)
    
    
# # Subclass for delay embedding with nonuniform delays identified by False Nearest Neighbor methods
# class NN_Garcia(NN_Dense):
#     def __init__(self, dynamics, input_mask, ratio=10, stop_threshold=0, max_tau=20, max_de=10, verbose=False, fnn_ind=0, 
#                  seed=2020, log_dir=None, tensorboard=False,
#                  Nlayer=2, Nneuron=5, learning_rate=0.001, activation='relu', output_activation='none', optimizer='adam'):
#         self.verbose=verbose
#         # It is assumed that the dynamics object comes with its own data already generated.
#         # Also assumes that the first dataset in Inputset is all the data we would use for finding the embedding.
#         # Also assumes that the hyperparameters are given, supervised.
#         self.ratio = ratio
#         self.stop_threshold = stop_threshold
#         self.max_tau = max_tau
#         self.max_dim = max_de
        
#         if len(input_mask) <= 0:
#             input_mask = [i+1 for i in range(dynamics.d)] # Because the first row of Inputset is always time
#         self.input_mask = input_mask
#         self.dynamics = dynamics
#         (self.js, self.ts, self.Fs) = self.find_de_via_fnn(fnn_ind)
#         self.tts = np.cumsum(self.ts) # Cumulative delay values
#         self.de = len(self.js) + len(input_mask)
#         # Below I'll hack the meaning of "input_mask"...
#         # You see, in the parent class, the length of "input_mask" is directly tied to the neural net input shape.
#         # The shape was seen as (len(input_mask),self.de) in that implementation, because it's expecting
#         # uniform delay for all the observation variables, and input_mask indicates which variables are observed.
#         # Here, the meaning of input dimension has significantly changed... Our input is now of the shape 
#         # (Nfeatures+len(js), 1), as indicated by the Garcia paper. 
#         # Because all other uses of "input_mask" was overwritten by the methods below, we can safely hack it.
#         input_mask_fake = [i for i in range(self.de)]
#         super().__init__(dynamics, input_mask_fake, seed, log_dir, tensorboard,
#                  Nlayer, Nneuron, learning_rate, activation, output_activation, optimizer, frame_size=1)
#         # Its super calculates the shapes for inputs and outputs of the network, and nothing else.
#         # Rectify the mistaken input mask here
#         self.input_mask = input_mask
    
#     def find_de_via_fnn(self, fnn_ind=0):
#         return find_delay_by_FNN_Garcia(self.dynamics.Inputset[fnn_ind][self.input_mask,:], 
#                                         ratio=self.ratio, pred=self.dynamics.pred, stop_threshold=self.stop_threshold, 
#                                         min_tau=1, max_tau=self.max_tau, max_dim=self.max_dim, 
#                                         init_i=0, end_early=True, verbose=self.verbose)
    
#     # Modified the original data geneation method, but not changing its structure by too much.
#     # This in turn modifies the training procedure, because self.train() calls this method in its beginning.
#     # Helper method to gather data and make them into framed training data.
#     # Notice that, UNLIKE other helper methods, this one has to do the input masking earlier.
#     def train_data_generation_helper(self, inds=[]):
#         if self.frame_size_changed:
#             inputsets = [inputset[self.input_mask,:] for inputset in self.dynamics.Inputset]
#             (self.Inputset, self.Outputset, self.TrainingTimeset) = delay_embed_Garcia(
#                     self.js, self.tts, inputsets, self.dynamics.Outputset, self.dynamics.pred, Timeset=self.dynamics.Timeset)
#             self.frame_size_changed = False
#         # Train the model and keep track of history
#         if len(inds) <= 0:
#             Inputset = self.Inputset
#             Outputset = self.Outputset
#         else:
#             (Inputset, Outputset) = ( [self.Inputset[i] for i in inds], [self.Outputset[i] for i in inds] )
#         # Put all data into one array, so that it could train
#         Inputset = np.concatenate(Inputset)
#         Outputset = np.concatenate(Outputset)
# #         if self.verbose:
# #             print('Inputset size = {0}; outputset size = {1}'.format(Inputset.shape, Outputset.shape))
#         # Mask input data that should remain unseen
# #         if len(self.input_mask) > 0:
# #             Inputset = Inputset[:,self.input_mask,:]
#         return (Inputset, Outputset)
    
#     # Modify the original test method, because we need to embed the input data...
#     # __I changed the embedding method and this method to allow it to return time data in frames__
#     # __Also because Garcia FNN has a different embedding__
#     def test(self, inds=[], squeeze=True):
#         inputsets = [inputset[self.input_mask,:] for inputset in self.dynamics.Inputset]
#         (Inputset, Outputset, Timeset) = delay_embed_Garcia(
#                     self.js, self.tts, inputsets, self.dynamics.Outputset, self.dynamics.pred, Timeset=self.dynamics.Timeset)
#         results = [self.model.predict(inputset) for inputset in Inputset]
        
#         # Squeezing reduces unnecessary dimensions.
#         if squeeze:
#             results = [np.squeeze(result) for result in results]
#             Outputset = [np.squeeze(result) for result in Outputset]
#             Timeset = [np.squeeze(result) for result in Timeset]
        
#         if self.verbose:
#             print('Dimensions: Outputset = {0}, results = {1}'.format(Outputset[0].shape, results[0].shape))
#         return results, Outputset, Inputset, Timeset
     
#     # Because delay embedding dimension is directly tied with the frame_size field in parent class,
#     # disable the original method that could arbitrarily set it. 
#     def set_frame_size(self, frame_size):
#         pass
    