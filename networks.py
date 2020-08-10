import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from dynamics_methods import *
# Parent class for neural nets

# Should be able to compile a neural net on its own, and then train it
# by using a system dynamics module inside it to generate training data.
# Should also be able to take in testing data - maybe take in arbitrary dynamics module as well.
# Note: Training history can be saved by https://stackoverflow.com/a/55901240
class NN():
    def __init__(self, input_shape, seed=2020, input_reshape=None, log_dir=None, tensorboard=False, 
                 Nlayer=2, Nneuron=5, learning_rate=0.001, activation='relu', output_activation='none', optimizer='adam',
                 dynamics=None, frame_size=1, input_mask=[], output_shape=(1,)):
        self.seed = seed
        if log_dir is None:
            self.log_dir = 'logs/fit'
        elif log_dir[0] != '.':
            self.log_dir = log_dir
        else:
            print('Bad path name. Quitting.')
        if tensorboard:
            self.train_callback = [tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=1)]
        else:
            self.train_callback = []
        #    %load_ext tensorboard
        self.Nlayer = Nlayer
        self.Nneuron = Nneuron
        self.learning_rate = learning_rate
        self.model = None
        self.input_shape = input_shape
        if input_reshape is None:
            self.input_reshape = input_shape
        else:
            self.input_reshape = input_reshape
        self.output_shape = output_shape
        
        self.activation = self.set_activation(activation)
        self.output_activation = self.set_activation(output_activation)
        self.optimizer = optimizer
        self.loss = tf.keras.losses.mean_squared_error
        
        #self.construct()
        
        # Dynamics models that generate training and test data
        self.dynamics = dynamics
        self.frame_size = frame_size
        self.frame_size_changed = True
        self.input_mask = input_mask # Controls which variables are observed
        
        # Other variables to log information during training
        self.history = []  # Will be storing dictionaries of histories
        self.pred_history = [] # Stores prediction history
        self.current_iter = 0 # Will increment after each call for train
        self.layers = [] # Keep a list of references to layers
        
    # Will be called by construct(). Only called when a new model is to be created and 
    # the old model (if any) to be erased.
    def train_prep(self):
        #tf.keras.backend.clear_session()
#         if os.path.isfile(self.log_dir):
#             os.rmdir(self.log_dir)
#         else:
#             os.makedirs(self.log_dir)
        try:
            os.system('rm -rf ./'+self.log_dir)
#             os.rmdir(self.log_dir)
#             print('Removed directories')
        except:
#             print('failed to remove directories')
            os.makedirs(self.log_dir)
#             print('Started a new one')
        tf.random.set_seed(self.seed)
        self.history = []
        self.pred_history = []
        self.current_iter = 0
        
    def construct(self):
        self.train_prep()
        
        # Construct model
        Regularizer = tf.keras.regularizers.l1(0.001)
        self.model = tf.keras.Sequential()
        
        # Start with input layer reshaping
        self.model.add( tf.keras.layers.Reshape(self.input_reshape, input_shape=self.input_shape) )
        
        # Add hidden layers
        if type(self.Nneuron) is list:
            nneuron_list = self.Nneuron
        else:
            nneuron_list = [self.Nneuron]*self.Nlayer
        
        for i in range(self.Nlayer):
            self.layers.append( tf.keras.layers.Dense(
                nneuron_list[i], activation = self.activation, kernel_regularizer = Regularizer
            ) )
            self.model.add(self.layers[-1])
        
        # Add output layer ## - assuming single output but I think I've extended it
        self.layers.append( tf.keras.layers.Dense(np.prod(self.output_shape), activation = self.output_activation) )
        self.model.add( self.layers[-1] )
#         print(self.layers[-1].output)
        self.model.add( tf.keras.layers.Reshape(self.output_shape) )

        self.model.compile(optimizer = self.optimizer, loss = self.loss)
        
    
    def train_data_generation_helper(self, inds=[]):
        # Helper method to gather data and make them into framed training data
        if self.frame_size_changed:
#             (self.Inputset, self.Outputset) = self.dynamics.framing(frame_size=self.frame_size)
            (self.Inputset, self.Outputset) = framing(self.dynamics.Inputset, self.dynamics.Outputset, frame_size=self.frame_size)
            self.frame_size_changed = False
        # Train the model and keep track of history
        if len(inds) <= 0:
            Inputset = self.Inputset
            Outputset = self.Outputset
        else:
            (Inputset, Outputset) = self.dynamics.take_dataset(inds)
            (Inputset, Outputset) = framing(input_data=Inputset, output_data=Outputset)
#             (Inputset, Outputset) = self.dynamics.framing(input_data=Inputset, output_data=Outputset)
        # Put all data into one array, so that it could train
        Inputset = np.concatenate(Inputset)
        Outputset = np.concatenate(Outputset)
        # New 0810: Normalize the data. 
        # *Not using sklearn's normalize() or StandardScaler, because normalize() is element-wise, and
        #  both normalize() and StandardScaler only accept 2D inputs
        Inputset, self.input_norm_params = normalize_frame(Inputset)
        Outputset, self.output_norm_params = normalize_frame(Outputset)
        # Mask input data that should remain unseen
        if len(self.input_mask) > 0:
            Inputset = Inputset[:,self.input_mask,:]
        return (Inputset, Outputset)
    
    def train(self, epoch=50, inds=[]):
        # Generate framed data if needed
        (Inputset, Outputset) = self.train_data_generation_helper(inds)
        # Train, and store all relevant data logs
        self.history.append( self.model.fit(
            Inputset, Outputset, epochs=epoch, callbacks=self.train_callback
        ) )
        self.current_iter += 1
    
    # Returns the prediction and correct answer.
    # This method expects you to do one of the followings:
    # 1. Provide Inputset and Outputset and set inds to default (empty list). 
    #    If they are of the right shape (framed & masked), and are normalized, set "processed" to be True.
    # 2. Use data already inside the dynamics by specifying their index in inds
    # Update 0810: Slightly modified the logic flow to allow normalization.
    def test(self, Inputset=None, Outputset=None, inds=[], squeeze=True, processed=False):
        # Data processing etc.
        if len(inds) > 0:
            (Inputset, Outputset) = self.dynamics.take_dataset(inds)
            processed = False
        if not processed:
            (Inputset, Outputset) = framing(input_data=Inputset, output_data=Outputset)
            Inputset = [normalize_frame(inputset, params=self.input_norm_params)[0][:,self.input_mask,:] for inputset in Inputset]
        
        # Do prediction and de-normalize the prediction result
        results = [normalize_frame(self.model.predict(inputset), 
                                   params=self.output_norm_params, reverse=True)[0] for inputset in Inputset]
        
#         if len(inds) <= 0:
#             # Use custom dataset
#             results = [self.model.predict(inputset) for inputset in Inputset]
#         else:
#             # Use existing dataset
#             (Inputset, Outputset) = self.dynamics.take_dataset(inds)
# #             (Inputset, Outputset) = self.dynamics.framing(input_data=Inputset, output_data=Outputset)
#             (Inputset, Outputset) = framing(input_data=Inputset, output_data=Outputset)
#             if len(self.input_mask) > 0:
#                 results = [self.model.predict(inputset[:,self.input_mask,:]) for inputset in Inputset]
#             else:
#                 results = [self.model.predict(inputset) for inputset in Inputset]
#             # The returned Inputset and Outputset are already framed.
        
        # Squeezing reduces unnecessary dimensions.
        if squeeze:
            results = [np.squeeze(result) for result in results]
            Outputset = [np.squeeze(result) for result in Outputset]
            #Inputset = [np.squeeze(result) for result in Inputset]
        
        return results, Outputset, Inputset #, seg_ind_list
    
    # Contains the method above
    def test_and_plot(self, Inputset=None, Outputset=None, inds=[]):
        (results, Outputset) = self.test(Inputset, Outputset, inds)
        
        # Plot
        # To be copied
        fig,axs = plt.subplots(len(results), results[0].shape[0], 
                               constrained_layout=True, figsize = (3*len(results),3*(results[0].shape[0])))
        
        return fig, axs
        
    def set_activation(self, act_func):
        if act_func == 'relu':
            return tf.keras.activations.relu
        elif act_func == 'tanh':
            return tf.keras.activations.tanh
        else:
            return None
            
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
    
    def set_loss(self, loss_func):
        self.loss = loss_func
        
    def get_loss_history(self):
        # Return the loss history over epoch
        hist = []
        for i in range(self.current_iter):
            hist += self.history[i].history['loss']
        return hist
    
    def plot_loss_history(self, log=False, axs=None):
        plt.clf()
        losses = self.get_loss_history()
        if log:
            losses = np.log(losses)
        if axs is None:
            l = plt.plot(np.arange(1,len(losses)+1,1), losses)
            plt.title('Loss history')
        else:
            l = axs.plot(np.arange(1,len(losses)+1,1), losses)
        return l
    
    def weights(self):
        for l in self.layers:
            print('Layer:')
            print(l.weights)
    
    def summary(self):
        self.model.summary()
    
    def save(self, save_dir='models', filename='model'):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.model.save(save_dir+'/'+filename)
        
    def load(self, file_dir):
        # Would get back the model, but would unfortunately lose all other data such as
        # error histories, prediction histories, and pointers to layers.
        self.model = tf.keras.models.load_model(file_dir)
        
    def set_frame_size(self, frame_size):
        self.frame_size = frame_size
        self.frame_size_changed = True
    
    # Note: Calling this function causes the network to reconstruct itself and losing all previously trained stuff.
    def set_input_mask(self, mask):
        self.input_mask = mask
        self.construct()

    
    ## Failed attempts below
    ## 0612-0619 failed attempts at plotting the change of prediction error over time as an animation
    def train_and_store_pred(self, epoch=50, inds=[], pred_history_interval=10):
        # This method stores the prediction during the training.
        (Inputset, Outputset) = self.train_data_generation_helper(inds)
        iter_count = 0
        # Also store the output... if it isn't already stored
        if len(self.pred_history) == 0:
            self.pred_history = [Outputset]
        
        while (iter_count < epoch):
            # Regular stuff that each training function should do
            self.history.append( self.model.fit(
                Inputset, Outputset, epochs=pred_history_interval, callbacks=self.train_callback
            ) )
            iter_count += pred_history_interval
            self.current_iter += 1
            
            # Additional stuff that this function does
            self.pred_history += [ self.model.predict(Inputset) ]
    
    def plot_predictions(self):
        # Plots the prediction history and its error.
        # Consider adding options to choose the loss function later.
        # Might consider adding options to save it as gif later on.
        plt.clf()
        fig,axs = plt.subplots(2, 1, constrained_layout=True, figsize = (4, 6))
        axs[0].set_title('Prediction history')
        axs[1].set_title('Error/Loss history')
        # Ref for modifying plot in-place: https://stackoverflow.com/a/4098938
        line0, = axs[0].plot(np.arange(0,self.pred_history[0].shape[0],1), np.squeeze(self.pred_history[0]))
        line1, = axs[1].plot(np.arange(0,self.pred_history[0].shape[0],1), np.squeeze(self.pred_history[0]))
        # Actually using animation instead, because modifying plot in-place can't reclect the
        # changes in the caller notebook.
        # Ref: https://matplotlib.org/gallery/animation/basic_example.html
        line0_anim = animation.FuncAnimation(fig, self.plot_predictions_helper, self.pred_history[0].shape[0],
                                            fargs=( self.pred_history, line0 ), blit=True)
        line1_anim = animation.FuncAnimation(fig, self.plot_predictions_helper_error, self.pred_history[0].shape[0],
                                            fargs=( self.pred_history ,line1 ), blit=True)
#         for i in range(len(self.pred_history)-1):
#             line0.set_ydata( np.squeeze(self.pred_history[i+1]) )
#             line1.set_ydata( np.square( np.squeeze(self.pred_history[i+1]) -
#                                         np.squeeze(self.pred_history[0])  ) )
#             fig.suptitle('{0}/{1}'.format(i+1, len(self.pred_history)-1))
#         print(os.getcwd()+'/testt.mp4')
#         line0_anim.save(os.getcwd()+'/testt.mp4',fps=10)
        return (fig, axs, line0_anim, line1_anim)
    
    def plot_predictions_helper(self, i, data, line):
        line.set_ydata(np.squeeze(data[i+1]))
        return line,
    
    def plot_predictions_helper_error(self, i, data, line):
        line.set_ydata( np.square( np.squeeze(data[i+1]) -
                                   np.squeeze(data[0])  ) )
        return line,
    
    ## 0612-0619 Batch training attempt - the result wasn't better
    def batch_train(self, max_epoch=200, inds=[], batch_percentage=0.1, tolerance=0.01, random=False):
        # Proposed method of dividing up training data into small batches, and 
        # use up one by one during training.
        # Smaller batch size is the total size times batch_percentage.
        # If the training loss doesn't change by more than tolerance, then consider it converged.
        # If the epoch has exceeded max_epoch, then the model quits (?) instead of keep trying.
        # Maybe consider bringing randomness later. 
        M = 10 # Find the mean over M epochs to see if it has reached convergence 
        
        # Generate framed data if needed
        (Inputset, Outputset) = self.train_data_generation_helper(inds)
        
        # Shuffle data if needed
        if random:
            state = np.random.get_state() # https://stackoverflow.com/q/4601373
            np.random.shuffle(Inputset)
            np.random.set_state(state)
            np.random.shuffle(Outputset)
            
        # Start training
        Nremaining = Inputset.shape[0]
        i = 0
        while (Nremaining > 0):
            # Index stuff
            Nsamples = int(Inputset.shape[0] * batch_percentage)
            if Nremaining <= Nsamples:
                Nsamples = Nremaining
            Nremaining -= Nsamples
            
            first_loss_history = self.model.fit(
                Inputset[:i+Nsamples], Outputset[:i+Nsamples], epochs=1, callbacks=self.train_callback)
            curr_loss = first_loss_history.history['loss'][0]
            prev_loss = curr_loss*2 # So that this initialization passes the while loop criteria
            
            # Start training, repetitively
            iter_count = 0
            # Note: Doesn't account for the possibility where curr_loss suddenly increases. 
            # Criteria for the loop to stop:
            # 1) Loss stops decreasing much (Note that it might be a local minimum), or
            # 2) Max epoch count reached.
            while (prev_loss*(1-tolerance) >= curr_loss and iter_count <= max_epoch):
                loss_history = self.model.fit(
                    Inputset[:i+Nsamples], Outputset[:i+Nsamples], epochs=M, callbacks=self.train_callback)
                iter_count += M
                prev_loss = curr_loss
                curr_loss = np.mean(loss_history.history['loss'][-M:])
                self.history.append(loss_history)
                self.current_iter += 1
                print('Trained for {0} epochs for batch {1}\%'.format(iter_count, 
                                                                        int(100*(1-Nremaining/Inputset.shape[0]))))
            if iter_count <= max_epoch:
                print('Starting next batch because model reached max # of iteration')
            else:
                print('Starting next batch because model converged')
            i += Nsamples
            print('Still needing to train {0}/{1}'.format(Nremaining, Inputset.shape[0]))
    
        
        
# Typical class of dense neural nets that we've been using for Duffing and F=ma
class NN_Dense(NN):
    def __init__(self, dynamics, input_mask, seed=2020, log_dir=None, tensorboard=False,
                 Nlayer=2, Nneuron=5, learning_rate=0.001, activation='relu', output_activation='none', optimizer='adam', 
                 frame_size=1, output_frame_size=1):
        input_shape = (len(input_mask), frame_size)
        input_reshape = (len(input_mask) * frame_size,)
        output_shape = (dynamics.output_d, output_frame_size)
        #print(output_shape)
        super().__init__(input_shape, seed, input_reshape, log_dir, tensorboard, 
                 Nlayer, Nneuron, learning_rate, activation, output_activation, optimizer, 
                         dynamics, frame_size, input_mask, output_shape)
    
    def set_input_mask(self, mask):
        # Our input shape relies on the mask...
        self.input_shape = (len(mask), self.frame_size)
        self.input_reshape = (len(mask) * self.frame_size,)
        super().set_input_mask(mask)

        
### Deprecated old code
## Addition during 0619-0626:
# Class for delay embedding learning
class NN_Delay_Old(NN_Dense): # See embed_dynamics.py for new version
    def __init__(self, dynamics, input_mask, seed=2020, log_dir=None, tensorboard=False,
                 Nlayer=2, Nneuron=5, learning_rate=0.001, activation='relu', output_activation='none', optimizer='adam', 
                 de=1, delay_int=1, sym=False):
        if de <= 0:
            de = self.find_de_via_fnn()
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
    
    # Modify the original data geneation method, but not changing its structure by too much.
    # This in turn modifies the training procedure, because self.train() calls this method in its beginning.
    def train_data_generation_helper(self, inds=[]):
        # Helper method to gather data and make them into framed training data
        if self.frame_size_changed:
#             (self.Inputset, self.Outputset) = self.dynamics.delay_embed(self.delay_int, self.de, symmetric=self.sym)
            (self.Inputset, self.Outputset) = self.dynamics.delay_embed(self.delay_int, self.de, symmetric=self.sym)
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
        (Inputset, Outputset) = self.dynamics.delay_embed(self.delay_int, self.de, inds=inds, Inputset=Inputset, Outputset=Outputset)
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