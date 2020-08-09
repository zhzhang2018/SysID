# Helper methods for system dynamics
import numpy as np

# Helper method for framing. Puts a data into multiple frames. Handles more general cases.
def framing_helper(data, framelen, interv=1, stride=1, axis=1, offset=0, Nframes=-1):
    # framelen: Length of each frame
    # interv  : Number of samples between neighboring frames (start counting from 1)
    # stride  : How many sample to go between two neighboring samples within a frame
    # axis    : Axis that's being framed
    # Example: Input is data=[1,2,3,4,5,6,7], N=3, interv=2, stride=1, then
    # output would be [ [1,2,3], [3,4,5], [5,6,7] ].
    # This method doesn't do error checking.
    if Nframes < 0:
        Nframes = int(np.ceil( (data.shape[axis] - (framelen-1)*stride - offset) / interv ))
    set_inds = [slice(None)]*data.ndim # https://stackoverflow.com/questions/42656930/numpy-assignment-like-numpy-take
    take_inds = set_inds[:]
    frames = np.zeros( tuple( [Nframes] + list( data.shape[:axis]) + [framelen] + list(data.shape[axis+1:] ) ) )

    for i in range(Nframes):
        set_inds[axis] = i
        take_inds[axis] = slice( i*interv+offset, i*interv+framelen*stride+offset, stride )
        frames[i] = data[tuple(take_inds)]
    return frames

# Process data into frames
def framing(input_data, output_data, frame_size=1, pred_size=0):
    # Segments input data into numerous training timeframes
    # Arguments:
    # input_data: The training input data that is to be made into overlapping frames. Should be 2D.
    # output_data: The labels of corresponding input data. Will be made into matching frames.
    # Outputs:
    # input_frames: A list of arrays of frames, each in the shape of (Nframes, <2D frame shape>)
    # output_frames: A list of array of outputs corresponding to each frame in input_frames

    Ninputrow = input_data[0].shape[0]
    Noutputrow = output_data[0].shape[0]

    # Generate data by splitting it into successive overlapping frames.
    # Its first dimension is going to be samples. Each 2D sample occupies the 2nd and 3rd.
    # Empty lists to hold the results:
    input_frames = []
    output_frames = []
    # Process each segment using framing_helper 
    for i in range(len(input_data)):
        Nframes = input_data[i].shape[1] - pred_size - frame_size
        input_frames.append( framing_helper(input_data[i], framelen=frame_size, Nframes=Nframes) )
        output_frames.append( framing_helper(output_data[i], framelen=1, offset=frame_size+pred_size-1, Nframes=Nframes) )
        
        # Preemptive error-checking
        if input_frames[-1].shape[0] != output_frames[-1].shape[0]:
            Nframes = min(input_frames[-1].shape[0], output_frames[-1].shape[0])
            input_frames[-1] = input_frames[-1][:Nframes]
            output_frames[-1] = output_frames[-1][:Nframes]
        
    return input_frames, output_frames