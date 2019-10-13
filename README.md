# prerequisites
1. python 3
1. pytorch >= 0.4.1

# how to use this code

Create a new folder
Clone general network training routines, the code for preprocessing of the dataset, and the experiment code into that folder  
`git clone https://github.com/mkozinski/NetworkTraining_py`  
`git clone https://github.com/mkozinski/WormData`  
`git clone https://github.com/mkozinski/Learning2SegmentWorms`  

To preprocess the dataset,
1. Unzip Ksenia's data (AIY_neuron.zip and AIA\&AIB_neurons.zip) into the WormData directory
1. Enter the WormData directory, run a python notebook, run the whole render_worm_gt notebook;
This generates the ground truth volumes from the .swc files.
1. In the WormData directory, from the shell, run ./split_data.sh; (if needed run chmod +x split_data.sh first); 
This generates the files: trainFiles... testFiles... containing the names of the training and test volumes

To run baseline training   
`cd Learning2SegmentWorms`  
`python "run_v2.py"`  
The training progress is logged to a directory called `log_v1`. It can be plotted in gnuplot:  
a) the epoch-averaged loss on the training data `plot "<logdir>/logBasic.txt" u 1`,  
b) the F1 performance on the test data `plot "<logdir>/logF1Test.txt" u 1`.

The training loss and test performance plots are not synchronised as testing is performed once every 500 epochs.

The networks weights are dumped in the log directories:  
a) the recent network is stored at `<logdir>/net_last.pth`,  
b) the network attaining the highest F1 score on the test set is stored at `<logdir>/net_Test_bestF1.pth`.

To generate prediction for the test set using a trained network run  
`python "segmentTestSet.py"`.  
The name of the file containing the network, and the output directory are defined at the beginning of the script.
The output is saved in form of numpy volumes, that you can view in the viewSegmentations notebook.


