import numpy as np
import numpy.ma as ma
from IPython.display import clear_output
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import pickle
from functools import partial
import itertools
import time
from multiprocessing import Process, Queue, Pool
import multiprocessing as mp
from numpy import random



def init_voterModel_params(nodes=10, k_nearest=4, p_reconnection=0.2, p_groups=0.5, r_L=0.5, 
                           r_H=0.5, cue_positions="random", vote_distribution= 'Kao', 
                           max_steps=1000, update_strategy = 'single', m = None):
    '''Inputs: 
    nodes............. number of nodes in the watts strogatz graph
    k_nearest......... number of neighbors a node is connected to
    p_reconnection.... probability to replace neighbor connection with random connection
    p_groups.......... probability a node gets the independent cue
    r_L............... reliability of the independent cue
    r_H............... reliability of the correlated cue
    cue_positions..... which nodes receive which cues: 'random', 'block', 'equal', 'choose_m' (only if vote distribution is 'random'
    vote_distribution. possible values: 
                            'Kao': distribute votes according to Kao model
                            'random': votes are drawn from a Bernoulli distribution with p= p_groups
    max_steps......... maximum number of interactions between neighbors
    update_strategy... possible values: 
                             'all-sync': each node takes the average over all neighbors votes, 
                             changes its vote to 1 to if the average is positive, to -1 if 
                             the average is negative and stays the same if the average is 0. 
                             Update is synchronous for all nodes.
                             'all-async': Same as above but asynchronous updates 
                             'single': A randomly chosen node changes its opinion to that of a 
                             randomly chosen neighbor. classic voter model
    '''
    params = {}
    params['nodes'] = nodes
    params['k_nearest'] = k_nearest
    params['p_reconnection'] = p_reconnection
    params['p_groups'] = p_groups
    params['r_L'] = r_L
    params['r_H'] = r_H
    params['cue_positions'] = cue_positions
    params['vote_distribution'] = vote_distribution
    params['max_steps'] = max_steps
    params['update_strategy'] = update_strategy
    params['m'] = m
    
    return params


def print_params(params): 
    ''' prints parameters from the params dictionary'''
        
    print("PARAMS:\n")
    print("----------------------------------------")
    for key in params: 
        if type(params[key]) == np.ndarray:
            print('{:<20} {:<8} '.format(key, 'numpy array of shape ' + str(params[key].shape)))
        else:
            print('{:<20} {:<8} '.format(key, str(params[key])))
    print("----------------------------------------")

    
def split_seq(seq, size):
    '''
    Inputs: 
    seq............. a list to be split in sublists of approximately equal size
    size............ desired length of sublist
    
    Output: 
    newseq.......... list of sublists
    
    Functionality: 
    this fuction takes a list and splits it in sublists of legth approximately 'size'. This function is 
    used by initialize_votes() to genarete equal spacing of cues. 
    cf: http://code.activestate.com/recipes/425397-split-a-list-into-roughly-equal-sized-pieces/
    '''
    newseq = []
    splitsize = 1.0/size*len(seq)
    for i in range(size):
            newseq.append(seq[int(round(i*splitsize)):int(round((i+1)*splitsize))])
    return newseq
    
def initialize_votes(params): 
    '''
    Inputs: 
    params.......... dictionary containing the model parameters
    
    Outputs: 
    votes........... numpy array containing the initial votes (-1 or 1) for each node in the network
    node_id......... numpy array conatining the id of the cue providing the initial information (1 = independent, 0 = correlated)
    
    Functionality: 
    this function creates the nodes initail votes based on the model parameters p, rL, rH 
    and the placement specified in params['cue_positions']
    ''' 
    n = params['nodes']
    p = params["p_groups"]
    rL = params['r_L']
    rH = params['r_H']

    votes = np.zeros(n)
    corr_cue = 2*int(np.random.rand() <rH)-1
    
    if params['cue_positions'] == 'random': 
        # recievers of independent and correlated information are placed randomly within the network
        get_ind_cue_idx = np.random.rand(n) < p
        get_corr_cue_idx = ~get_ind_cue_idx
        n_ind = len(get_ind_cue_idx[get_ind_cue_idx==True])
    
    elif params['cue_positions'] == 'block': 
        # first p % of nodes receive correlated information, all others receive independent information
        n_corr = len(np.where(np.random.rand(n) >= p)[0])
        get_corr_cue_idx = np.arange(0,n_corr, 1)
        get_ind_cue_idx = np.arange(n_corr, n, 1)
        n_ind = n-n_corr
        
    elif params['cue_positions'] == 'equal':        
        # recievers of independent and correlated information are pllaced at (approximately) even distances 
        n_corr = len(np.where(np.random.rand(n) >= p)[0]) 
        n_ind = n-n_corr                
        if n_corr == 0:
            get_corr_cue_idx = np.array([]).astype(int)
        else: 
            get_corr_cue_idx = [sublist[0] for sublist in split_seq(np.arange(n), n_corr)]
        
        get_ind_cue_idx = list(set(range(n)).difference(set(get_corr_cue_idx)))

    else: 
        print("Unknown cue distribution! \nValid distributions are: 'random', 'block', 'equal' and 'choose_m'")
        return
    
    votes[get_ind_cue_idx] = 2*(np.random.rand(n_ind) < rL).astype(int) -1
    votes[get_corr_cue_idx] = corr_cue

    node_id = np.zeros(n)
    node_id[get_ind_cue_idx] = 1
    
    return votes, node_id
    
    



def initial_vote_nocorr(params): 
    '''
    The initial votes are distributed according to a beroulli distribution 
    here a node votes positive with p = params['p_groups']. 
    The distribution of positive/ negative voters can be random, block or equal
    '''

    n = params['nodes']
    p = params["p_groups"]    
    votes = 2*(np.random.rand(n) < p).astype(int)-1
    
    if params['cue_positions'] == 'random': 
        pass    
    elif params['cue_positions'] == 'block': 
        votes = np.sort(votes)
    elif params['cue_positions'] == 'choose_m': 
        votes = -1*np.ones(n)
        m = params['m']
        set_pos = np.random.choice(np.arange(0, n, 1), m, replace = False)
        votes[set_pos] =1
    else: 
        print('not a valid cue distr')
        return    
    return votes



def assign_votes(G, params): 
    '''
    Inputs: 
    G................ a watts strogatz graph generated using the model parameters
    params........... dictionary containing the model parameters
    
    Output: 
    G................ The network which now contains the initial votes and cue IDs as attribute
    
    Functionality: 
    Generate initial votes using (initialize_votes()) and assign them to the graph as a node attribute. 
    If the votes are generated via the Kao model we also save whether the vote was 
    drawn from the correlated or the individual cue. This proberty is stored in the 
    node attribute cue_id. If the Kao model is not unsed these values are set to nan. 
    '''
    n = params['nodes']
    if params['vote_distribution'] == "Kao":
        votes, cid = initialize_votes(params)

    else: 
        votes = initial_vote_nocorr(params)
        cid = np.zeros(len(votes))
        cid[:] = np.nan
        
    vote_dict = {}
    cue_id_dict = {}
    for nn in range(n):
        vote_dict[nn] = votes[nn]
        cue_id_dict[nn] = cid[nn]
        
    nx.set_node_attributes(G, vote_dict, 'vote')
    nx.set_node_attributes(G, cue_id_dict, 'cue_id')
    return G


def generate_positions(N, radius = 1): 
    ''' Input: 
        N...........number of nodes
        radius......radius of the circle where nodes are placed
        
        Output: 
        pos: position of each node equidistantly spaced along a
        circle with given radius
        
        Functionality: 
        Creates evenly spaced positions of nodes along a circle for plotting purposes
        '''
    
    pos = np.zeros((2, N))    
    for n in range(N): 
        alpha = 2*np.pi*n/float(N)
        pos[0, n] = radius*np.cos(alpha)
        pos[1, n] = radius*np.sin(alpha)
    return pos


def create_network(params, plot = True): 
    '''
    Inputs:
    params........... dictionary containing the model parameters
    plot............. boolean variablie ndicating whether the network is supposed to be potted
    
    Output: 
    G................ A networkx Graph object 
    
    Functionality: 
    create a watts strogatz graph using the modified networkx function my_watts_strogatz_graph() which also allows for 
    odd values of nearest neighbors. After creating the graph initial votes are assigned as attributes. 
    '''
    
    n = params['nodes']
    k = params['k_nearest']
    p = params['p_reconnection']
    pos = generate_positions(n)
    pos_dict = {}
    
    for nn in range(n):
        pos_dict[nn] = pos[:, nn]

    if k > n-1: 
        G = my_watts_strogatz_graph(n, n-1, p)
    else: 
        G = my_watts_strogatz_graph(n, k, p)     
        
        
    nx.set_node_attributes(G, pos_dict, 'position')
    G = assign_votes(G, params)
    
    if plot:
        #nx.draw(G, pos.T)
        #plt.show()
        plot_network(G)
    return G


def identify_changable_nodes(votes, Adj_matrix): 
    """ Iterates through all nodes in random order and checks wheter further change is possible 
    i.e if a majority of the node's neighbores have an opposing opinion. 
    Inputs: 
    votes...................... list of the nodes' individual votes
    Adj_matrix................. numpy array. The networks adjacency matrix
    Output: 
    changable_nodes............ list containing all nodes which can potentially change their opinion
    """
    node_order =  np.arange(len(votes))# test allnode_orderes in random order
    np.random.shuffle(node_order)
    changable_nodes = []
    
    for node in node_order: 
        vote = votes[node]
        neighbor_mean = np.mean(ma.array(votes, mask = 1-Adj_matrix[node, :]))
        neighbor_vote = np.sign(neighbor_mean)
        
        if vote*neighbor_vote < 0: 
            changable_nodes.append(node)
        
    return changable_nodes
        

def is_it_stuck(votes, Adj_matrix): 
    """ Checks if the network is stuck by identifying the list of all potentially changable nodes. 
    Returns true if this list is empty."""
    changable_nodes = identify_changable_nodes(votes, Adj_matrix)
    if len(changable_nodes) == 0: 
        return True
    else: 
        return False
    

def update_votes_all_sync(G, params, record=False): 
    print("update startegy is deprecated. Pleas use 'update_votes_all_sync' (for threshold voter model) or 'update_votes_single' (for classic voter model)")
    return
    max_steps = params['max_steps']
    votes = np.array(list(nx.get_node_attributes(G,'vote').values()))
    selector = np.array(nx.adjacency_matrix(G).todense().astype(bool))
    steps = 0
    #save = []
    
    while (abs(np.sum(votes)) < len(votes)) and (steps < max_steps):
        neighbor_mean = np.array([np.mean(votes[selector[k,:]]) for k in range(selector.shape[0])])
        votes[np.where(neighbor_mean > 0)] = 1 
        votes[np.where(neighbor_mean < 0)] = -1 
        steps += 1      
        
        if steps%n == 0: 
        # every couple of steps check wheter it makes sense to continue. 
        # The time between checks is proportional to the number of nodes 
        # to account for the fact that larger networks will take longer. 
            stop = is_it_stuck(votes, A)
            if stop: 
                steps = max_steps+1
      
        #if (steps == 1):
        #    first_round = np.mean(votes)
        
        #if (steps > 10) and  (steps < 21):
            # typically the network is decided after <10 steps. 
            # if this is not the case we save the mean votes to determine why the network got stuck 
           # save.append(np.mean(votes)) 
            
        #if steps == 21: 
        #    test1 = sum(np.diff(abs(np.diff(np.array(save))))) #circling
        #    test2 = sum(np.diff(np.array(save))) # constant value
        #    if test2 == 0:
        #        if record:
        #            return np.mean(votes), max_steps + 1, first_round, votes
        #        else: 
        #            return np.mean(votes), max_steps + 1, first_round
        #    if test1 == 0:
        #        if record:
        #            return np.mean(votes), max_steps + 2, first_round, votes
        #        else:
         #           return np.mean(votes), max_steps + 2, first_round
    
    #if steps == 0:
    #    first_round = np.mean(votes)
        
    if record:    
        return np.mean(votes), steps, votes
    else: 
        return np.mean(votes), steps
    

def update_votes_all_async(G, params, record=False): 
    ''' Inputs:
    G................... networkx object representing a Watts Strogatz graph
                         initial votes are stored in the attributes
    params.............. dictionary containing the model parameters
    record.............. if True an array containing all individual votes is returned
    
    Outputs: 
    np.mean(votes)...... arithmetic mean over all inidvidual votes if consensus is reached this is either 1 or -1, 
                         otherwise its a value between -1 and 1
    steps............... number of steps needed to reach consensus bounded by the 'max_steps' parameter. 
                         If the network gets stuck the fucntion returns max_steps +1, 
                         if the network oscillates between a finite number of values (CIRCLING) the fucntion returns max_steps +2 
    first_round......... mean accuracy after the first round of vote updates
    votes............... only returned if record= True, array containing all individual votes
    
    Functionality: 
    Votes are changed using the asynchronous threshold model without tie-break. A node is picked at random and the mean over the votes of its neighbors is calculated. If the mean is positive the node's vote is set to 1, if the mean is negative the noe'd vote is set to -1, if the mean is 0 no change is made. To avoid extreme running times after 210 steps(value determined by previous simulations) it is checked if the updating process is actually changing the vote distribution. If it is determined that the network is stuck or oscillating updating is stopped.
    
    '''
    
    max_steps = params['max_steps']
    n = params['nodes']

    votes = np.array(list(nx.get_node_attributes(G,'vote').values()))
    steps = 0
    #save = []
    
    while (abs(np.sum(votes)) < len(votes)) and (steps < max_steps):
    #while (votes) and (steps < max_steps):
        
        node = np.random.choice(np.arange(0, n, 1), 1)[0]
        neighbor_mean = np.mean(votes[list(G.neighbors(node))])
        if neighbor_mean > 0: 
            votes[node] = 1
        elif neighbor_mean < 0: 
            votes[node] = -1
        steps += 1      
        
        if steps%n == 0: 
        # every couple of steps check wheter it makes sense to continue. 
        # The time between checks is proportional to the number of nodes 
        # to account for the fact that larger networks will take longer. 
            stop = is_it_stuck(votes, np.array(nx.adjacency_matrix(G).todense()))
            if stop and abs(sum(votes))<n: 
                steps = max_steps+1       #if (steps == 1):
        #    first_round = np.mean(votes)
        
        #if (steps > 210) and  (steps < 221):
            # THIS BREAKING CONDITION IS NOT VERY GOOD AND DOES NOT WORK FOR NETWORKS LARGER THAN 50 NODES
            # typically the network is decided after <10 steps. 
            # if this is not the case we save the mean votes to determine why the network got stuck 
            #save.append(np.mean(votes)) 
        #if steps == 221: 
        #    test1 = sum(np.diff(abs(np.diff(np.array(save))))) #circling
         #   test2 = sum(np.diff(np.array(save))) # constant value
          #  if test2 == 0:
           #     if record:
            #        return np.mean(votes), max_steps + 1, first_round, votes
            #    else: 
            #        return np.mean(votes), max_steps + 1, first_round
           # if test1 == 0:
            #    if record:
             #       return np.mean(votes), max_steps + 2, first_round, votes
              #  else:
            #        return np.mean(votes), max_steps + 2, first_round

            
    #if steps == 0:
     #   first_round = np.mean(votes)
        
    if record:    
        return np.mean(votes), steps, votes
    else: 
        return np.mean(votes), steps

        
def update_votes_single(G, params, record=False): 
    ''' Inputs:
    G................... networkx object representing a Watts Strogatz graph
                         initial votes are stored in the attributes
    params.............. dictionary containing the model parameters
    record.............. if True an array containing all individual votes is returned
    
    Outputs: 
    np.mean(votes)...... arithmetic mean over all inidvidual votes if consensus is reached this is either 1 or -1, 
                         otherwise its a value between -1 and 1
    steps............... number of steps needed to reach consensus bounded by the 'max_steps' parameter. 
    first_round......... mean accuracy after the first round of vote updates
    votes............... only returned if record= True, array containing all individual votes
    
    Functionality: 
    Votes are changed using the voter model. A node is picked at random and it's vote is changed to match that of one of its randomly chosen neighbors. This method is guaranteed to converge, therefore it does not require a break condition like in the cases of threshold updating methods.    
    '''
    max_steps = params['max_steps']
    n = params['nodes']

    votes = np.array(list(nx.get_node_attributes(G,'vote').values()))
    steps = 0
    while (abs(np.sum(votes)) < len(votes)) and (steps < max_steps):

        node = np.random.choice(np.arange(0, n, 1), 1)[0]
        neighbor = np.random.choice(list(G.neighbors(node)), 1)[0]

        votes[node] = votes[neighbor]
        steps += 1      

        #if (steps == 1):
         #   first_round = np.mean(votes)

    #if steps == 0:
     #   first_round = np.mean(votes)

    if record:    
        return np.mean(votes), steps, votes
    else: 
        return np.mean(votes), steps

def connections_between_non_equals(G): 
    ''' Inputs:
    	G................... networkx object representing a Watts Strogatz graph
                         initial votes are stored in the attributes
	Outputs
    	frac_uneq_cues...... fraction of edges connecting nodes who received cues of different type (independent/correlated)
    	frac_uneq_votes..... fraction of edges connecting nodes who voted for different options (independent of cue type)
	
	Functionality: 
	this function calcualtes the fraction of edges connecting nodes that received cues from different sources (indenpendent/correlated)
	or voted differently. This This serves as a proxy for the clustering of opinions and can predict how likely a anetwork is to reach consensus.
	'''
    votes = np.array(list(nx.get_node_attributes(G, 'vote').values()))
    cueID = np.array(list(nx.get_node_attributes(G, 'cue_id').values()))
    A = nx.adj_matrix(G).todense()
    if len(cueID) == 0: # if no cueID is recorded e.g. when initializationn is done in random fashion
        frac_uneq_cues = np.nan
    elif len(set(cueID)) == 1: #if all cueIDs are the same, the matrc slicing does not work
        frac_uneq_cues = 0
    else: 
        frac_uneq_cues = np.sum(A[np.where(cueID == 1)[0], :][:, np.where(cueID == 0)[0]])*2/np.sum(A)
    
    if len(set(votes)) == 1: #all votes the same 
        frac_uneq_votes = 0
    else: 
        frac_uneq_votes = np.sum(A[np.where(votes == 1)[0], :][:, np.where(votes == -1)[0]])*2/np.sum(A)
    
    return frac_uneq_cues, frac_uneq_votes
    

def single_run(params):     
    '''Input: 
    params........... dictionary containing the model parameters
    
    Outputs:
    vote............. arithmetic mean over all inidividual votes in the network. 
   steps............. number of steps until consensus was reached. If consensus was not reached this value will be 
                     either max_steps +1 (if the network gets stuck) or max_steps +2 (if the network oscillates)
    first_round...... arithmetic mean over all inidividual votes after first updating step
    initial.......... arithmetic mean over all inidividual votes before first updating step
    uneq_cue_edges... fraction of edges connecting nodes receiving the independnet cue with nodes receiving the correlted cue
    uneq_vote_edges.. fractionn of edges connecting nodes of unequal initial votes
    
    Functionality:
    This function unites all steps necessary to create a Watts Strogatz network, initializing the votes, calculating the fraction of interesting edges and updating  the votes until consensus is reached or the network gets stuck. For description of the sub processes refer to the documentation of relevant functions. 
    '''
    G = create_network(params, plot=False)
    votes_init = nx.get_node_attributes(G, 'vote')
    if (np.array(list(votes_init.values()))==0).any(): 
        print('something went wrong!')
        return
        
        
    initial = np.mean(list(votes_init.values()))
    uneq_cue_edges,  uneq_vote_edges = connections_between_non_equals(G)
    if params["update_strategy"] == "all_sync":
        print("currently not working, breaking condition false")
        return
        vote, steps = update_votes_all_sync(G, params, record=False)    
    elif params["update_strategy"] == "all_async":
        vote, steps = update_votes_all_async(G, params, record=False)    
    elif params["update_strategy"] == "single":
        print("currently not working, breaking condition false")
        return
        vote, steps = update_votes_single(G, params, record=False)    
    else: 
        print('update strategy not known')
        return
	
	
    return vote, steps, initial, uneq_cue_edges,  uneq_vote_edges

    
def run_many(params, iters, vals):
    ''' Inputs:
    params ................. dictionary with model parameters
    iters................... number of iterations
    vals.................... list of lists containing the values of the scanning parameters (only for saving)
    
    Output:
    list containing the currently used parameter values and the averaged results of the simulation
    
    Functionality:  
    This fuction makes multiple call to single_run() and saves the results to arrays which are then analysed using organize_output()
    
    '''
    accuracy = np.zeros(iters)
    sats = np.zeros(iters)
    initial_acc = np.zeros(iters)
    #acc_first_update = np.zeros(iters)
    frac_unec_cue_edges = np.zeros(iters)
    frac_unec_vote_edges = np.zeros(iters)
    max_steps = params['max_steps']
    
    for i in range(iters):
        vote, steps, initial, uneq_cue, uneq_vote = single_run(params)
        initial_acc[i] = int(initial>0)
        #acc_first_update[i] = int(first_round>0)
        accuracy[i] = int(vote>0)
        sats[i] = steps
        frac_unec_cue_edges[i] = uneq_cue
        frac_unec_vote_edges[i] = uneq_vote
        
    mean_list = organize_output(initial_acc, accuracy, sats, max_steps, frac_unec_cue_edges,frac_unec_vote_edges)

    return list(vals) + mean_list

def mean_without_warning(array): 
    ''' this function calcualtes the mean of a numpy array and returns NaN if the array is empty. 
    This is used only to surpress warings'''
    if len(array) == 0: 
        return np.nan
    else: 
        return np.nanmean(array)
                
def organize_output(intial, accuracy, sats, max_steps, uneq_cue_edges, uneq_vote_edges): 
    ''' 
    Inputs: 
        arrays containing the results of several simulations calculated using run_many(), 
        refer run_many() documentation for more details
        
    Outputs: 
        list of interesting means, refer to inline comments
    '''

    mean_init = np.mean(intial) # mean initial vote (before updating)
    #mean_first = np.mean(first) # mean vote after first round of upating
    
    # mean accuracy over all trials (independent of wether the network reached consensus
    mean_acc = np.mean(accuracy) 
    # mean accuracy over all trials where consensus was reached
    mean_sat_acc = mean_without_warning(accuracy[np.where(sats < max_steps)[0]]) 
    # mean accuracy over all trials where consensus was not reached
    mean_not_sat_acc = mean_without_warning(accuracy[np.where(sats >= max_steps)[0]]) 
    # mean accuracy over all trials where the network got stuck
    mean_acc_stuck = mean_without_warning(accuracy[np.where(sats == max_steps + 1)[0]])
    # mean accuracy over all trials where the network is oscillatig 
    mean_acc_circ = mean_without_warning(accuracy[np.where(sats == max_steps + 2)[0]])
    
    percent_sat = float(len(sats[sats < max_steps]))/ len(sats) # percent of networks that reached consensus
    percent_stuck = float(len(sats[sats == max_steps +1]))/ len(sats) # percent of networks that got stuck
    percent_circ = float(len(sats[sats == max_steps +2]))/ len(sats) # percent of networks that oscillated
    mean_sat = mean_without_warning(sats[sats < max_steps]) # mean saturation time for those networks which converged
    
    # average fraction of edges connecting nodes receiving different cues
    mean_frac_uneq_cue_edges = mean_without_warning(uneq_cue_edges) 
    # average fraction of edges connecting nodes with opposing initial votes
    mean_frac_uneq_vote_edges = mean_without_warning(uneq_vote_edges)

    return [mean_init, mean_acc, mean_sat_acc, mean_not_sat_acc, mean_acc_stuck, mean_acc_circ, percent_sat, percent_stuck, percent_circ, mean_sat, mean_frac_uneq_cue_edges, mean_frac_uneq_vote_edges]
        

def make_filename(params, iters, varying_params): 
    ''' 
    Inputs: 
    params................ dictionary containing model parameters
    iters................. number of iterations 
    varying_params........ names of parameters to be scanned
    
    Outputs: 
    filename.............. string containing all parameter information, 
                           to be used as filename when saving simulation results
    
    Functionality: 
    A .csv filename is created which contains all important information about a simulation. To be called from scan()
    '''
    filename = 'Simulation_'
    for param_name in params: 
        if param_name in varying_params: 
            filename += param_name + 'VARY_' #mark those parameters that wree scanned
        else: 
            if (type(params[param_name]) == float) or (type(params[param_name]) == int): 
                filename += param_name + str(np.round(params[param_name], 2)) + '_'
            else: 
                filename += param_name  + str(params[param_name]) + '_'
                
    filename = filename + "ITERS{}.csv".format(iters)
    return filename
      

def start_process():
    print('Starting', mp.current_process().name)

def call_calculation(params, param_names, iters, param_vals): 
    '''
    Inputs: 
    params................ dictionary containing model parameters
    param_names........... names of parameters currently scanned 
    iters................. number of iterations 
    param_vals............ cuurent values for the scanning parameters
    
    Output: 
    results............... array containing all values calcualted by run_many() 
    
    Functionality: 
    helper function which is called by scan(). Assigns the current parameter values to the param dictionary which is then passes on to run_many() 
    '''
    p = params.copy()
    for j, pn in enumerate(param_names): 
        p[pn] = param_vals[j] 
    
    result = run_many(p, iters, param_vals)
    return result
    
    
def scan(params, param_names, param_vals, iters=50, filename=''):
    ''' Inputs
    params ................. dictionary with model parameters
    param_names............. list of strings containing the names of parameters to be scanned
    param_vals.............. list of lists containing the values of the scanning parameters
    iters................... number of sample to be calculated per value tuple
    filename................ string for saving the data, if none is given filenmae will be created automatically
        Functionality
    This function creates a list of tuples for all combinations of parameter values. For each tuple 
    all significant measures will be calculated (see dokumentatin and averaged over the specified number of iterations. Insead of returning the results are directly 
    '''

    pool_size = mp.cpu_count()
    pool = Pool(processes=pool_size, initializer=start_process)
    
    call_this_calculation = partial(call_calculation, params, param_names, iters)
    param_list =  list(itertools.product(*param_vals))
    results = pool.map_async(call_this_calculation, param_list)
      
    while (True):
        if (results.ready()): break
        remaining = results._number_left
        print("\rWaiting for {} tasks to complete...".format(remaining), end ='')
        time.sleep(0.5)
        
        
    pool.close() # no more tasks
    pool.join()  # wrap up current tasks

    write2file(params, results.get(), param_names, iters, filename=filename)

     
    
def write2file(params, results, param_names, iters, filename=''):
    ''' Inputs
    params................ dictionary conatining the model parameters
    results............... 
    param_names........... parameters that were scanned across
    filename............... name of the file where data will be saved, if not specified a name will be created from params
    
    This function takes the results of a parameter scanning with scan and writes 
    them to a pandas dataframe which is then saved as csv file.
    '''    
    #    """Outputs: 
#    results[:, 0] ..... groupsize
#    results[:, 1] ..... mean accuracy of initial votes
#    results[:, 2] ..... mean accuracy after first update
#    results[:, 3] ..... mean over all final accuracies (consensus and not)
#    results[:, 4] ..... mean accuracy when converged to consensus
#    results[:, 5] ..... mean accuracy when stuck
#    results[:, 6] ..... mean accuracy when circling
#    results[:, 7] ..... percent of trials converging to consensus
#    results[:, 8] ..... percent of trials getting stuck
#    results[:, 9]...... precent of trials circling
#    results[:, 10]..... mean saturation time for consensus
#    results[:, 11]..... fraction of edges connection unequal cues
#    results[:, 12]..... fraction of edges connection unequal votes
    
    
    
    if len(filename) == 0: #create filename from parameters
        filename = make_filename(params, iters, param_names)
    
    results= np.array(results)
    if results.ndim == 1: 
        results = np.reshape(results, (1, len(results)))
    #params_array = np.array(list(results_array[:, 0] ))
    results_dict = {}
    
    for param_idx, param_name in enumerate(param_names): 
        results_dict[param_name] = results[:, param_idx]
        
        
    result_names = ['accuracy_initial', 'accuracy_all', 'accuracy_consensus', 'acuuracy_no_consensus', 'accuracy_stuck', 'accuracy_circle', 'percent_converged', 'percent_stuck', 'percent_circling', 'mean_ssaturation', 'mean_frac_uneq_cue_edges', 'mean_frac_uneq_vote_edges']
    
    for result_idx, result_name in enumerate(result_names):
        results_dict[result_name] = results[:, param_idx+1+result_idx]

    df = pd.DataFrame(data=results_dict)
    df.to_csv(filename)
    print('saved data to: \n', filename)    

    
    

def visualize(params, filename='example.png'):    
    
    G = create_network(params, plot=False)
    pos = nx.get_node_attributes(G, 'position')
    votes_init = nx.get_node_attributes(G, 'vote')
    cue_ids = nx.get_node_attributes(G, 'cue_id')
    if params["update_strategy"] == "all":
        v, s, fr, votes = update_votes_all(G, params, record = True)
    elif params["update_strategy"] == "random":
        v, s, fr, votes = update_votes_random(G, params, record = True)
    
              
    colors = ['purple', 'green']
    
    titles = ['n = {} , p = {}, r_L = {}, r_H = {} \ninitialization {} convergence\
    after {} iterations'.format(params['nodes'], params['p_groups'], params['r_L'], params['r_H'], params['cue_positions'], s), 
              'n = {} , p = {}, r_L = {}, r_H = {} \ninitialization {} no convergence'.format(params['nodes'], params['p_groups'], params['r_L'], params['r_H'], params['cue_positions'], s)]    
    

    
    plt.figure(figsize=(10,4))
    if s <= params['max_steps']:
        t = plt.suptitle(titles[0], size = 18)
        t.set_position([.5, 1.05])
    else: 
        t = plt.suptitle(titles[1], size = 18)
        t.set_position([.5, 1.05])

    plt.subplot(121)
    nx.draw(G, pos=pos, node_color=[colors[int(votes_init[i]>0)] for i in range(len(votes_init))], node_size =400, alpha = 0.7)
    plt.text(-0.5, -1.5, 'initialization, \nmean vote = {}'.format(np.round(np.mean(list(votes_init.values())), 2)), size =16)
    cue_labels = ['C', 'I']
    for k in range(len(G.nodes)): 
        cue_ids[k] = cue_labels[int(cue_ids[k])]
        
    nx.draw_networkx_labels(G,pos,labels = cue_ids,font_size=16)

    plt.subplot(122)
    nx.draw(G, pos=pos, node_color=[colors[int(votes[i]>0)] for i in range(len(votes))], node_size =400, alpha = 0.7)
    plt.text(-0.5, -1.5, 'final, \nmean vote = {}'.format(v), size =16)


    patch0 = mpatches.Patch(color=colors[0], label='vote -1')
    patch1 = mpatches.Patch(color=colors[1], label='vote  1')
    plt.legend(handles=[patch0, patch1], fontsize =16, bbox_to_anchor=(1.05, 1))
    

    plt.savefig(filename, bbox_inches='tight')          
    plt.show()
    

def scatter_results(results, var_name, figsize=(12, 8), s=14): 
    ''' will crerate a scatter plot from the results list generated by one 
    of the scan functions'''
    
    x = results[:, 0] # groupsize
    init_acc = results[:, 1] # mean accuracy of initial votes
    acc_all = results[:, 3] # mean over all final accuracies (consensus and not)
    acs_sat = results[:, 4] # mean accuracy when converged to consensus
    sat_sat = results[:, -1] #mean saturation time for consensus
    
    acc_min = min([min(init_acc), min(acc_all), min(acs_sat)]) - 0.05
    acc_max = max([max(init_acc), max(acc_all), max(acs_sat)]) + 0.05
    
    plt.figure(figsize=figsize)
    plt.subplot(411)
    plt.scatter(x, init_acc, s=70, c ='b', alpha =0.7)
    plt.ylabel('accuracy', size=s)   
    plt.ylim(acc_min, acc_max)
    plt.title('int acc', size =16)
    plt.xticks([])
                   
    plt.subplot(412)
    plt.scatter(x, acc_all, s=70, c ='b', alpha =0.7)
    plt.ylabel('accuracy', size=s)
    plt.ylim(acc_min, acc_max)
    plt.title('all acc', size =16)
    plt.xticks([])

    
    plt.subplot(413)
    plt.scatter(x, acs_sat, s=70, c ='b', alpha =0.7)
    plt.ylabel('accuracy', size=s)
    plt.ylim(acc_min, acc_max)
    plt.title('sat acc', size =16)
    plt.xticks([])

    
    plt.subplot(414)
    plt.scatter(x, sat_sat, s=70, c='g', alpha =0.7)
    plt.xlabel(var_name, size=s)
    plt.xticks(size =s)
    plt.title('satration time')
    plt.ylabel('saturation time', size =16)
    
    plt.show()    
        
def my_watts_strogatz_graph(n, k, p, seed=None):
    """Return a Watts-Strogatz small-world graph.


    Parameters
    ----------
    n : int
        The number of nodes
    k : int
        Each node is connected to k nearest neighbors in ring topology
    p : float
        The probability of rewiring each edge
    seed : int, optional
        Seed for random number generator (default=None)

    See Also
    --------
    newman_watts_strogatz_graph()
    connected_watts_strogatz_graph()

    Notes
    -----
    First create a ring over n nodes.  Then each node in the ring is
    connected with its k nearest neighbors (k-1 neighbors if k is odd).
    Then shortcuts are created by replacing some edges as follows:
    for each edge u-v in the underlying "n-ring with k nearest neighbors"
    with probability p replace it with a new edge u-w with uniformly
    random choice of existing node w.

    In contrast with newman_watts_strogatz_graph(), the random
    rewiring does not increase the number of edges. The rewired graph
    is not guaranteed to be connected as in  connected_watts_strogatz_graph().

    References
    ----------
    .. [1] Duncan J. Watts and Steven H. Strogatz,
       Collective dynamics of small-world networks,
       Nature, 393, pp. 440--442, 1998.
    """
    if k>=n:
        raise nx.NetworkXError("k>=n, choose smaller k or larger n")
    if seed is not None:
        random.seed(seed)

    G = nx.Graph()
    G.name="watts_strogatz_graph(%s,%s,%s)"%(n,k,p)
    nodes = list(range(n)) # nodes are labeled 0 to n-1
    # connect each node to k/2 neighbors
    if k == n-1: 
        for v in range(n): 
            for w in range(v+1, n): 
                G.add_edge(v,w)
    
    else:
        for j in range(1, k // 2+1):
            targets = nodes[j:] + nodes[0:j] # first j nodes are now last in list
            G.add_edges_from(zip(nodes,targets))

        if k%2 == 1: 
            v1 = random.choice(nodes)
            w1 = (v1 +  k // 2 + 1)%n
            G.add_edge(v1,w1)
            v1_explored =[]
            degs = np.array(list(dict(G.degree).values()))
            while len(np.where((degs) < k)[0]) >= 2 and len(list(set(v1_explored))) <n: 
                v1_explored.append(v1)
                v1 = (v1 + 1)%n            
                w1 = (v1 +  k // 2 + 1)%n
                if degs[v1] < k and degs[w1]<k:
                    G.add_edge(v1,w1)
                degs = np.array(list(dict(G.degree).values()))
        
    # rewire edges from each node
    # loop over all nodes in order (label) and neighbors in order (distance)
    # no self loops or multiple edges allowed
    for j in range(1, k // 2+1): # outer loop is neighbors
        targets = nodes[j:] + nodes[0:j] # first j nodes are now last in list
        # inner loop in node order
        for u,v in zip(nodes,targets):
            if random.random() < p:
                w = random.choice(nodes)
                # Enforce no self-loops or multiple edges
                while w == u or G.has_edge(u, w):
                    w = random.choice(nodes)
                    if G.degree(u) >= n-1:
                        break # skip this rewiring
                else:
                    G.remove_edge(u,v)
                    G.add_edge(u,w)
    return G    
    
def prepare_data_for_plot(filename, variable = 'k_nearest', what_to_plot = 'accuracy_consensus'): 
    if filename[:-1] == 'p':
        df = pickle.load(open(filename, 'rb'))
    else: 
        df = pd.read_csv(filename)
    print(df.columns)
    param = df[variable].values
    plotter = df[what_to_plot].values
    gs = df['nodes'].values
    
    VARs = np.sort(list(set(param)))
    plot_dict = {}
    
    k = 0

    for var in VARs:
        idx = np.where(np.isclose(param,var))[0]
        idxx = np.argsort(gs[idx])
        plot_dict[str(var)] = plotter[idx][idxx]
        
        if k == 0: 
            init = df["accuracy_initial"].values[idx][idxx]
        else: 
            init+= df["accuracy_initial"].values[idx][idxx]
        k +=1 
    sizes = np.sort(gs[idx])  
    plot_dict['sizes'] = sizes
    plot_dict['info'] =  what_to_plot + '__' + variable
    plot_dict['init'] = init / float(len(VARs))
    return plot_dict


def plot_network(G, filename='example.png'):
    node_size=1000
    colors = ['indianred', 'cornflowerblue']
 
    pos = nx.get_node_attributes(G,'position')
    vote = nx.get_node_attributes(G,'vote')
    cueID = nx.get_node_attributes(G,'cue_id')
    connectionstyle = 'arc3,rad=0.2'
    H = G.to_directed()
    edges2plot = list(H.edges)

    plt.figure(figsize=(6,6))
    plt.axis("off")

    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=[colors[int(vote[i]>0)] for i in range(len(vote))], node_shape='o', linewidths=2.5, edgecolors='k', label=None)

    bla = nx.draw_networkx_edges(H, pos, edgelist=edges2plot, width=3.0, arrows=True, arrowsize=1, edge_color='grey', style='solid', alpha=0.7, node_size=node_size)
    for line in bla:
        line.set_connectionstyle(connectionstyle)
    plt.show()
    
