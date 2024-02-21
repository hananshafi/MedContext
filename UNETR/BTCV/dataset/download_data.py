import synapseclient 
import synapseutils 
 
syn = synapseclient.Synapse() 
syn.login('hanangani','Synapse.com@123') 
files = synapseutils.syncFromSynapse(syn, 'syn3193805')
