import os
import re
import glob

import torch
from torch_geometric.data import Data

'''
Take from Reza's colab notebook
'''


def adj_mat_fun(filename):
  #The file name should be dumped using -fdump-tree-ssa-gimple. It's the 021 tree.
  lines    = []

  with open(filename, "r") as file:
      for line in file:
          lines.append(line.strip())

  #################################################################################
  #Create the list of using variables.

  function_name    = 0  #Here I just enumerate
  initial_part     = 0
  var_dict_temp    = {} #It's local for any function and has the temperory vars
  var_dict_static  = {} #It's local for any function and has the args and defined vars
  var_all          = {} #Key's are corresponded to the funcation name and values are the var_dict_temp of each function
  fun_def          = 0  #check for __GIMPLE to see if we are at the beginning of a function

  for line in lines:
      words = line.split()

      if fun_def: #since we know the immidiate line after the __GIMPLE is the argument lines of the function
          fun_def = 0
          args = line.split(sep=',') #first split it based on ','
          for arg in args:
              arg_list = arg.split()
              if '(' not in arg: #check if it is the first arguman and has the function name and other stuff in it
                  if ')' not in arg: #check if it is the last arguman and has the last ')'
                      var_dict_static[arg_list[-1]] = arg[1:-(len(arg_list[-1])+1)]
                  else:
                      var_dict_static[arg_list[-1][0:-1]] = arg[1:-(len(arg_list[-1])+1)]
              else:
                  par_ind = arg.find('(')
                  if ')' not in arg:
                      var_dict_static[arg_list[-1]] = arg[par_ind+1:-(len(arg_list[-1])+1)]
                  else:
                      if '(' not in arg_list[-1]:
                          var_dict_static[arg_list[-1][0:-1]] = arg[par_ind+1:-(len(arg_list[-1])+1)]

      for word in words:
          if '__GIMPLE' in word: #check for the __GIMPLE that is a distinctive annotation for the beginning of the function
              fun_def                 = 1

          if '{ ' == word: #check to see if we are in the function scope.
              initial_part            = 1 #since in the SSA all the initializations are at the beginning
          if '}' == word: #check to see if we are out of the function scope.
              var_all_local           = {} #to have temp and static vars in dict
              var_all_local['static'] = var_dict_static
              var_all_local['temp']   = var_dict_temp
              var_all[function_name]  = var_all_local #updating var_all for values in the parsed function
              var_dict_temp           = {} #clear the local var_dict_temp
              var_dict_static         = {} #clear the local var_dict_static
              function_name          += 1 #increase the index of function
          if '__BB' in word:
              initial_part            = 0 #check the end of initialization

      if initial_part and len(words)>1:
          if words[-1][0] == '_': #having '_' at the beginning indicates it's temperoray and made by the compiler
              var_dict_temp[words[-1][0:-1]]   = line[0:-(len(words[-1])+1)]
          else:
              var_dict_static[words[-1][0:-1]] = line[0:-(len(words[-1])+1)]

  #################################################################################
  #Create the adjacent matrix for BBs
  fun_def                  = 0  #check for __GIMPLE to see if we are at the beginning of a function
  function_scope           = 0  #to check if we are inside a scope
  function_name            = 0  #Here I just enumerate
  BB_name                  = 0  #to have the name of current BB. It's only valid if BB_scope=1
  BB_ind                   = -1 #to store a numeric value for BBs.
  BB_scope                 = 0  #to check if we are inside a scope
  BB_first_line            = 0  #to store the BBs name and info
  BB_local_dict            = {} #it has the 0-indexed vertex as key and BBs name in val for each function
  BB_global_dict           = {} #it has the 0-indexed vertex as key and BBs name in val for all functoins
  BB_rev_local_dict        = {} #it has the 0-indexed vertex as val and BBs name in key for each function
  BB_rev_global_dict       = {} #it has the 0-indexed vertex as val and BBs name in key for all functoins
  BB_adjacent_mat          = [] #it's for all the BBs inside a function
  BB_local_adj             = [] #it's local to each BB in a function
  BB_adjacent_mat_global   = {} #it's for all of the funcitons
  BB_livness               = [] #it's for all the BBs inside a function
  BB_livness_local         = set() #it's local to each BB in a function
  BB_livness_gloabal       = {} #it's for all of the funcitons
  BB_livness_start         = {} #it's for all the BBs inside a function
  BB_livness_gloabal_start = {} #it's for all of the funcitons
  BB_all_val               = [] #have all the variable use in a function
  BB_all_val_global        = {} #have entire variables
  for line in lines:
      words = line.split()

      if fun_def: #since we know the immidiate line after the __GIMPLE is the argument lines of the function
          fun_def = 0

      if '__GIMPLE' in line: #check for the __GIMPLE that is a distinctive annotation for the beginning of the function
          fun_def                                 = 1

      if '{' in line and len(line)<2: #check to see if we are in the function scope.
          function_scope                          = 1 #check if we are inside of the function scope
      if '}' in line and len(line)<2: #check to see if we are out of the function scope.
          function_scope                          = 0
          BB_global_dict[function_name]           = BB_local_dict
          BB_local_dict                           = {}
          BB_rev_global_dict[function_name]       = BB_rev_local_dict
          BB_rev_local_dict                       = {}
          BB_adjacent_mat.append(BB_local_adj)
          BB_adjacent_mat_global[function_name]   = BB_adjacent_mat
          BB_local_adj                            = []
          BB_adjacent_mat                         = []
          BB_livness.append(BB_livness_local)
          BB_livness_gloabal[function_name]       = BB_livness
          BB_livness_gloabal_start[function_name] = BB_livness_start
          BB_livness_local                        = set()
          BB_livness                              = []
          BB_livness_start                        = {}
          BB_ind                                  = -1
          function_name                          += 1 #increase the index of function

      if line[0:4] == '__BB':
          BB_scope                        = 1 #check that we are inside a BB
          BB_first_line                   = 1

      if BB_first_line: #store BBs data
          if BB_ind>=0: #not for the first one
              BB_adjacent_mat.append(BB_local_adj)
              BB_livness.append(BB_livness_local)
          BB_local_adj                    = []
          BB_livness_local                = set()
          BB_livness_local_start          = set()
          BB_first_line                   = 0 #reset the first line flag for BBs
          start                           = line.find('(')+1
          if line.find(',') > 0: #check if there is other info
              end                         = line.find(',')
          else:
              end                         = line.find(')')
          BB_name                         = line[start: end]
          BB_ind                         += 1
          BB_local_dict[BB_ind]           = int(BB_name) #update the local dict for BBs' name
          BB_rev_local_dict[int(BB_name)] = BB_ind

      if BB_scope:
          if ' =' in line and 'if' not in line: #Check if there is an assignment. This can be consider as the start of the livenss analysis since the tree is in SSA form
              if '__MEM' not in words[0]:
                  BB_livness_local.add(words[0])
                  BB_livness_start[words[0]] = BB_ind

              for word in words:
                  var  = ''
                  vars = set()
                  if '_' in word and '__' not in word:
                      for char in word:
                          asci = ord(char)
                          if (asci<=90 and asci >=65) or (asci<=57 and asci>=48) or (asci<=122 and asci>=97) or char =='_':
                              var += char
                          else:
                              if len(var)>0:
                                  vars.add(var)
                                  var = ''
                      vars.add(var)
                      for var in vars:
                          if '_' in var:
                              BB_livness_local.add(var)

          if 'goto' in line:
              start = line.find('B')+2
              BB_local_adj.append(int(line[start:-1]))
          if 'return' in line and '//' not in line:
              BB_local_adj.append(-1)


  #################################################################################
  #Traversing the tree to check the liveness

  live_global    = BB_livness_gloabal #To store liveness for each BBs of all function
  live_local     = set() #To store liveness for each BBs of each function
  live           = [] #to store liveness for each function
  visited        = [] #To store visited edge of all functions
  visited_times  = [] #To store number of visited edge of all functions
  function_name  = -1
  last_visted    = [10]
  for BB_adjacent_mat_key in BB_adjacent_mat_global:

      BB_adjacent_mat = BB_adjacent_mat_global[BB_adjacent_mat_key]
      function_name += 1
      visited_times  = []
      for vertex in BB_adjacent_mat: #set all vertex zero visited
          visited_times.append(0)

      '''print('adjmat\n')
      for i in range(len(BB_adjacent_mat)):
        print(i+2, ':', BB_adjacent_mat[i], sep='', end=' |&| ')
      print('\n')'''
      while True: #Continue until all paths have been visited
          next = 2
          visited = []
          temp_visited_times = [0 for x in visited_times]
          for i in range(len(BB_adjacent_mat)): #set all vertex zero visited
            temp_visited_times[i] = visited_times[i]

          while next!=-1:#Continue until the last BB
              visited.append(next)
              BB_ind   = BB_rev_global_dict[function_name][next]
              len_next = len(BB_adjacent_mat[BB_ind])
              if temp_visited_times[BB_ind] < len_next:
                  next = BB_adjacent_mat[BB_ind][temp_visited_times[BB_ind]]
                  temp_visited_times[BB_ind] += 1
              else:
                  next = BB_adjacent_mat[BB_ind][temp_visited_times[BB_ind]-1]

          #print('visited:\n', visited)
          #print('times:\n', visited_times)

          for var in BB_livness_gloabal_start[function_name]:#Update the liveness of the BB in the path
              BB_ind = BB_livness_gloabal_start[function_name][var]
              BB     = BB_global_dict[function_name][BB_ind]
              if BB in visited:
                start  = visited.index(BB)
                last   = start
                for i in range(start+1,len(visited)):
                    BB_ind_visited = BB_rev_global_dict[function_name][visited[i]]
                    if var in BB_livness_gloabal[function_name][BB_ind_visited]:
                        last = i
                for i in range(start+1, last+1):
                    BB_ind_visited = BB_rev_global_dict[function_name][visited[i]]
                    live_global[function_name][BB_ind_visited].add(var)

          visited.reverse()
          check = 1
          for vert in visited:
              BB_ind   = BB_rev_global_dict[function_name][vert]
              len_next = len(BB_adjacent_mat[BB_ind])
              if visited_times[BB_ind] < len_next-1:
                  visited_times[BB_ind] += 1
                  check = 0
                  break

          for i in range(BB_ind+1,len(visited_times)):
              visited_times[i] = 0

          if check:
            break

          last_visted = visited
          #print('Updated times:\n', visited_times)
         # x = input()





  ##################################################################################



  BB_ind = 0
  adj_list = []
  for BB_local_adj in BB_adjacent_mat_global[0]:
      for adj in BB_local_adj:
          if adj > 0:
              adj_list.append([BB_global_dict[0][BB_ind]-2, adj-2])
      BB_ind += 1

  BB_ind = 0
  live_list = []
  for BB_livness in BB_livness_gloabal[0]:
      live_list.append([len(BB_livness)/100])
      BB_ind += 1
  BB_ind = 0

  for BB_livness in live_global[0]:
      live_list[BB_ind].append(len(BB_livness)/100)
      BB_ind += 1

  return [live_list, adj_list]


def read_graphs(folder):
    files = glob.glob(f'{folder}{os.sep}*.ssa.c')

    graphs = [adj_mat_fun(f) for f in files]

    return graphs

def graph_to_data(G, label):
    edge_index = torch.tensor(list(G[1])).t().contiguous()
    
    x = torch.tensor(G[0])
    
    y = torch.tensor([label])
    
    return Data(x=x, edge_index=edge_index, y=y)
    