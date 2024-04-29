
# !pip install pulp

import random
import numpy as np
import pulp
from tabulate import tabulate
h=w=50 #height is h, |S|=2500
# w=3 #width is w
nos=h*w
# nos=9
noa=5
no_agent=10
alpha=0.99
p=1-(0.1*(noa-1))
actions=[0,1,2,3,4]

def get_co(state):
    return [int(state/h),int(state%h)]

def get_nx_st(st,ac):
    x,y=get_co(st)    
    tr_p=[[x-1,y],[x+1,y],[x,y-1],[x,y+1]]
    return tr_p[ac]
def get_P(cur_st,cur_ac,next_st):
    # p=0.7
    pro=0
    other_actions = list(filter(lambda a: a!=cur_ac, actions))
    if(next_st==cur_st):
        if((get_nx_st(cur_st,cur_ac)[0]<0 and cur_ac==0) or (get_nx_st(cur_st,cur_ac)[0]>(h-1) and cur_ac==1) or (get_nx_st(cur_st,cur_ac)[1]<0 and cur_ac==2) or (get_nx_st(cur_st,cur_ac)[1]>(h-1) and cur_ac==3)):
          pro+=p
        for ac in other_actions: 
          if(((get_nx_st(cur_st,ac)[0]<0)or (get_nx_st(cur_st,ac)[0]>(h-1)) or (get_nx_st(cur_st,ac)[1]<0)or (get_nx_st(cur_st,ac)[1]>(h-1)))):
            pro+=(1-p)/(noa-1) 
          else:
            pro+=0
    else:        
        if(get_co(next_st)==get_nx_st(cur_st,cur_ac)):
          pro+=p
        for ot_ac in other_actions:
          if(get_co(next_st)==get_nx_st(cur_st,ot_ac)):
            pro+=((1-p)/(noa-1))
          else:
            pro+=0  
        
    return pro

def return_prob(current_state,action,next_state):
  goal_state=[[0,nos-1],[nos-1,0]]
  if(current_state==goal_state[0] and next_state == goal_state[0]) or (current_state==goal_state[1] and next_state == goal_state[1]):
    return 1
  elif (current_state!=goal_state[0] and current_state!=goal_state[1]): 
    return (get_P(current_state[0],action[0],next_state[0])* get_P(current_state[1],action[1],next_state[1]))
  else:  
    return 0

def return_reward(cur_state, ac, next_state):
  cur_state0=cur_state[0]
  cur_state1=cur_state[1]
  if(cur_state==[0,nos-1]) or (cur_state==[nos-1,0]):
    return 0
  else:
    # if(cur_state0==cur_state1 and ac[0]==ac[1]):
    #   return 6
    if(next_state[0]==next_state[1]):
      return 6  
    else:
      return 2

def base_policy(state):
  return [random.choice(actions),random.choice(actions)]

multi_state=[]
for st1 in range(nos):
  for st2 in range(nos):
    multi_state.append([st1,st2])
# print(multi_state)
multi_action=[]
for action1 in range(noa):
  for action2 in range(noa):
    multi_action.append([action1,action2])
# print(multi_action)

def get_index(state_tuple):
  return int(state_tuple[0]*nos+state_tuple[1])

def action_index(action_tuple):
  return int(action_tuple[0]*noa+action_tuple[1])

# p_mat=np.zeros((noa*noa,nos*nos,nos*nos))
# for ac in multi_action:
#   for st in multi_state:
#     for next_st in multi_state:
#       p_mat[action_index(ac)][get_index(st)][get_index(next_st)]=return_prob(st,ac,next_st)
# p_mat=p_mat.reshape((noa*noa,nos*nos,nos*nos))  
# print (tabulate(p_mat, tablefmt="grid"))
# for st in multi_state:
#   for ac in multi_action:
#     total_p=sum(return_prob(st,ac,next_st) for next_st in multi_state)
#     print(st,ac,total_p)

nobs=12 #number of basis function
def bFun1(i):
  x,y=get_co(i)
  return np.abs(x)

def bFun2(i):
  x,y=get_co(i)
  return np.abs(y)

def bFun3(i):
  x,y=get_co(i)
  return np.abs(x-y)

def bFun4(i):
  return 1 

def bFun5(i):
  x,y=get_co(i)
  return np.abs(x-1)

def bFun6(i):
  x,y=get_co(i)
  return np.abs(y-1) 

def bFun7(i):
  x,y=get_co(i)
  return pow(x,2)

def bFun8(i):
  x,y=get_co(i)
  return pow(y,2)

def bFun9(i):
  x,y=get_co(i)
  return pow(x,3)

def bFun10(i):
  x,y=get_co(i)
  return pow(y,3)

def bFun11(i):
  x,y=get_co(i)
  return pow(np.abs(x-y),2)

def bFun12(i):
  x,y=get_co(i)
  return pow(np.abs(x-y),3)

def bFun13(i):
  x,y=get_co(i)
  return pow(np.abs(x-1),2)

def bFun14(i):
  x,y=get_co(i)
  return pow(np.abs(y-1),2) 

def bFun15(i):
  x,y=get_co(i)
  return pow(x,4)

def bFun16(i):
  x,y=get_co(i)
  return pow(y,4)

def bFun17(i):
  x,y=get_co(i)
  return pow(np.abs(x-y),4)

def bFun18(i):
  x,y=get_co(i)
  return pow(np.abs(x-y),5)

# bFun=[bFun1,bFun2,bFun3,bFun4,bFun5,bFun6,bFun7,bFun8,bFun9,bFun10,bFun11,bFun12,bFun13,bFun14,bFun15,bFun16,bFun17,bFun18]
# bFun=[bFun1,bFun2,bFun3,bFun4,bFun5,bFun6,bFun7,bFun8,bFun9,bFun10,bFun11,bFun12,bFun13,bFun14,bFun17,bFun18]
bFun=[bFun1,bFun2,bFun3,bFun4,bFun5,bFun6,bFun7,bFun8,bFun9,bFun10,bFun11,bFun12]

def CACFN(alpha):
  theta=[[pulp.LpVariable.dicts ("s{}".format(l*no_agent+ag), (range (no_agent)))for ag in range(no_agent)] for l in range(nobs)] # the variables
  problem = pulp.LpProblem ("PI_ExactLP", pulp.LpMaximize) # minimize the objective function

  problem += (sum(sum(sum((theta[l][ag][0]* bFun[l](cur_st[ag])) for ag in range(no_agent)) for l in range(nobs))for cur_st in multi_state)) # defines the objective function

  # print("Objective fun:", prob)
  # now, we define the constrain: there is one for each (state, action) pair.
  
  for cur_st in multi_state:
    for cur_ac in multi_action:
      problem += sum(return_prob(cur_st,cur_ac,next_st)*(return_reward(cur_st,cur_ac,next_st)+ alpha* (sum(sum((theta[l][ag][0] * bFun[l](next_st[ag])) for ag in range(no_agent)) for l in range(nobs)))) for next_st in multi_state) >= (sum(sum((theta[l][ag][0] * bFun[l](cur_st[ag])) for ag in range(no_agent)) for l in range(nobs)))
  # print(problem)    
           
  # Solve the LP
  problem.solve()
  # print(problem.status,problem.valid())

  Theta=np.zeros((nobs,no_agent))
  for l in range(nobs):
    for ag in range(no_agent):
      Theta[l][ag]=theta[l][ag][0].varValue

  print("Theta values",Theta)


  # # extract the value function
  Value = np.zeros (nos*nos)
  for st in multi_state:
    Value[get_index(st)]=sum(sum((Theta[w][ag]*bFun[w](st[ag])) for ag in range(no_agent)) for w in range(nobs))

  # print("Value:\n")
  # val=Value.reshape (16, 16)

  # Q_app=np.zeros((noa*noa, nos*nos))
  # v_app=np.zeros(nos*nos)
  # for cur_st in multi_state:  
  #   for cur_ac in multi_action:
  #     Q_app[action_index(cur_ac)][get_index(cur_st)]= sum (return_prob(cur_st, cur_ac, next_st) *(return_reward (cur_st, cur_ac, next_st)+alpha* Value[get_index(next_st)]) for next_st in multi_state)
  #   v_app[get_index(cur_st)] =min(Q_app[:,get_index(cur_st)])

  # # print("Value:\n")
  # val=Value.reshape (nos,nos)
  # v_app=v_app.reshape(nos,nos)

  # print("value:",val)
  # print("Approx Val",v_app)

  return Value

# CACFN(alpha)

# Initialisations
Q =np.zeros((nos*nos, noa*noa))
alpha=0.99
pi_base=[base_policy(cur_st) for cur_st in multi_state]
pi_til=pi_base.copy()
J=CACFN(alpha)

k=0
flag=True
while flag:
  flag=False
  k=k+1
  for cur_st in multi_state:
    for cur_ac in multi_action:
      # print(get_index(cur_st), action_index(cur_ac))
      Q[get_index(cur_st), action_index(cur_ac)]= sum(return_prob(cur_st, cur_ac, next_st) *(return_reward (cur_st, cur_ac,next_st)+alpha* J[get_index(next_st)]) for next_st in multi_state)
    combo_action=Q[get_index(cur_st), :].argmin()  
    pi_til[get_index(cur_st)] = [int(combo_action/noa),int(combo_action%noa)]
    # print(cur_st, Q)      
  # print("Policy is: {}".format(pi_til))

  J_old=J.copy()
  J=CACFN(alpha) 
  for st in multi_state:
    if(J[get_index(st)] < J_old[get_index(st)]):
      pi_base=pi_til.copy()
      flag=True
      break
cost=np.array(J)
cost=cost.reshape(nos,nos)
print (tabulate(cost, tablefmt="grid"))

pi_til=np.array(pi_til)
pi=pi_til.reshape(nos,nos,no_agent)
print (tabulate(pi, tablefmt="grid"))        
# print("Policy is: {}".format(pi_til.reshape(nos,nos,no_agent)))



