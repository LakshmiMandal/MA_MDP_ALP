
!pip install pulp

import random
import numpy as np
import pulp
from tabulate import tabulate
h=w=50 #height is h
# w=3 #width is w
nos=h*w
# nos=9
noa=4
no_agent=8
alpha=0.99
p=1-(0.1*(noa-1))
actions=[0,1,2,3]
goal_state=[[0,nos-1],[nos-1,0]]

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

# for st in range(9):
#   for ac in range(4):
#     total_p=sum(get_P(st,ac,next_st) for next_st in range(9))
#     print(st,ac,total_p)

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
    if(cur_state0==cur_state1 and ac[0]==ac[1]):
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

nobs=4 #number of basis function

def bFun1(mul_st):
  x0,y0=get_co(mul_st[0])
  x1,y1=get_co(mul_st[1])
  return pow(np.abs(x0-x1),2) + pow(np.abs(y0-y1),2)+1

def bFun2(mul_st):
  x0,y0=get_co(mul_st[0])
  x1,y1=get_co(mul_st[1])
  return np.sqrt(pow(np.abs(x0-x1),2) + pow(np.abs(y0-y1),2))


def bFun3(mul_st):
  x0,y0=get_co(mul_st[0])
  x1,y1=get_co(mul_st[1])
  return np.abs(x0-y0)+ np.abs(x1-y1)+1

def bFun4(mul_st):
  x0,y0=get_co(mul_st[0])
  x1,y1=get_co(mul_st[1])
  return pow(np.abs(x0-y0),2)+ pow(np.abs(x1-y1),2)+1

# def bFun5(mul_st):
#   x0,y0=get_co(mul_st[0])
#   x1,y1=get_co(mul_st[1])
#   return x0+2*pow(y0,2)+3*pow(x1,3)+4*pow(y1,3)+1
######################################################
#Do't use below basis functions:
# def bFun5(mul_st):
#   x0,y0=get_co(mul_st[0])
#   x1,y1=get_co(mul_st[1])
#   return pow(np.abs(x0-x1),3) + pow(np.abs(y0-y1),3)+1
# def bFun7(mul_st):
#   x0,y0=get_co(mul_st[0])
#   x1,y1=get_co(mul_st[1])
#   return pow(np.abs(x0-x1),4) + pow(np.abs(y0-y1),4)

# def bFun5(mul_st):
#   x0,y0=get_co(mul_st[0])
#   x1,y1=get_co(mul_st[1])
#   return np.abs(x0) + np.abs(x1)

# def bFun6(mul_st):
#   x0,y0=get_co(mul_st[0])
#   x1,y1=get_co(mul_st[1])
#   return np.abs(y0) + np.abs(y1)

# def bFun10(mul_st):
#   return 1

bFun=[bFun1,bFun2,bFun3,bFun4]

#Using identity feature Matrix

# nobs=81 #number of basis function
# bFun=np.zeros(nobs)

# def fun(arr,l):
#   if(get_index(arr)==l):
#     return 1
#   else:
#     return 0

# bFun=[]
# for i in range(nobs):
#   bFun.append(fun)

# print(bFun[21]([2,3],21))

# nobs=9 #number of basis function
# bFun=np.zeros(nobs)

# def fun(arr,l):
#   if(get_index(arr)==get_index(goal_state[0]) or get_index(arr)==get_index(goal_state[0])):
#     return pow(get_index(goal_state[0]), l)
#   else:
#     return pow(get_index(arr)+1, l)

# bFun=[]
# for i in range(nobs):
#   bFun.append(fun)

def CACFN(alpha):
  # theta=[pulp.LpVariable.dicts ("s{}".format(l), (range (1))) for l in range(nobs)] # the variables
  theta=pulp.LpVariable.dicts("c", range (nobs))
  problem = pulp.LpProblem ("PI_ALP", pulp.LpMaximize) # minimize the objective function

  # problem += sum(sum((theta[l] * bFun[l](cur_st)) for l in range(nobs)) for cur_st in multi_state)

  coef=np.zeros(nobs)


  for l in range(nobs):
      coef[l]=sum(bFun[l](cur_st) for cur_st in multi_state)
  print("coefficient of theta",coef)

  problem+= sum(theta[l]*coef[l] for l in range(nobs))
  # problem +=nos*(theta[0]*coef[0]+theta[1]*coef[1]+theta[2]*coef[2]+theta[3]*coef[3]+theta[4]*coef[4]+theta[5]*coef[5]+theta[6]*coef[6]+theta[7]*coef[7])
  # problem += (sum(sum(sum((theta[ag*nobs+l]* bFun[l](cur_st[ag])) for ag in range(no_agent)) for l in range(nobs))for cur_st in multi_state))

  cons1=np.zeros((nos*nos, noa*noa, nobs))

  for cur_st in multi_state:
    for cur_ac in multi_action:
      for l in range(nobs):
          cons1[get_index(cur_st)][action_index(cur_ac)][l]= bFun[l](cur_st) - alpha*(sum(return_prob(cur_st,cur_ac,next_st)* bFun[l](next_st) for next_st in multi_state))
  # print("coefficient of constraints",cons1)

  reward=np.zeros((nos*nos, noa*noa))
  for cur_st in multi_state:
    for cur_ac in multi_action:
      reward[get_index(cur_st)][action_index(cur_ac)] = sum(return_prob(cur_st,cur_ac,next_st)*return_reward(cur_st,cur_ac,next_st) for next_st in multi_state)


  for cur_st in multi_state:
    for cur_ac in multi_action:
      problem += sum(theta[l]*cons1[get_index(cur_st)][action_index(cur_ac)][l] for l in range(nobs)) <= (reward[get_index(cur_st)][action_index(cur_ac)])
  # for cur_st in multi_state:
  #   for cur_ac in multi_action:
  #     problem+= sum (sum (bFun[l](cur_st[ag])*theta[ag*nobs+l] for ag in range(no_agent)) for l in range(nobs)) - alpha*sum(return_prob(cur_st,cur_ac,next_st)* (sum (sum(bFun[l](next_st[ag])*theta[ag*nobs+l] for ag in range(no_agent)) for l in range(nobs))) for next_st in multi_state) <=sum(return_prob(cur_st,cur_ac,next_st)*return_reward(cur_st,cur_ac,next_st) for next_st in multi_state)
  # Solve the LP
  problem.solve()
  # print(problem.status,problem.valid())

  Theta=np.zeros(nobs)
  for l in range(nobs):
      Theta[l]=(theta[l]).varValue
  print(Theta)
  # Theta=np.zeros((nobs*no_agent))
  # for l in range(nobs):
  #   for ag in range(no_agent):


  # Theta=Theta1.reshape(nobs)
  # print("After Reshape",Theta1)
  # Theta=Theta.transpose()

  print("Theta values",Theta)
  Value = np.zeros (nos*nos)
  for st in multi_state:
    Value[get_index(st)]=sum((Theta[l]*bFun[l](st)) for l in range(nobs))
  print("Values\n",Value)

  Q_app=np.zeros((noa*noa, nos*nos))
  v_app=np.zeros(nos*nos)
  for cur_st in multi_state:
    for cur_ac in multi_action:
      Q_app[action_index(cur_ac)][get_index(cur_st)]= sum (return_prob(cur_st, cur_ac, next_st) *(return_reward (cur_st, cur_ac, next_st)+alpha* Value[get_index(next_st)]) for next_st in multi_state)
    v_app[get_index(cur_st)] =min(Q_app[:,get_index(cur_st)])

  # for cur_st in multi_state:
  #   if(v_app[get_index(cur_st)] < Value[get_index(cur_st)]):
  #     print("state, Approx Val, Val", cur_st,v_app[get_index(cur_st)], Value[get_index(cur_st)])

  # print("Value:\n")
  val=Value.reshape (nos,nos)
  # v_app=v_app.reshape(nos,nos)

  print("value:",val)
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