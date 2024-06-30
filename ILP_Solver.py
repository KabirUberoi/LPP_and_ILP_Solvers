import numpy as np
import math
from fractions import Fraction

"""##The Simplex Code From A1

####This is the simplex code we had implemented in A1 in the form of a function. It takes as input -
* A, b, c, objective and constraint_types

And returns a hash-map which consists of -
* initial_tableau, final_tableau, solution_status, optimal_value and optimal_solution

###Helper Functions

#### Cb

`get_cb` calculates Cb (cost coef. of basic variables)
"""

def get_cb(c , b_idx) :
  m = len(b_idx)
  cb = np.zeros(m)
  for i in range(m) :
    cb[i] = c[b_idx[i]]
  return cb

"""#### C_bar

C_bar = C_new - Cb dot tableau

don't take transpose as Cb is already row vector
"""

def get_C_bar(c , cb , tab) :
  return c - np.dot(cb, tab)

"""#### Pivot col index (ENTERING VARIABLE)
this function returns the column number (0 based indexing) of pivot column in Tableau

process same as class/book

examine `c_bar` vector and get 1st index where `c_bar[i]` is -ve

if `c_bar` >= `0` then returns `-1` so that process can be terminated
"""

def get_pivot_col_idx(c_bar) :
  l = len(c_bar)
  for i in range(l) :
    if c_bar[i] < 0 :
      return i  # ith col of tableau is pivot col
  return -1 # when -1 return then terminate tableau method

"""#### Pivot row index (LEAVING VARIABLE)
this function returns the row number (0 based indexing) of pivot column in Tableau

process same as class/book

also returns `theta` that is used for calculating next bfs

`x_next` = `x` + `theta`*`d`
"""

def get_pivot_row_idx(pivot_col , x , b_idx) :
  # pivot_col is a vertical vector containing the col corresponding to pvt_col_idx
  l = len(b_idx)
  theta = Fraction(10**15, 1)
  idx = -1

  for i in range(l) :
    if pivot_col[i] > 0 :
      temp = x[b_idx[i]]/pivot_col[i]
      if temp < theta :
        theta = temp
        idx = i

  for i in range(l) :
    if pivot_col[i] > 0 and theta == x[b_idx[i]]/pivot_col[i] and b_idx[i] < idx :
      idx = i

  return idx, theta

"""#### Next Tableau
this functions does required row transformations to get new tableau when we are given the pivot index

process same as class/book
"""

def get_nxt_tab(tab , pvt_row_idx , pvt_col_idx) :

  pivot_element = tab[pvt_row_idx , pvt_col_idx]
  tab[pvt_row_idx,:] = tab[pvt_row_idx,:] / pivot_element

  m = tab.shape[0]

  for i in range(m) :
    if i != pvt_row_idx :
      factor = tab[i,pvt_col_idx]
      tab[i,:] = tab[i,:] - (factor*tab[pvt_row_idx,:])

  return tab

"""### Tableau function"""

def Tableau(tab , x , b_idx , c) :

  cb = get_cb(c , b_idx)
  c_bar = get_C_bar(c , cb , tab)


  pvt_col_idx = get_pivot_col_idx(c_bar)

  is_infinity = False

  while pvt_col_idx > -1 :

    pvt_col = np.transpose(tab[: , pvt_col_idx])

    pvt_row_idx , theta = get_pivot_row_idx(pvt_col , x , b_idx)

    if pvt_row_idx == -1 :
      is_infinity = True
      break

    l = len(x)

    d = np.zeros(l , dtype=object)
    d = d + Fraction()

    d[pvt_col_idx] = Fraction(1,1)

    for i in range(len(b_idx)) :
      d[b_idx[i]] = - tab[i,pvt_col_idx]

    tab = get_nxt_tab(tab , pvt_row_idx , pvt_col_idx)
    # now get new x, b_idx, ca

    # next adjacent BFS
    x = x + (theta*d)
    # print(f"x for this TAB = {x}")

    # chage the basic variable index for next itteration
    b_idx[pvt_row_idx] = pvt_col_idx
    # print(f"b_idx for this TAB = {b_idx}")

    # new basis cost vector
    cb = get_cb(c , b_idx)
    # print(f"cb for this TAB = {cb}")

    # new reduced vector
    c_bar = get_C_bar(c,cb,tab)
    # print(f"c_bar for this TAB = {c_bar}")

    pvt_col_idx = get_pivot_col_idx(c_bar)

  return tab , x , b_idx , is_infinity

"""###Simplex Algo"""

def simplex_algo(A,b,c,objective,constraint_types):

    """## Format the input data to fit specifications

    ###if maximise then switch signs of c to -ve to change to minimization
    """

    # if objective == 'maximize':
    #     c = c*(-1)

    """### Slack variables
    1. for `>=` new slack variable to the right i.e. new column with `-1` at that row
    2. for `<=` new slack variable to the left i.e. new column with `1` at that row
    3. in the case that `b == 0`, `A_row == e_i`, and `constraint == '<='` a `column == -xi` has to be inserted
    """

    # count_slack_variable = 0
    # for i,constraint in enumerate(constraint_types):
    #     if constraint == ">=": ## >= -> new slack variable to the right -> new column with -1 at that row
    #         count_slack_variable += 1
    #         ei = np.zeros((A.shape[0],1))
    #         ei[i,0] = -1
    #         A = np.hstack((A, ei))
    #     elif constraint == "<=":## <= -> new slack variable to the left -> new column with 1 at that row
    #         count_slack_variable += 1
    #         ei = np.zeros((A.shape[0],1))
    #         ei[i,0] = 1
    #         A = np.hstack((A, ei))

    # c = np.append(c , np.zeros(count_slack_variable))

    """#### If b is -ve for a row
    multiply the row with `-1` to make sure that the vector b is `>`
    """

    for i,temp in enumerate(b):
        if temp<0:
            A[i] = A[i]*-1
            b[i] = b[i]*-1

    """# Creating Auxiliary LPP"""

    m , n = A.shape
    I = np.eye(m , dtype = object)
    I = I + Fraction()
    A_aux = np.hstack((A , I))

    c_aux = np.zeros(n + m)
    c_aux[n:n+m] = 1

    """CHECK"""

    # print("A (aux) : ")
    # print(A_aux)
    # print("")
    # print("C aux : ")
    # print(c_aux)

    """### Current BFS

    `x_cur` is current BFS

    `B_idx` stores indices of basic variables

    `Cb` stores cost of basic variables

    `C_bar` is reduced cost vector
    """

    n_aux = n + m

    x_aux = np.zeros(n_aux , dtype = object)
    x_aux = x_aux + Fraction()
    x_aux[n : n_aux] = b[0 : m]

    b_idx_aux = np.zeros(m , dtype=int)
    for i in range(n , n_aux , 1) :
        b_idx_aux[i - n] = i

    tableau = A_aux

    # print(x_aux)
    # print(b_idx_aux)

    """# PHASE 1"""

    A_aux ,  x_aux , b_idx_aux , is_infinity_aux = Tableau(A_aux , x_aux , b_idx_aux , c_aux)

    """## Calculate optimal cost for Auxilary LPP"""

    cost_aux = np.dot(x_aux , c_aux)

    """If the optimal cost in the auxiliary problem is positive, the original problem is infeasible and the algorithm terminates."""
    solution_status = ""
    if(cost_aux >= 0.00000001) :
        solution_status = "infeasible"

    """## Eliminating y variables out the phase 1 tableau

    """

    if(cost_aux < 0.00000001):
        i = 0
        p, q = A.shape

        while(i < len(b_idx_aux)):
            if(b_idx_aux[i] >= q):

                f = False
                for j in range(0,q):
                    if(A_aux[i][j] != 0):
                        f = True
                        b_idx_aux[i] = j
                        x_aux[j] = 0
                        for jj in range(len(b_idx_aux)):
                            if(jj != i):
                                pi = (A_aux[jj][j])/(A_aux[i][j])
                                for cn in range(0,q):
                                    A_aux[jj][cn] -= A_aux[i][cn]*pi
                        for cn in range(0,q):
                            A_aux[i][cn] = A_aux[i][cn]/A_aux[i][j]
                        break
                if(f == True):
                    # print("Y")
                    i += 1
                else:
                    # print("y")
                    A_aux = np.delete(A_aux, i, axis=0)
                    b_idx_aux = np.delete(b_idx_aux, i, axis=0)
            else:
                #   print("N")
                i+=1


    #   print(b_idx_aux)
    #   print(x_aux)
    #   print(A_aux)
    #   print(c)

    """# phase 2

    """
    b_idx_final = None
    x_f = None
    if(cost_aux < 0.00000001):
        A_f ,  x_f , b_f , is_infinity = Tableau(A_aux[:, :n] , x_aux[:n] , b_idx_aux , c)
        optimal_cost=np.dot(c,x_f)
        b_idx_final = b_f
        if(is_infinity) :
            # print("UNBOUNDED")
            solution_status = "unbounded"
        else :
            # print(optimal_cost)
            solution_status = "optimal"

    if(cost_aux < 0.00000001):
        x_final = x_f
        #  [0:(number_of_initial_var)]
    else:
        x_final = None

    #solution_status already exists
    initial_tableau = tableau
    result = {}

    if(solution_status == "infeasible"):
        final_tableau = A_aux
        optimal_solution = None
        optimal_value = None

        result['initial_tableau'] = initial_tableau
        result['final_tableau'] = final_tableau
        result['solution_status'] = solution_status
        result['optimal_value'] = optimal_value
        result['x_final'] = optimal_solution
        result['basis_idx_final'] = b_idx_final


    elif(solution_status == "optimal"):
        final_tableau = A_f
        optimal_solution = x_final
        # print(optimal_solution)
        optimal_value = optimal_cost
        if(objective == "maximize"):
            optimal_value = optimal_value * (-1)
        # print(optimal_value)

        result['initial_tableau'] = initial_tableau
        result['final_tableau'] = final_tableau
        result['solution_status'] = solution_status
        result['optimal_value'] = optimal_value
        result['x_final'] = optimal_solution
        result['basis_idx_final'] = b_idx_final

    elif(solution_status == "unbounded"):
        final_tableau = A_f
        optimal_solution = None
        optimal_value = None

        result['initial_tableau'] = initial_tableau
        result['final_tableau'] = final_tableau
        result['solution_status'] = solution_status
        result['optimal_value'] = optimal_value
        result['x_final'] = optimal_solution
        result['basis_idx_final'] = b_idx_final

    return result

"""
# Dual Simplex Functions
"""

# return {row_idx in tab , index of basis of leaving basic variable}
# if no basic variable is leaving then return -1
def get_leaving_idx_DS():
  # accessing global variables
  global Tab, basis_idx, RHS, Cj, is_basic, x
  h = len(RHS)
  for i in range(h):
    if RHS[i] < 0 :
      return i , basis_idx[i]
  return -1 , -1

def get_entering_idx_DS(leaving_row_idx):

  # accessing global variables
  global Tab, basis_idx, RHS, Cj, is_basic, x

  h , l = Tab.shape # height , lenght of tablue

  ans = -1

  C_bar = np.zeros(l,dtype = object)
  C_bar = C_bar + Fraction()
  Theta_mini = float('inf')

  # compute c_bar and corresponding theta
  for i in range(l) :
    if is_basic[i] == False :
      C_bar[i] = Cj[i]
      for j in range(h) :
        C_bar[i] = C_bar[i] - (Tab[j,i] * Cj[basis_idx[j]])   # verified

      if C_bar[i] >= 0 and Tab[leaving_row_idx , i] < 0 :
        Theta = - C_bar[i] / Tab[leaving_row_idx , i]
        if Theta < Theta_mini :
          Theta_mini = Theta
          ans = i

  return ans

def next_tab(leaving_row_idx , leaving_basis_idx , entering_basis_idx) :

  # leaving_row_idx , leaving_basis_idx , entering_basis_idx are verified that correct

  # accessing global variables
  global Tab, basis_idx, RHS, Cj, is_basic, x

  h , l = Tab.shape # height , lenght of tablue

  pvt = Tab[leaving_row_idx , entering_basis_idx] # verified

  Tab[leaving_row_idx,:] = Tab[leaving_row_idx,:] / pvt
  RHS[leaving_row_idx] = RHS[leaving_row_idx] / pvt

  for i in range(h) :
    if i != leaving_row_idx :
      m = Tab[i,entering_basis_idx] / Tab[leaving_row_idx , entering_basis_idx] # verified BUT IT IS IN DECIMAL NOT FRACTIONS
      Tab[i,:] = Tab[i,:] - m*Tab[leaving_row_idx,:]
      RHS[i] = RHS[i] - m*RHS[leaving_row_idx]

  # update basis_idx, is_basic
  is_basic[leaving_basis_idx] = False
  is_basic[entering_basis_idx] = True
  for i in range(h) :
    if basis_idx[i] == leaving_basis_idx :
      basis_idx[i] = entering_basis_idx

# returns:
# 1 : itration complete normally
# 2 : leaving_row_idx == -1
# 3 : leaving_row_idx =!= -1 and entering_idx == -1 :

def dual_simplex_itr():

  # accessing global variables
  global Tab, basis_idx, RHS, Cj, is_basic, x

  leaving_row_idx , leaving_basis_idx = get_leaving_idx_DS()
  if leaving_row_idx != -1 :
    entering_idx = get_entering_idx_DS(leaving_row_idx)
    if entering_idx != -1 :
      next_tab(leaving_row_idx , leaving_basis_idx , entering_idx)
      return 1
    else :
        return 3
  else :
    return 2

# input parameters are expected to be set according to
# " = " constraints
def dual_simplex_loop() :

  # accessing global variables
  global Tab, basis_idx, RHS, Cj, is_basic, x, result

  is_basic = np.array([False] * Tab.shape[1])
  for i in basis_idx:
    is_basic[i] = True

  final_tableau = None
  solution_status = None
  optimal_value = None
  x_final = None
  basis_idx_final = None

  while True :

    temp = dual_simplex_itr()

    if temp == 1 :

      continue

    elif temp == 2 :

      final_tableau = Tab

      solution_status = 'optimal'

      x = np.zeros(Tab.shape[1] , dtype = object)
      x = x + Fraction()

      for i in range(Tab.shape[0]):
        x[basis_idx[i]] = RHS[i]

      optimal_value = np.dot(Cj,x)

      x_final = x

      basis_idx_final = basis_idx

      break

    else :

      final_tableau = Tab

      solution_status = 'unbounded'

      x = np.zeros(Tab.shape[1], dtype = object)
      x = x + Fraction()

      for i in range(Tab.shape[0]):
        x[basis_idx[i]] = RHS[i]

      optimal_value = np.dot(Cj,x)

      x_final = x

      basis_idx_final = basis_idx

      break
  result = {}

  result['final_tableau'] = final_tableau
  result['solution_status'] = solution_status
  result['optimal_value'] = optimal_value
  result['x_final'] = x_final
  result['basis_idx_final'] = basis_idx_final

  return result

def check(lst):
  ch = -1
  for i in range(len(lst)):
    if(int(math.floor(lst[i])) != lst[i]):
        ch = i
        break
  return ch

"""# SOLVING RELAXED ILP"""
def gomory_cut_algo():
  A = []
  b = []
  c = []
  objective = None
  initial_objective = None
  constraint_types = []

  """###Takes input from the input.txt files and stores into np arrays"""

  with open("input_ilp.txt",'r') as input:
      current_section = None
      for line in input:
          line = line.strip()
          if line.startswith('[') and line.endswith(']'):
            current_section = line[1:-1]
          else:
            if not line:
              continue
            elif current_section == 'objective':
              objective = line
              initial_objective = objective
            elif current_section == 'A':
              temp = line.split(',')
              row = []
              for i in temp:
                i = i.strip()
                if i!="":
                  row.append(Fraction(i))
              A.append(row)
            elif current_section == 'b':
              if b!="":
                b.append(Fraction(line))
            elif current_section == 'c':
              temp = line.split(',')
              for i in temp:
                i = i.strip()
                if i!="":
                  c.append(Fraction(i))
            elif current_section == 'constraint_types':
              constraint_types.append(line)
  A = np.array(A)
  global number_of_initial_var
  number_of_initial_var = A.shape[1]
  b = np.array(b)
  c = np.array(c)

  """### Format the input data to fit specifications
  if maximise then switch signs of c to -ve to change to minimization
  """

  if objective == 'maximize':
    c = c*(-1)
    objective = 'minimize'

  """
  #### Slack variables
  1. for `>=` new slack variable to the right i.e. new column with `-1` at that row
  2. for `<=` new slack variable to the left i.e. new column with `1` at that row
  3. in the case that `b == 0`, `A_row == e_i`, and `constraint == '<='` a `column == -xi` has to be inserted"""

  count_slack_variable = 0
  for i,constraint in enumerate(constraint_types):
    if constraint == ">=": ## >= -> new slack variable to the right -> new column with -1 at that row
      count_slack_variable += 1
      ei = np.zeros((A.shape[0],1),dtype=object)
      ei = ei + Fraction()
      ei[i,0] = Fraction(-1)
      A = np.hstack((A, ei))
    elif constraint == "<=":## <= -> new slack variable to the left -> new column with 1 at that row
      count_slack_variable += 1
      ei = np.zeros((A.shape[0],1),dtype=object)
      ei = ei + Fraction()
      ei[i,0] = Fraction(1)
      A = np.hstack((A, ei))

  slack_zero = np.zeros(count_slack_variable,dtype=object)
  slack_zero = slack_zero + Fraction()
  c = np.append(c , slack_zero)

  """#### If b is -ve for a row
  multiply the row with `-1` to make sure that the vector b is `>`
  """

  for i,temp in enumerate(b):
    if temp<0:
      A[i] = A[i]*-1
      b[i] = b[i]*-1
      
  #Check
  # print(A)
  # print(b)
  # print(c)
  # print(objective)
  # print(constraint_types)

  res = simplex_algo(A,b,c,objective,constraint_types)
  
  status = res["solution_status"]
  if status == "infeasible":
    print(f'initial_solution:',None)
    print(f'final_solution:',None)
    print('solution_status: infeasible')
    print('number_of_cuts: 0')
    print(f'optinal_vale:',None)
    return
  elif status == "unbounded":
    print(f'initial_solution:',None)
    print(f'final_solution:',None)
    print(f'solution_status:',status)
    print('number_of_cuts: 0')
    print(f'optinal_vale:',None)
    return

  global Tab, basis_idx, RHS, Cj, is_basic, x, solution_status
  Tab = None
  basis_idx = None
  RHS = None
  Cj = None
  is_basic = None
  x = None
  solution_status = None

  Tab = res['final_tableau']

  basis_idx = res['basis_idx_final']

  x = res['x_final']

  m = Tab.shape[0]
  # if basis_idx is None:
  #   m = 0
  # else :
  #   m = len(basis_idx)
  xb=None
  xb = np.zeros(m,dtype = object)
  xb = xb + Fraction()
  if res['solution_status'] == 'optimal' :
    for i in range(m) :
      xb[i] = x[basis_idx[i]]

  RHS = np.transpose(xb)

  Cj = c

  is_basic = np.array([False] * Tab.shape[1])
  for i in basis_idx:
    is_basic[i] = True

  global result
  result = {}
  result['final_tableau'] = Tab
  result['solution_status'] = res['solution_status']
  result['optimal_value'] = res['optimal_value']
  result['x_final'] = x
  result['basis_idx_final'] = basis_idx

  # print('TAB : ')
  # print(Tab)
  # print(' ')
  # print('basis_idx : ')
  # print(basis_idx)
  # print(' ')
  # print("RHS : ")
  # print(RHS)
  # print(' ')
  # print('C : ')
  # print(Cj)
  # print(' ')
  # print('is basic : ')
  # print(is_basic)
  # print(' ')
  # print('x_final : ')
  # print(x)


  """Gomory Cut method

  """
  # print("simplex")
  # print('Tabi : ')
  # print(Tab)
  # print('basis_idx : ')
  # print(basis_idx)
  # print('RHS : ')
  # print(RHS)
  # print('x : ')
  # print(x)
  # print('is_basic : ')
  # print(is_basic)
  # print('Cj')
  # print(Cj)

  final_ans = None

  # print("jo")
  # print(" ")
  # print(" ")
  # print(" ")
  # print(" ")
  # print("jo")

  op = 0
  feasible = True
  cnt = 0
  while(check(RHS) != -1 and feasible == True):
    new = []
    source_no = check(RHS)
    cnt = cnt + 1
    for i in range(len(Tab[0])):
      frac = Fraction(Tab[source_no][i] - int(math.floor(Tab[source_no][i])),1)
      new.append(-1*frac)
    new.append(Fraction(1))
    Tab = Tab.tolist()
    Tab.append(new)


    for i in range(len(Tab) - 1):
      Tab[i].append(Fraction(0))

    Tab = np.array(Tab)
    basis_idx = np.append(basis_idx,len(x))
    frac = Fraction(RHS[source_no] - int(math.floor(RHS[source_no])))
    RHS = np.append(RHS,-1*frac)
    x = np.append(x,-1*frac)
    is_basic = np.append(is_basic,True)
    Cj = np.append(Cj,Fraction(0))

    # if(cnt == 1 or cnt == 2):
      # print('Tab : ')
      # print(Tab)
      # print('basis_idx : ')
      # print(basis_idx)
      # print('RHS : ')
      # print(RHS)
      # print('x : ')
      # print(x)
      # print('is_basic : ')
      # print(is_basic)
      # print('Cj')
      # print(Cj)





    # print(" ")
    # print(" ")
    # print(" ")
    # print(" ")
    # print("dual simplex starts")


    # the dual simplex loop access and also edits the following things:
    # global Tab, basis_idx, RHS, Cj, is_basic, x
    result = dual_simplex_loop()


    # print("dual simplex ends")
    # print(" ")
    # print(" ")
    # print(" ")
    # print(" ")

    if(cnt == 1 or cnt == 2):
      # print(f'optimal cost = ')
      # print(result['optimal_value'])
      final_ans = result['optimal_value']
      # print('Tab : ')
      # print(Tab)
      # print('basis_idx : ')
      # print(basis_idx)
      # print('RHS : ')
      # print(RHS)
      # print('x : ')
      # print(x)
      # print('is_basic : ')
      # print(is_basic)
      # print('Cj')
      # print(Cj)
      
    solution_status = result['solution_status']
                            # optimal or unbounded
                            # (we are expecting that feasaibility is sured)
                            # because when we call dual simplex it is assumed
                            # cj-zj >= 0 for all columns in the tablue
                            # and cj-zj >= 0 is maintained throughout dual simplex

    optimal_value = result['optimal_value'] # C*X cost at optimal

    if(solution_status == "unbounded"):
      feasible = False

  # print(f'optimal cost = ')
  # print(result['optimal_value'])
  final_ans = result['optimal_value']
  # print('Tab : ')
  # print(Tab)
  # print('basis_idx : ')
  # print(basis_idx)
  # print('RHS : ')
  # print(RHS)
  # print('x : ')
  # print(x)
  # print('is_basic : ')
  # print(is_basic)
  # print('Cj')
  # print(Cj)

  if result['solution_status'] == 'optimal':
    if initial_objective == 'maximize' and feasible == True:
      final_ans = final_ans*(-1)
  # else:
  #   print(result['solution_status'])
  # print(" ")
  # print(" ")
  # print(" ")
  # print(f'final optimal cost = {final_ans}')
  
  if res['x_final'] is None:
    initial_solution = None
    print('initial_solution: ',end = "")
    print(initial_solution)
  else:
    initial_solution = (res['x_final'])[0:(number_of_initial_var)]
    print('initial_solution: ',end = "")
    for i in range(len(initial_solution)-1):
      print(float(initial_solution[i]),end=", ")
    print(float(initial_solution[-1]))
    
  if x is None:
    final_solution = None
    print('final_solution: ',end = "")
    print(final_solution)
  else:
    final_solution = x[0:(number_of_initial_var)]
    print('final_solution: ',end = "")
    for i in range(len(final_solution)-1):
      print(int(final_solution[i]),end=", ")
    print(int(final_solution[-1]))
    
  #solution_status == solution_status
  number_of_cuts = cnt
  optimal_value = final_ans
  
  print(f'solution_status: {solution_status}')
  print(f'number_of_cuts: {number_of_cuts}')
  print(f'optimal_value: {optimal_value}')
  