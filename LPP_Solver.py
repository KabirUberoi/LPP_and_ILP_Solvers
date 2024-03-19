import numpy as np

"""# Define Functions for Tableau

## Cb

`get_cb` calculates Cb (cost coef. of basic variables)
"""

def get_cb(c , b_idx) :
  m = len(b_idx)
  cb = np.zeros(m)
  for i in range(m) :
    cb[i] = c[b_idx[i]]
  return cb

"""## C_bar

C_bar = C_new - Cb dot tableau

don't take transpose as Cb is already row vector
"""

def get_C_bar(c , cb , tab) :
  return c - np.dot(cb, tab)

"""## Pivot col index (ENTERING VARIABLE)
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

"""## Pivot row index (LEAVING VARIABLE)
this function returns the row number (0 based indexing) of pivot column in Tableau

process same as class/book

also returns `theta` that is used for calculating next bfs

`x_next` = `x` + `theta`*`d`
"""

def get_pivot_row_idx(pivot_col , x , b_idx) :
  # pivot_col is a vertical vector containing the col corresponding to pvt_col_idx
  l = len(b_idx)
  theta = float('inf')
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

"""## Next Tableau
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

"""## PRINT TABLEAU
The format of the tableau will be like a matrix having m rows, where m is the number of equations
given in constraints and the columns will be as follows: first column for the values of current BFS,
followed by the variables in optimization problem, followed by slack/surplus variables, and finally
artificial variables.
"""

def print_tableau(tab , x , b_idx) :
  m = len(b_idx)
  xb = np.zeros((m , 1))
  for i in range(m) :
    xb[i] = x[b_idx[i]]

  tableau = np.hstack((xb, tab))
  # print(tableau,'\n')
  return tableau

"""# Tableau function"""

def Tableau(tab , x , b_idx , c) :


  # print("Before Start, Tab = ")
  # print("")
  # print(tab)
  # print("")
  # print(f"x = {x}")
  # print(f"b_idx = {b_idx}")
  # print(f"c = {c}")
  # print("=====================")


  cb = get_cb(c , b_idx)
  c_bar = get_C_bar(c , cb , tab)


  pvt_col_idx = get_pivot_col_idx(c_bar)

  is_infinity = False

  while pvt_col_idx > -1 :

    # print("for above TAB")
    # print(f"pvt_col_idx = {pvt_col_idx}")

    pvt_col = np.transpose(tab[: , pvt_col_idx])

    pvt_row_idx , theta = get_pivot_row_idx(pvt_col , x , b_idx)

    # print(f"pvt_row_idx = {pvt_row_idx}")
    # print("=====================")

    if pvt_row_idx == -1 :
      is_infinity = True
      # print("UNBOUNDED")
      break

    l = len(x)

    d = np.zeros(l)

    d[pvt_col_idx] = 1

    for i in range(len(b_idx)) :
      d[b_idx[i]] = - tab[i,pvt_col_idx]

    tab = get_nxt_tab(tab , pvt_row_idx , pvt_col_idx)
    # print("Tab after itr : ")
    # print("")
    # print(tab)
    # print("")

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

    # print("=====================")


  return tab , x , b_idx , is_infinity

def simplex_algo():
    """# taking Input from file

    ### Declaration of required variables
    """

    A = []
    b = []
    c = []
    objective = None
    constraint_types = []

    """##Takes input from the input.txt files and stores into np arrays"""

    with open("input.txt",'r') as input:
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
                elif current_section == 'A':
                    temp = line.split(',')
                    row = []
                    for i in temp:
                        i = i.strip()
                        if i:
                            row.append(float(i))
                    A.append(row)
                elif current_section == 'b':
                    b.append(float(line))
                elif current_section == 'c':
                    temp = line.split(',')
                    for i in temp:
                        i = i.strip()
                        if i:
                            c.append(int(i))
                elif current_section == 'constraint_types':
                    constraint_types.append(line)

    A = np.array(A)
    b = np.array(b)
    c = np.array(c)

    """## Format the input data to fit specifications

    ###if maximise then switch signs of c to -ve to change to minimization
    """

    if objective == 'maximize':
        c = c*(-1)

    """### Adjusting for free variables

    """

    def is_ei(row): ## Checking if the inputted row is an e_i vector or not
        count = 0
        idx = 0
        for i,val in enumerate(row):
            if val==0:
                count+=1
            else:
                idx = i
        if count == (len(row)-1):
            return True,idx
        else:
            return False,-1

    #Check
    # print(A)
    # print(b)
    # print(c)
    # print(objective)
    # print(constraint_types)

    number_of_initial_var = A.shape[1]
    
    # free_checker = np.zeros(number_of_initial_var)  # for keeping track of which of the variables are free
    # split_var = np.zeros(number_of_initial_var, dtype=int)
    # time = 1

    # for i,RHS in enumerate(b): ## to handle the x_i <= 0 case
    #     if RHS == 0:
    #         flag,idx = is_ei(A[i])
    #         if flag == True:
    #             free_checker[idx] = 1;
    #             if constraint_types[i] == "<=":
    #                 negated_A = (-1 * A[:, idx]).reshape((-1, 1))
    #                 A = np.hstack((A, negated_A))
    #                 c = np.append(c, (-1) * c[idx])
    #                 split_var[idx] = time
    #                 time+=1

    # for idx,val in enumerate(free_checker):
    #     if val == 0:
    #         negated_column = (-1 * A[:, idx]).reshape((-1, 1))
    #         A = np.hstack((A, negated_column))
    #         c = np.append(c, (-1) * c[idx])
    #         split_var[idx] = time
    #         time+=1

    # print(A)

    """### Slack variables
    1. for `>=` new slack variable to the right i.e. new column with `-1` at that row
    2. for `<=` new slack variable to the left i.e. new column with `1` at that row
    3. in the case that `b == 0`, `A_row == e_i`, and `constraint == '<='` a `column == -xi` has to be inserted
    """

    count_slack_variable = 0
    for i,constraint in enumerate(constraint_types):
        if constraint == ">=": ## >= -> new slack variable to the right -> new column with -1 at that row
            count_slack_variable += 1
            ei = np.zeros((A.shape[0],1))
            ei[i,0] = -1
            A = np.hstack((A, ei))
        elif constraint == "<=":## <= -> new slack variable to the left -> new column with 1 at that row
            count_slack_variable += 1
            ei = np.zeros((A.shape[0],1))
            ei[i,0] = 1
            A = np.hstack((A, ei))

    c = np.append(c , np.zeros(count_slack_variable))

    """#### If b is -ve for a row
    multiply the row with `-1` to make sure that the vector b is `>`
    """

    for i,temp in enumerate(b):
        if temp<0:
            A[i] = A[i]*-1
            b[i] = b[i]*-1

    """1. Define a function called tableau and carry out the steps there too maybe

    2. After formating, find whether basic sol exists or not by adding "m" slack variables and running tableau on it

    3. Then using that basic solution, work on the OG tableau

    4. Then using the final Tableau, do the outputs as needed in the form of a dictionary
    """

    #Check
    # print(A)
    # print(b)
    # print(c)
    # print(objective)
    # print(constraint_types)

    """# Creating Auxiliary LPP"""

    m , n = A.shape
    A_aux = np.hstack((A , np.eye(m)))

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

    x_aux = np.zeros(n_aux)
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
    # print(f"cost of Auxilary LPP = {cost_aux}")

    """If the optimal cost in the auxiliary problem is positive, the original problem is infeasible and the algorithm terminates."""
    solution_status = ""
    if(cost_aux >= 0.001) :
        #   print("INFEASIBLE")
        solution_status = "infeasible"
        # else :
        #   print("THERE EXISTS SOLUTION")

    """## Eliminating y variables out the phase 1 tableau

    """

    if(cost_aux < 0.001):
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
    if(cost_aux < 0.001):
        A_f ,  x_f , b_f , is_infinity = Tableau(A_aux[:, :n] , x_aux[:n] , b_idx_aux , c)
        optimal_cost=np.dot(c,x_f)
        if(is_infinity) :
            # print("UNBOUNDED")
            solution_status = "unbounded"
        else :
            # print(optimal_cost)
            solution_status = "optimal"

    #print(split_var)
    # x_final = np.zeros(number_of_initial_var)
    # for i,val in enumerate(split_var):
    #     if(val!=0):
    #         x_final[i] = x_f[i] - x_f[number_of_initial_var - 1 + val]
    #     else:
    #         x_final[i] = x_f[i]

    if(cost_aux < 0.001):
        x_final = x_f[0:(number_of_initial_var)]
    else:
        x_final = None

    #solution_status already exists
    initial_tableau = tableau
    result = {}

    if(solution_status == "infeasible"):
        final_tableau = print_tableau(A_aux,x_aux,b_idx_aux)
        optimal_solution = None
        optimal_value = None

        result['initial_tableau'] = initial_tableau
        result['final_tableau'] = final_tableau
        result['solution_status'] = solution_status
        result['optimal_value'] = optimal_value
        result['optimal_solution'] = optimal_solution


    elif(solution_status == "optimal"):
        final_tableau = print_tableau(A_f,x_f,b_f)
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
        result['optimal_solution'] = optimal_solution
        
    elif(solution_status == "unbounded"):
        final_tableau = print_tableau(A_f,x_f,b_f)
        optimal_solution = None
        optimal_value = None

        result['initial_tableau'] = initial_tableau
        result['final_tableau'] = final_tableau
        result['solution_status'] = solution_status
        result['optimal_value'] = optimal_value
        result['optimal_solution'] = optimal_solution
        
    return result