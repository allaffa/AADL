import torch
import time
import random

def anderson_qr_factorization(X, relaxation=1.0, regularization = 0.0):
    # Anderson Acceleration
    # Take a matrix X of iterates such that X[:,i] = g(X[:,i-1])
    # Return acceleration for X[:,-1]

    assert X.ndim==2, "X must be a matrix"
    assert regularization >= 0.0, "regularization for least-squares must be >=0.0"

    # Compute residuals
    DX =  X[:,1:] -  X[:,:-1] # DX[:,i] =  X[:,i+1] -  X[:,i]
    DR = DX[:,1:] - DX[:,:-1] # DR[:,i] = DX[:,i+1] - DX[:,i] = X[:,i+2] - 2*X[:,i+1] + X[:,i]

    if regularization == 0.0:
       # solve unconstrained least-squares problem
       start = time.time()
       gamma = torch.linalg.lstsq(DR, DX[:, -1]).solution
       finish = time.time()
    else:
       # solve augmented least-squares for Tykhonov regularization
       rhs = DX[:,-1]
       expanded_rhs    = torch.cat( (rhs, torch.zeros(DR.size(1))) )
       expanded_matrix = torch.cat( (DR, torch.sqrt(torch.tensor(regularization)) * torch.eye(DR.size(1))) )
       start = time.perf_counter()
       gamma = torch.linalg.lstsq(expanded_matrix, expanded_rhs).solution
       finish = time.perf_counter()

    #print("Time: ", finish-start)

    # compute acceleration
    extr = X[:,-2] + DX[:,-1] - (DX[:,:-1]+DR)@gamma

    if relaxation!=1:
        assert relaxation>0, "relaxation must be positive"
        # compute solution of the contraint optimization problem s.t. gamma = X[:,1:]@alpha
        alpha = torch.zeros(gamma.numel()+1).to(DX.device)
        alpha[0]    = gamma[0]
        alpha[1:-1] = gamma[1:] - gamma[:-1]
        alpha[-1]   = 1 - gamma[-1]
        extr = relaxation*extr + (1-relaxation)*X[:,:-1]@alpha

    return extr

def anderson_qr_factorization_reduced(X, relaxation=1.0, regularization = 0.0, reduction_type="random"):
    # Anderson Acceleration
    # Take a matrix X of iterates such that X[:,i] = g(X[:,i-1])
    # Return acceleration for X[:,-1]

    assert X.ndim==2, "X must be a matrix"
    assert regularization >= 0.0, "regularization for least-squares must be >=0.0"

    init_ratio = 0.01
    max_ratio = 0.20

    reduction_ratio = init_ratio

    # Compute residuals
    DX =  X[:,1:] -  X[:,:-1] # DX[:,i] =  X[:,i+1] -  X[:,i]
    DR = DX[:,1:] - DX[:,:-1] # DR[:,i] = DX[:,i+1] - DX[:,i] = X[:,i+2] - 2*X[:,i+1] + X[:,i]

    if regularization == 0.0:
       # solve unconstrained least-squares problem
       if reduction_type == "maximum":
          num_entries_kept = int(DX.shape[0] * reduction_ratio)
          indices = torch.topk(DX[:, -1], num_entries_kept).indices
          restricted_residual = torch.zeros(DX.shape[0],)
          restricted_residual[indices] = DX[indices, -1]
          while torch.norm(DX[:, -1]-restricted_residual) > (1/200*(torch.norm(DX[:, -1])**2) * 1e-8):
              reduction_ratio = reduction_ratio * 2
              if reduction_ratio > max_ratio:
                 reduction_ratio = max_ratio
                 break
              num_entries_kept = int(DX.shape[0] * reduction_ratio)
              indices = torch.topk(DX[:, -1], num_entries_kept).indices
              restricted_residual = torch.zeros(DX.shape[0],)
              restricted_residual[indices] = DX[indices, -1]
       elif reduction_type == "random":
          num_entries_kept = int(DX.shape[0] * reduction_ratio)
          start = time.perf_counter()
          #indices = random.sample(range(0, int(DX.shape[0])), num_entries_kept)
          indices = list(range(0, int(DX.shape[0])))
          indices = random.choices(indices, k=num_entries_kept)
          finish = time.perf_counter()
          restricted_residual = torch.zeros(DX.shape[0],)
          restricted_residual[indices] = DX[indices, -1]
          while torch.norm(DX[:, -1]-restricted_residual) > (1/200*(torch.norm(DX[:, -1])**2) * 1e-8):
              reduction_ratio = reduction_ratio * 2
              if reduction_ratio > max_ratio:
                 reduction_ratio = max_ratio
                 break
              #indices = random.sample(range(0, int(DX.shape[0])), num_entries_kept)
              indices = list(range(0, int(DX.shape[0])))
              indices = random.choices(indices, k=num_entries_kept)
              finish = time.perf_counter()
              restricted_residual = torch.zeros(DX.shape[0],)
              restricted_residual[indices] = DX[indices, -1]
          #print("Time for generation of random numbers: ", finish-start)
       else:
          raise ValueError("reduction type NOT recognized: ")

       start = time.perf_counter()
       gamma = torch.linalg.lstsq(DR[indices, :], DX[indices, -1]).solution
       finish = time.perf_counter()
       #print("Time to solve the least-squares: ", finish - start)
    else:
       # solve augmented least-squares for Tykhonov regularization
       rhs = DX[:,-1]
       expanded_rhs    = torch.cat( (rhs, torch.zeros(DR.size(1))) )
       expanded_matrix = torch.cat( (DR, torch.sqrt(torch.tensor(regularization)) * torch.eye(DR.size(1))) )
       gamma = torch.linalg.lstsq(expanded_matrix, expanded_rhs).solution

    # compute acceleration
    extr = X[:,-2] + DX[:,-1] - (DX[:,:-1]+DR)@gamma

    if relaxation!=1:
        assert relaxation>0, "relaxation must be positive"
        # compute solution of the contraint optimization problem s.t. gamma = X[:,1:]@alpha
        alpha = torch.zeros(gamma.numel()+1).to(DX.device)
        alpha[0]    = gamma[0]
        alpha[1:-1] = gamma[1:] - gamma[:-1]
        alpha[-1]   = 1 - gamma[-1]
        extr = relaxation*extr + (1-relaxation)*X[:,:-1]@alpha

    return extr


def anderson_normal_equation(X, relaxation=1.0, regularization = 0.0):
    # Anderson Acceleration
    # Take a matrix X of iterates such that X[:,i] = g(X[:,i-1])
    # Return acceleration for X[:,-1]

    assert X.ndim==2, "X must be a matrix"
    assert regularization >= 0.0, "regularization for least-squares must be >=0.0"

    # Compute residuals
    DX =  X[:,1:] -  X[:,:-1] # DX[:,i] =  X[:,i+1] -  X[:,i]
    DR = DX[:,1:] - DX[:,:-1] # DR[:,i] = DX[:,i+1] - DX[:,i] = X[:,i+2] - 2*X[:,i+1] + X[:,i]

    # # use QR factorization
    # q, r = torch.qr(DR)
    # gamma, _ = torch.triangular_solve( (q.t()@DX[:,-1]).unsqueeze(1), r )
    # gamma = gamma.squeeze(1)

    # solve unconstrained least-squares problem
    if regularization != 0.0: 
       RR = DR.t()@DR + regularization * torch.eye(DR.size(1))
    else: 
       RR = DR.t()@DR

    projected_residual = DR.t()@DX[:,-1].unsqueeze(1)
    
    gamma = torch.linalg.solve( RR, projected_residual )
    gamma = gamma.view(-1)

    # compute acceleration
    extr = X[:,-2] + DX[:,-1] - (DX[:,:-1]+DR)@gamma

    if relaxation!=1:
        assert relaxation>0, "relaxation must be positive"
        # compute solution of the contraint optimization problem s.t. gamma = X[:,1:]@alpha
        alpha = torch.zeros(gamma.numel()+1).to(DX.device)
        alpha[0]    = gamma[0]
        alpha[1:-1] = gamma[1:] - gamma[:-1]
        alpha[-1]   = 1 - gamma[-1]
        extr = relaxation*extr + (1-relaxation)*X[:,:-1]@alpha

    return extr
