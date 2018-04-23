# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 09:50:37 2018

@author: shipratg3
"""
#%%
def SVD(data):
    import numpy as np
    import matplotlib.pyplot as plt 
    from sklearn.metrics import mean_absolute_error as mae
    data=data.select_dtypes([np.number])
    data.dropna(how="all", inplace=True) # drops the empty line at file-end
    data_matrix=np.matrix(data)
    data_transpose=np.transpose(data_matrix)
    #A^T_A=V
    V=np.matmul(data_transpose,data_matrix)
    eig_vals_V, eig_vecs_V = np.linalg.eig(V)
    eig_val_index=eig_vals_V.argsort()[::-1]
    eig_val_V=eig_vals_V[eig_val_index]
    eig_vec_V=eig_vecs_V[:,eig_val_index]
    for i in range(len(V)):
            matrix_V=np.asmatrix(eig_vec_V)
            matrix_E=np.asmatrix(np.diag(eig_val_V))
            sqrt_matrix=np.sqrt(matrix_E)
            A_V_matrix=np.matmul(data_matrix,matrix_V[i:i+1,:].T)
            u=np.divide(A_V_matrix,sqrt_matrix[i:i+1,i:i+1])
            if(i==0):
                matrix_U=u
            else:
                matrix_U=np.concatenate((matrix_U,u),axis=1)
    error=[]
    for i in range(min(len(matrix_U),len(matrix_V))):
        U_1=matrix_U[:,:i+1]
        E_1=sqrt_matrix[:i+1,:i+1]
        V_1=matrix_V[:i+1,:]
        decomp_matrix=np.matmul(U_1,np.matmul(E_1,V_1))
        error.append(mae(data_matrix,decomp_matrix))
    print("The final decomposed matrix in UE(V.T):\n",decomp_matrix,"\n")
    print("The error in decompostion:\n",error,"\n\n")
    n=min(len(matrix_U),len(matrix_V))+1
    plt.title("The error plot of each term: ") 
    plt.plot(range(1,n),error,color='red', marker='s',markerfacecolor='green', markersize=8)
 