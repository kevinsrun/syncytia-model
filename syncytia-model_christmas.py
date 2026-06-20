import numpy as np
import scipy
import csv
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.integrate import odeint

file = open(r'Fig.csvs\Fig2_mock_TF.csv')
type(file)
csvreader = csv.reader(file)
header_Fig2_mock_TF = []
header_Fig2_mock_TF = next(csvreader)
rows_Fig2_mock_TF = []
for row in csvreader:
    rows_Fig2_mock_TF.append(row)
file.close

mock = np.array(rows_Fig2_mock_TF)
mock_flt = mock.astype(float)
arr_sorted_mock = sorted(mock_flt,key = lambda x: x[0])
arr_mock_new = np.array(arr_sorted_mock)
tr_arr_mock_new = np.transpose(arr_mock_new)
sublist1_mock, sublist2_mock = tr_arr_mock_new.tolist()
key = sublist2_mock[0]
for i in range(0,21):
    sublist2_mock[i] = sublist2_mock[i] - key
sublist2_mock[0] = 0.0
final_mock = np.array(list(zip(sublist1_mock, sublist2_mock)))
#Normalize the cell index values to be between 0 and 1 by finding largest value in the cell index column and dividing all values by that number
final_mock[:,1] = final_mock[:,1] / np.max(final_mock[:,1])

file = open(r'Fig.csvs\Fig2A_spike-TF.csv')
type(file)
csvreader = csv.reader(file)
header_Fig2A_spike_TF = []
header_Fig2A_spike_TF = next(csvreader)
rows_Fig2A_spike_TF = []
for row in csvreader:
    rows_Fig2A_spike_TF.append(row)
file.close

spike = np.array(rows_Fig2A_spike_TF)
spike_flt = spike.astype(float)
arr_sorted_spike = sorted(spike_flt,key = lambda x: x[0])
arr_spike_new = np.array(arr_sorted_spike)
tr_arr_spike_new = np.transpose(arr_spike_new)
sublist1_spike, sublist2_spike = tr_arr_spike_new.tolist()
key = sublist2_spike[0]
for i in range(0,21):
    sublist2_spike[i] = sublist2_spike[i] - key
sublist2_spike[0] = 0.0
final_spike = np.array(list(zip(sublist1_spike, sublist2_spike)))
#Normalize the cell index values to be between 0 and 1 by finding largest value in the cell index column and dividing all values by that number
final_spike[:,1] = final_spike[:,1] / np.max(final_spike[:,1])

arr_mock_new = np.array(arr_sorted_mock)

_, unique_idx = np.unique(arr_mock_new[:,0], return_index=True)
arr_mock_new = arr_mock_new[np.sort(unique_idx)]

def ode_list(li,t,gamma,k_val,sigma):
    D = li[0]
    A = li[1]
    F1 = li[2]
    F2 = li[3]
    S = li[4]
   
    dDdt = (-gamma*D*A) - sigma*D
    dAdt = (-gamma*D*A) - (gamma*S*A) - sigma*A
    dF1dt = (2*gamma*D*A)+(gamma*S*A)-(k_val*F1) - sigma*F1
    dF2dt = (k_val*F1)-(k_val*F2) - sigma*F2
    dSdt = (k_val*F2) - sigma*S
   
    return [dDdt, dAdt, dF1dt, dF2dt, dSdt]

def SSR_code_mock(li):
    t = final_mock[:, 0]
   
    donor_SSR_mock = li[0]
    gamma_SSR_mock = li[1]
    k_val_SSR_mock = li[2]
    sigma_SSR_mock = li[3]
   
    var = [donor_SSR_mock, 1-donor_SSR_mock, 0, 0, 0]
    check = final_mock[:,1]

    if sigma_SSR_mock < 0:
        return 1e12

    return_values_mock = odeint(ode_list, var, t, args = (gamma_SSR_mock, k_val_SSR_mock, sigma_SSR_mock))
    y_predicted_val_mock = return_values_mock[:,4]/(return_values_mock[:,0] + return_values_mock[:,1] + return_values_mock[:,2] + return_values_mock[:,3])
    SSR_mock = sum((y_predicted_val_mock - check)**2)
    return SSR_mock

def SSR_code_spike(li):
    t = final_spike[:, 0]
   
    donor_SSR_spike = li[0]
    gamma_SSR_spike = li[1]
    k_val_SSR_spike = li[2]
    sigma_SSR_spike = li[3]
   
    var = [donor_SSR_spike, 1-donor_SSR_spike, 0.1, 0, 0]
    check = final_spike[:,1]
   
   #No negatives plz or absurbly high values
    if donor_SSR_spike <= 0 or donor_SSR_spike >= 1:
        return 1e12

    if sigma_SSR_spike < 0:
        return 1e12

    return_values_spike = odeint(ode_list, var, t, args = (gamma_SSR_spike, k_val_SSR_spike, sigma_SSR_spike))
    y_predicted_val_spike = return_values_spike[:,4]/(return_values_spike[:,0] + return_values_spike[:,1] + return_values_spike[:,2] + return_values_spike[:,3])
    SSR_spike = sum((y_predicted_val_spike - check)**2)
    return SSR_spike

def true_y_values_mock(li):
    donor_y_mock = li.x[0]
    gamma_y_mock = li.x[1]
    k_val_y_mock = li.x[2]
    sigma_y_mock = li.x[3]
   
    t_mock = final_mock[:,0]
   
    true_vals_mock = odeint(ode_list,[donor_y_mock, 1-donor_y_mock, 0, 0, 0], t_mock, args = (gamma_y_mock, k_val_y_mock, sigma_y_mock))
    y_vals_mock = true_vals_mock[:,4] / (true_vals_mock[:,0] + true_vals_mock[:,1] + true_vals_mock[:,2] + true_vals_mock[:,3])
    return y_vals_mock

def true_y_values_spike(li):
    donor_y_spike = li.x[0]
    gamma_y_spike = li.x[1]
    k_val_y_spike = li.x[2]
    sigma_y_spike = li.x[3]
   
    t_spike = final_spike[:,0]
   
    true_vals_spike = odeint(ode_list,[donor_y_spike, 1-donor_y_spike, 0, 0, 0], t_spike, args = (gamma_y_spike, k_val_y_spike, sigma_y_spike))
    y_vals_spike = true_vals_spike[:,4] / (true_vals_spike[:,0] + true_vals_spike[:,1] + true_vals_spike[:,2] + true_vals_spike[:,3])
    return y_vals_spike

def chi_squared(li_1,li_2):
    chi = sum((li_1 - li_2)**2/li_1)
    return chi

def aic(li):
    SSR = opti_mock.fun
    return len(li)*np.log(SSR/len(li))+2*4

parameter_bounds = [(0.001, 0.999), (1e-8, 500.0), (1e-8, 500.0), (0.0, 10.0)]

initial_guess_mock = [0.74035, 199.3162, 0.10981, 0.30664]
opti_mock = scipy.optimize.minimize(SSR_code_mock, initial_guess_mock, method='Nelder-Mead', bounds=parameter_bounds, options={'maxiter':2000, 'xatol':1e-3, 'fatol':1e-3})
print(opti_mock)

initial_guess_spike = [0.42857, 0.26210, 0.19921, 0.23228]
opti_spike = scipy.optimize.minimize(SSR_code_spike, initial_guess_spike, method='Nelder-Mead', bounds=parameter_bounds, options={'maxiter':2000, 'xatol':1e-3, 'fatol':1e-3})
print(opti_spike)

gamma_mock = opti_mock.x[1]
k_val_mock = opti_mock.x[2]
observed_mock = true_y_values_mock(opti_mock)

gamma_spike = opti_spike.x[1]
k_val_spike = opti_spike.x[2]
sigma_spike = opti_spike.x[3]
observed_spike = true_y_values_spike(opti_spike)

plt.scatter(final_mock[:,0], final_mock[:,1])
plt.plot(final_mock[:,0], observed_mock)

plt.scatter(final_spike[:,0], final_spike[:,1])
plt.plot(final_spike[:,0], observed_spike)

plt.xlabel("Time (hr)", fontsize=10)
plt.ylabel("Cell Index (CI)", fontsize=10)
plt.title("Mock Vs. Spike", fontsize=10)

plt.show()

print('chi^2', chi_squared(final_mock[:,1][1:], observed_mock[1:]))
print('aic', aic(final_mock[:,1]))

# %% FIGURE 3A

# Fig3A_1_1
file = open(r'Fig.csvs\Fig3A_1_1.csv')
type(file)
csvreader = csv.reader(file)
header_Fig3A_1_1 = next(csvreader)
rows_Fig3A_1_1 = []
for row in csvreader:
    rows_Fig3A_1_1.append(row)
file.close

fig3A_1_1 = np.array(rows_Fig3A_1_1)
fig3A_1_1_flt = fig3A_1_1.astype(float)
arr_sorted_fig3A_1_1 = sorted(fig3A_1_1_flt, key=lambda x: x[0])
arr_fig3A_1_1_new = np.array(arr_sorted_fig3A_1_1)

tr_arr_fig3A_1_1_new = np.transpose(arr_fig3A_1_1_new)
time_fig3A_1_1, ci_fig3A_1_1 = tr_arr_fig3A_1_1_new.tolist()

key = ci_fig3A_1_1[0]
for i in range(len(ci_fig3A_1_1)):
    ci_fig3A_1_1[i] -= key
ci_fig3A_1_1[0] = 0.0

final_fig3A_1_1 = np.array(list(zip(time_fig3A_1_1, ci_fig3A_1_1)))
final_fig3A_1_1[:,1] = final_fig3A_1_1[:,1] / np.max(final_fig3A_1_1[:,1])


#Fig3A_1_2
file = open(r'Fig.csvs\Fig3A_1_2.csv')
type(file)
csvreader = csv.reader(file)
header_Fig3A_1_2 = next(csvreader)
rows_Fig3A_1_2 = []
for row in csvreader:
    rows_Fig3A_1_2.append(row)
file.close

fig3A_1_2 = np.array(rows_Fig3A_1_2)
fig3A_1_2_flt = fig3A_1_2.astype(float)
arr_sorted_fig3A_1_2 = sorted(fig3A_1_2_flt, key=lambda x: x[0])
arr_fig3A_1_2_new = np.array(arr_sorted_fig3A_1_2)

tr_arr_fig3A_1_2_new = np.transpose(arr_fig3A_1_2_new)
time_fig3A_1_2, ci_fig3A_1_2 = tr_arr_fig3A_1_2_new.tolist()

key = ci_fig3A_1_2[0]
for i in range(len(ci_fig3A_1_2)):
    ci_fig3A_1_2[i] -= key
ci_fig3A_1_2[0] = 0.0

final_fig3A_1_2 = np.array(list(zip(time_fig3A_1_2, ci_fig3A_1_2)))
final_fig3A_1_2[:,1] = final_fig3A_1_2[:,1] / np.max(final_fig3A_1_2[:,1])

#Fig3A_2_1
file = open(r'Fig.csvs\Fig3A_2_1.csv')
type(file)
csvreader = csv.reader(file)
header_Fig3A_2_1 = next(csvreader)
rows_Fig3A_2_1 = []
for row in csvreader:
    rows_Fig3A_2_1.append(row)
file.close

fig3A_2_1 = np.array(rows_Fig3A_2_1)
fig3A_2_1_flt = fig3A_2_1.astype(float)
arr_sorted_fig3A_2_1 = sorted(fig3A_2_1_flt, key=lambda x: x[0])
arr_fig3A_2_1_new = np.array(arr_sorted_fig3A_2_1)

tr_arr_fig3A_2_1_new = np.transpose(arr_fig3A_2_1_new)
time_fig3A_2_1, ci_fig3A_2_1 = tr_arr_fig3A_2_1_new.tolist()

key = ci_fig3A_2_1[0]
for i in range(len(ci_fig3A_2_1)):
    ci_fig3A_2_1[i] -= key
ci_fig3A_2_1[0] = 0.0

final_fig3A_2_1 = np.array(list(zip(time_fig3A_2_1, ci_fig3A_2_1)))
final_fig3A_2_1[:,1] = final_fig3A_2_1[:,1] / np.max(final_fig3A_2_1[:,1])

def true_y_values_fig3A_1_1(li):
    donor = li.x[0]
    gamma = li.x[1]
    k_val = li.x[2]
    sigma = li.x[3]

    t = final_fig3A_1_1[:,0]

    vals = odeint(ode_list, [donor, 1-donor, 0, 0, 0], t, args=(gamma, k_val, sigma))
    return vals[:,4] / (vals[:,0] + vals[:,1] + vals[:,2] + vals[:,3])

def true_y_values_fig3A_1_2(li):
    donor = li.x[0]
    gamma = li.x[1]
    k_val = li.x[2]
    sigma = li.x[3]

    t = final_fig3A_1_2[:,0]

    vals = odeint(ode_list, [donor, 1-donor, 0, 0, 0], t, args=(gamma, k_val, sigma))
    return vals[:,4] / (vals[:,0] + vals[:,1] + vals[:,2] + vals[:,3])

def true_y_values_fig3A_2_1(li):
    donor = li.x[0]
    gamma = li.x[1]
    k_val = li.x[2]
    sigma = li.x[3]

    t = final_fig3A_2_1[:,0]

    vals = odeint(ode_list, [donor, 1-donor, 0, 0, 0], t, args=(gamma, k_val, sigma))
    return vals[:,4] / (vals[:,0] + vals[:,1] + vals[:,2] + vals[:,3])

def SSR_code_fig3A_1_1(li):
    t = final_fig3A_1_1[:, 0]

    donor = li[0]
    gamma = li[1]
    k_val = li[2]
    sigma = li[3]

    var = [donor, 1-donor, 0, 0, 0]
    check = final_fig3A_1_1[:,1]

    if donor <= 0 or donor >= 1:
        return 1e12
    if sigma < 0:
        return 1e12

    return_vals = odeint(ode_list, var, t, args=(gamma, k_val, sigma))
    y_pred = return_vals[:,4] / (return_vals[:,0] + return_vals[:,1] + return_vals[:,2] + return_vals[:,3])

    return sum((y_pred - check)**2)

def SSR_code_fig3A_1_2(li):
    t = final_fig3A_1_2[:, 0]

    donor = li[0]
    gamma = li[1]
    k_val = li[2]
    sigma = li[3]

    var = [donor, 1-donor, 0, 0, 0]
    check = final_fig3A_1_2[:,1]

    if donor <= 0 or donor >= 1:
        return 1e12
    if sigma < 0:
        return 1e12

    return_vals = odeint(ode_list, var, t, args=(gamma, k_val, sigma))
    y_pred = return_vals[:,4] / (return_vals[:,0] + return_vals[:,1] + return_vals[:,2] + return_vals[:,3])

    return sum((y_pred - check)**2)

def SSR_code_fig3A_2_1(li):
    t = final_fig3A_2_1[:, 0]

    donor = li[0]
    gamma = li[1]
    k_val = li[2]
    sigma = li[3]

    var = [donor, 1-donor, 0, 0, 0]
    check = final_fig3A_2_1[:,1]

    if donor <= 0 or donor >= 1:
        return 1e12
    if sigma < 0:
        return 1e12

    return_vals = odeint(ode_list, var, t, args=(gamma, k_val, sigma))
    y_pred = return_vals[:,4] / (return_vals[:,0] + return_vals[:,1] + return_vals[:,2] + return_vals[:,3])

    return sum((y_pred - check)**2)

def true_y_values_fig3A_1_1(li):
    donor = li.x[0]
    gamma = li.x[1]
    k_val = li.x[2]
    sigma = li.x[3]

    t = final_fig3A_1_1[:,0]

    vals = odeint(ode_list, [donor, 1-donor, 0, 0, 0], t, args=(gamma, k_val, sigma))
    return vals[:,4] / (vals[:,0] + vals[:,1] + vals[:,2] + vals[:,3])

def true_y_values_fig3A_1_2(li):
    donor = li.x[0]
    gamma = li.x[1]
    k_val = li.x[2]
    sigma = li.x[3]

    t = final_fig3A_1_2[:,0]

    vals = odeint(ode_list, [donor, 1-donor, 0, 0, 0], t, args=(gamma, k_val, sigma))
    return vals[:,4] / (vals[:,0] + vals[:,1] + vals[:,2] + vals[:,3])

def true_y_values_fig3A_2_1(li):
    donor = li.x[0]
    gamma = li.x[1]
    k_val = li.x[2]
    sigma = li.x[3]

    t = final_fig3A_2_1[:,0]

    vals = odeint(ode_list, [donor, 1-donor, 0, 0, 0], t, args=(gamma, k_val, sigma))
    return vals[:,4] / (vals[:,0] + vals[:,1] + vals[:,2] + vals[:,3])

def chi_squared(li_1,li_2):
    chi = sum((li_1 - li_2)**2/li_1)
    return chi

def aic(li):
    SSR = opti_fig3A_1_1.fun
    return len(li)*np.log(SSR/len(li))+2*4


initial_guess_fig3A_1_1 = [0.89833, 105.6878, 11.79493, 4.28405]
opti_fig3A_1_1 = scipy.optimize.minimize(SSR_code_fig3A_1_1, initial_guess_fig3A_1_1, method='Nelder-Mead', bounds=parameter_bounds, options={'maxiter':2000, 'xatol':1e-3, 'fatol':1e-3})
print(opti_fig3A_1_1)

gamma_fig3A_1_1 = opti_fig3A_1_1.x[1]
k_val_fig3A_1_1 = opti_fig3A_1_1.x[2]
sigma_fig3A_1_1 = opti_fig3A_1_1.x[3]
observed_fig3A_1_1 = true_y_values_fig3A_1_1(opti_fig3A_1_1)

initial_guess_fig3A_1_2 = [0.94725, 141.0677, 12.03868, 4.98449]
opti_fig3A_1_2 = scipy.optimize.minimize(SSR_code_fig3A_1_2, initial_guess_fig3A_1_2, method='Nelder-Mead', bounds=parameter_bounds, options={'maxiter':2000, 'xatol':1e-3, 'fatol':1e-3})
print(opti_fig3A_1_2)

gamma_fig3A_1_2 = opti_fig3A_1_2.x[1]
k_val_fig3A_1_2 = opti_fig3A_1_2.x[2]
sigma_fig3A_1_2 = opti_fig3A_1_2.x[3]
observed_fig3A_1_2 = true_y_values_fig3A_1_2(opti_fig3A_1_2)

initial_guess_fig3A_2_1 = [0.88185, 92.68115, 4.18241, 1.69556]
opti_fig3A_2_1 = scipy.optimize.minimize(SSR_code_fig3A_2_1, initial_guess_fig3A_2_1, method='Nelder-Mead', bounds=parameter_bounds, options={'maxiter':2000, 'xatol':1e-3, 'fatol':1e-3})
print(opti_fig3A_2_1)

gamma_fig3A_2_1 = opti_fig3A_2_1.x[1]
k_val_fig3A_2_1 = opti_fig3A_2_1.x[2]
sigma_fig3A_2_1 = opti_fig3A_2_1.x[3]
observed_fig3A_2_1 = true_y_values_fig3A_2_1(opti_fig3A_2_1)

print(opti_fig3A_1_1)
print(opti_fig3A_1_2)
print(opti_fig3A_2_1)

plt.scatter(final_fig3A_1_1[:,0], final_fig3A_1_1[:,1])
plt.plot(final_fig3A_1_1[:,0], true_y_values_fig3A_1_1(opti_fig3A_1_1))

plt.scatter(final_fig3A_1_2[:,0], final_fig3A_1_2[:,1])
plt.plot(final_fig3A_1_2[:,0], true_y_values_fig3A_1_2(opti_fig3A_1_2))

plt.scatter(final_fig3A_2_1[:,0], final_fig3A_2_1[:,1])
plt.plot(final_fig3A_2_1[:,0], true_y_values_fig3A_2_1(opti_fig3A_2_1))

plt.legend()
plt.xlabel("Time (hr)")
plt.ylabel(" CI")
plt.title("3A")
plt.show()

print('chi^2', chi_squared(final_fig3A_1_1[:,1][1:], observed_fig3A_1_1[1:]))
print('aic', aic(final_fig3A_1_1[:,1]))

print('chi^2', chi_squared(final_fig3A_1_2[:,1][1:], observed_fig3A_1_2[1:]))
print('aic', aic(final_fig3A_1_2[:,1]))

print('chi^2', chi_squared(final_fig3A_2_1[:,1][1:], observed_fig3A_2_1[1:]))
print('aic', aic(final_fig3A_2_1[:,1]))
