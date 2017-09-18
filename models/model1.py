import numpy
import random
# import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt


def SIRD(beta, gamma, sigma, e_index, S_0, I_0, R_0, D_0, t):
    # host
    S_h = numpy.zeros(t)
    I_h = numpy.zeros(t)
    R_h = numpy.zeros(t)
    D_h = numpy.zeros(t)
    D_ht = numpy.zeros(t)
    # vector
    S_v = numpy.zeros(t)
    I_v = numpy.zeros(t)
    # initial conditions
    # host
    S_h[0] = S_0
    I_h[0] = I_0
    R_h[0] = R_0
    D_h[0] = D_0
    D_ht[0] = D_0
    # vector
    S_v[0] = e_index * (S_0 + R_0)
    I_v[0] = e_index * I_0

    for i in range(1, t):
        N_h = S_h[i - 1] + I_h[i - 1] + R_h[i - 1]

        n_i = min(S_h[i - 1], beta * S_h[i - 1] * I_v[i - 1] / N_h)
        n_rm = min(I_h[i - 1], gamma * I_h[i - 1])
        n_d = sigma * n_rm
        n_rv = n_rm - n_d

        S_h[i] = S_h[i - 1] - n_i
        I_h[i] = I_h[i - 1] + n_i - n_rm
        R_h[i] = R_h[i - 1] + n_rv
        D_h[i] = n_d
        D_ht[i] = D_ht[i - 1] + n_d

        e_k = e_index * N_h
        if (S_v[i - 1] + I_v[i - 1] / e_k) < 1:
            e_births = 0.111 * S_v[i - 1] * (1 - (S_v[i - 1] + I_v[i - 1]) / e_k)
        elif (S_v[i - 1] + I_v[i - 1] / e_k) > 1:
            e_births = 0.
            n_i_e = beta * S_v[i - 1] * I_h[i - 1] / N_h
            n_r_e = 0.33 * I_v[i - 1]

        S_v[i] = S_v[i - 1] + e_births - n_i_e
        I_v[i] = I_v[i - 1] + n_i_e - n_r_e

    return S_h, I_h, R_h, D_ht


def run():
    plt.style.use('ggplot')
    plt.rc('font', size=16)
    plt.rc('lines', linewidth=2)
    plt.rc('figure', autolayout=True)

    beta = .065
    gamma = .10
    sigma = .6
    e_index = 15.0

    S_0 = 25000.0
    I_0 = 9.0
    R_0 = 0.0
    D_0 = 0.0
    sim_time = numpy.array([random.randrange(0, 15) for _ in range(250)])
    t = len(sim_time)

    S, I, R, D = SIRD(beta, gamma, sigma, e_index, S_0, I_0, R_0, D_0, t)

    plt.xlabel('time')
    plt.ylabel('number of people')
    plt.plot(S, label='Suseptable')
    plt.plot(I, label='Infected')
    plt.plot(R, label='Recovered')
    plt.plot(D, label='Dead')
    plt.legend(loc='best')
    # plt.savefig('SIRD_model.png')
    plt.show()
