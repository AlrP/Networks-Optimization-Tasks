import scipy.stats as st
import numpy as np
import tqdm
import matplotlib.pyplot as plt

def get_second_norm_dict():
    return dict([(k,v) for k,v in zip([x for x in range(1,24*3600+1)],
                                np.linspace(0.1, 1, 24*3600+1))])

def get_mean_time_list(mean_time, a, b):
    seconds_normed_dict = get_second_norm_dict()
    x = np.linspace(st.beta.ppf(0.01, a, b),
                    st.beta.ppf(0.99, a, b), 500)
    rv = st.beta(a, b)
    max_ = max(rv.pdf(x))
    mean_time_list = []
    for i in tqdm.tqdm(range(1,24*3600+1)):
        mean_time_list.append((rv.pdf(seconds_normed_dict[i])/max_)*mean_time)
    return mean_time_list

def get_time_arrays_dict(nodes_time_dependent_coefs):
    time_arrays_dict = {}
    print('Расчет зависимости интервалов между заказами от суточного времени')
    for k,v in nodes_time_dependent_coefs.items():
        print(f'Node {k}')
        time_arrays_dict.update({k:get_mean_time_list(v['mean'], v['a'], v['b'])})
    return time_arrays_dict

def plot_time_between_claims(time_arrays_dict, figsize=(10,3)):
    nplots = len(time_arrays_dict.keys())
    """
    График зависимость между интевалами заказов и общим временем для узлов графа
    """
    fig, axes = plt.subplots(nrows=1, ncols=nplots, figsize=figsize)
    plt.suptitle(r'Зависимость межзаказного времени $t_{customer}$ от суток')
    nodes = 3
    for ax, name, k in zip(axes, 
                          [f'Node {i+1}' for i in range(nodes)],
                          [i for i in range(1,nplots+1)]):
            ax.plot(time_arrays_dict[k], lw=2)
            ax.set_title(name)
            ax.set_xlabel('Model Time')
            ax.set_ylabel('Time between claims')

    fig.tight_layout()
    fig.subplots_adjust(top=0.8)
    plt.show()

def normalized_sigmoid_fkt(a, b, x):
    s = 1/(1+np.exp(-b*(x-a)))
    return s

def plot_agent_service_time(working_space, time_between_service, a, b):
    fig = plt.figure(figsize=(8,5))
    plt.plot(np.linspace(0,1,100),
             normalized_sigmoid_fkt(a, b, np.linspace(0,1,100))*time_between_service, label='$t_{agent}$')
    plt.axvspan(working_space[0], working_space[1], alpha=0.4, color='green', label='Рабочее время')
    plt.title(r'Зависимость времени агентского обслуживания $t_{agent}$ от суточного времени')
    plt.legend()
    plt.show()